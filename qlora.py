import torch, datasets, wandb, os
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# ---------- 1. Logging ----------
wandb.init(
    project="reasoning-metamath",
    name="qlora-sft-10k_prod",
    tags=["sft", "qlora", "nf4", "dynamic4bit", "8bitadam"],
)

# ---------- 2. Production-grade config ----------
max_seq_len        = 1024
load_in_4bit       = True
dynamic_4bit       = True          # unsloth dynamic 4-bit
use_double_quant   = True          # double quantization
use_gradient_checkpointing = True
lora_r, lora_alpha = 16, 32        # bigger LoRA rank for 8B
lora_dropout       = 0.05
optim_name         = "paged_adamw_8bit"   # 8-bit Adam via bitsandbytes
mixed_precision    = "bf16"
batch_size         = 1
grad_acc_steps     = 32            # gives global batch ≈ 32
max_steps          = 60            # quick demo; set higher for real run
warmup_ratio       = 0.05
lr                 = 2e-4
save_dir           = "checkpoints/qlora_sft"

# ---------- 3. Load model ----------
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_len,
    load_in_4bit=load_in_4bit,
    fast_inference=True,
    dynamic_4bit=dynamic_4bit,
    use_double_quant=use_double_quant,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=lora_dropout,
    use_gradient_checkpointing=use_gradient_checkpointing,
)

# ---------- 4. Dataset ----------
EOS_TOKEN = tokenizer.eos_token
PROMPT = """<|system|>
You are a helpful math tutor. Solve the following problem step-by-step with clear reasoning.
<|user|>
{}
<|assistant|>
{}"""

def formatting_prompts_func(examples):
    texts = [
        PROMPT.format(q, a) + EOS_TOKEN
        for q, a in zip(examples["question"], examples["answer"])
    ]
    return {"text": texts}

ds = datasets.load_dataset("json", data_files="data/metamath_10k.jsonl", split="train")
ds = ds.map(formatting_prompts_func, batched=True)

# ---------- 5. Trainer ----------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=max_seq_len,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=max_steps // 5,
        output_dir="outputs",
        report_to="wandb",
        optim=optim_name,
        fp16=False,
        bf16=(mixed_precision == "bf16"),
        remove_unused_columns=False,
        gradient_checkpointing=use_gradient_checkpointing,
        dataloader_pin_memory=False,   # saves a bit of VRAM
    ),
)

trainer.train()
trainer.save_model(save_dir)
wandb.finish()