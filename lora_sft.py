import torch, transformers, datasets, wandb
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
wandb.init(project="reasoning-metamath", name="lora-sft-10k", tags=["sft"])

model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)
PROMPT = """<|system|>
You are a helpful math tutor. Solve the following problem step-by-step with clear reasoning.
<|user|>
{}
<|assistant|>
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        # use the actual answer format from your dataset
        text = PROMPT.format(question, answer) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# dataset
ds = datasets.load_dataset("json", data_files="data/metamath_10k.jsonl", split="train")
ds = ds.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        report_to="wandb",   # 👈 flip to W&B
        remove_unused_columns=False,  # safety
    ),
)
trainer.train()
trainer.save_model("checkpoints/lora_sft")
wandb.finish()