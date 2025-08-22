import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel
from utils import xmlcount_reward_func,soft_format_reward_func,strict_format_reward_func,int_reward_func,correctness_reward_func
from datasets import load_dataset

wandb.init(project="reasoning-metamath", name="grpo-rl-2k", tags=["rl"])

base_model, tokenizer = FastLanguageModel.from_pretrained(
    "checkpoints/lora_sft",
    max_seq_length=1024,
    load_in_4bit=True,
    #fast_inference=True,
)
#base_model.load_adapter("checkpoints/lora_sft")   # warm start

ds = load_dataset("json", data_files="try_grpo_dataset.json", split="train")

grpo_config = GRPOConfig(
    output_dir="checkpoints/lora_grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    max_steps=60,
    learning_rate=5e-6,
    report_to="wandb",
    logging_steps=1,
)

trainer = GRPOTrainer(
    model=base_model,
    processing_class=tokenizer,
    reward_funcs=[xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    train_dataset=ds,
    args=grpo_config,
)
trainer.train()
trainer.save_model("checkpoints/lora_grpo")

wandb.finish()