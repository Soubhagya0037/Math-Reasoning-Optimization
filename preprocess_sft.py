from datasets import load_dataset
import random, json, os, wandb
import re

wandb.init(project="reasoning-metamath", job_type="data")

ds = load_dataset("meta-math/MetaMathQA", split="train")
ds = ds.shuffle(seed=42).select(range(10000))

# Filter to keep only query and response columns, then rename them
ds = ds.select_columns(["query", "response"])
ds = ds.rename_column("query", "question")
ds = ds.rename_column("response", "answer")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

artifact = wandb.Artifact("metamath_10k", type="dataset")
with open("data/metamath_10k.jsonl", "w") as f:
    for row in ds:
        json.dump(row, f)
        f.write("\n")
artifact.add_file("data/metamath_10k.jsonl")
wandb.log_artifact(artifact)
wandb.finish()


