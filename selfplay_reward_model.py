# ============================================================
# Mini InstructGPT v2 — Self-Play Reward Model (v3)
# Fix: completely isolate CPU training from CUDA context
# Resumes from step 200 checkpoint automatically
# ============================================================

import os
# Completely hide CUDA from the training process
# This prevents AdamW from touching CUDA context during CPU training
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ── 1. Config ────────────────────────────────────────────────
CONFIG = {
    "sft_model_path":        "/kaggle/input/datasets/arnavx/mini-instructgpt-models/sft_model_v2/sft_model_v2",
    "checkpoint_path":       "/kaggle/input/datasets/arnavx/mini-instructgpt-models/reward_model_v2/reward_model_v2",  # resume from here
    "output_dir":            "/kaggle/working/reward_model_v3",
    "max_length":            128,
    "batch_size":            2,
    "grad_accum":            8,
    "epochs":                3,
    "lr":                    1e-5,
    "warmup_steps":          50,
    "num_hh_samples":        5000,
    "num_eval":              200,
    "log_every":             20,
    "save_every":            200,
    "selfplay_cache":        "./selfplay_pairs.json",
    "resume_from_step":      0,   # skip first 200 steps already trained
}

INAPPROPRIATE_PATTERNS = [
    r'\bfuck\b', r'\bshit\b', r'\bcunt\b', r'\bfucker\b',
    r'\bporn\b', r'\bnude\b', r'\bsuicide\b', r'\bcum\b',
    r'\bcocaine\b', r'\bheroin\b', r'\brake\b', r'\bdick\b',
    r'cuss word', r'swear word', r'\bwhore\b', r'\bslut\b',
]

# Force CPU — GPU completely hidden
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── 2. Load Reward Model from Checkpoint ─────────────────────
# Resume from step 200 checkpoint — don't retrain from scratch!
print(f"\nLoading reward model from checkpoint: {CONFIG['checkpoint_path']}")
reward_tokenizer = AutoTokenizer.from_pretrained(CONFIG["sft_model_path"])
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_tokenizer.model_max_length = CONFIG["max_length"]

reward_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["checkpoint_path"],
    num_labels=1,
)
reward_model.config.pad_token_id = reward_tokenizer.eos_token_id
reward_model = reward_model.to(device)
print("Resumed from checkpoint. Previous training not wasted! ✅")


# ── 3. Load Self-Play + HH-RLHF Data ─────────────────────────
def is_clean(sample):
    full_text = (sample["chosen"] + " " + sample["rejected"]).lower()
    return not any(re.search(p, full_text) for p in INAPPROPRIATE_PATTERNS)

def extract_last_turn(text):
    parts = text.split("\n\nAssistant:")
    if len(parts) > 1:
        last_human = parts[-2].split("\n\nHuman:")[-1].strip()
        last_assistant = parts[-1].strip()
        return f"Human: {last_human}\n\nAssistant: {last_assistant}"
    return text

print("\nLoading mixed reward training pairs...")
dataset = load_dataset("json", data_files={"train": "/kaggle/working/reward_train_mix.jsonl"}, split="train")

all_pairs = [{"chosen": x["chosen"], "rejected": x["rejected"]} for x in dataset]

import random
random.seed(42)
random.shuffle(all_pairs)

print(f"Total: {len(all_pairs):,} pairs")


# ── 4. Tokenize ──────────────────────────────────────────────
def tokenize_pair(sample):
    chosen = reward_tokenizer(
        sample["chosen"], truncation=True,
        max_length=CONFIG["max_length"], padding="max_length", return_tensors="pt"
    )
    rejected = reward_tokenizer(
        sample["rejected"], truncation=True,
        max_length=CONFIG["max_length"], padding="max_length", return_tensors="pt"
    )
    return {
        "chosen_input_ids":        chosen["input_ids"].squeeze(),
        "chosen_attention_mask":   chosen["attention_mask"].squeeze(),
        "rejected_input_ids":      rejected["input_ids"].squeeze(),
        "rejected_attention_mask": rejected["attention_mask"].squeeze(),
    }

print("Tokenizing...")
train_pairs = all_pairs[:-CONFIG["num_eval"]]
eval_pairs  = all_pairs[-CONFIG["num_eval"]:]

tok_train = Dataset.from_list(train_pairs).map(tokenize_pair, remove_columns=["chosen","rejected"], batched=False)
tok_eval  = Dataset.from_list(eval_pairs).map(tokenize_pair,  remove_columns=["chosen","rejected"], batched=False)
tok_train.set_format(type="torch")
tok_eval.set_format(type="torch")

train_loader = DataLoader(tok_train, batch_size=CONFIG["batch_size"], shuffle=True,  pin_memory=False)
eval_loader  = DataLoader(tok_eval,  batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=False)

total_steps = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
remaining_steps = total_steps - CONFIG["resume_from_step"]
print(f"Total steps: {total_steps} | Remaining: {remaining_steps}")


# ── 5. Optimizer ─────────────────────────────────────────────
optimizer = AdamW(reward_model.parameters(), lr=CONFIG["lr"])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CONFIG["warmup_steps"],
    num_training_steps=total_steps
)

# Fast-forward scheduler to match checkpoint step
for _ in range(CONFIG["resume_from_step"]):
    scheduler.step()


# ── 6. Eval ──────────────────────────────────────────────────
def evaluate():
    reward_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            cs = reward_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            ).logits
            rs = reward_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            ).logits

            correct += (cs > rs).sum().item()
            total += cs.shape[0]
    reward_model.train()
    return correct / total


# ── 7. Training Loop ─────────────────────────────────────────
print(f"\nResuming training from step {CONFIG['resume_from_step']}...")
print(f"Training on {device}.")

reward_model.train()
global_step = CONFIG["resume_from_step"]
accum_loss = 0.0
optimizer.zero_grad()
steps_done_this_run = 0

for epoch in range(CONFIG["epochs"]):
    print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
    for step, batch in enumerate(train_loader):

        # Skip steps already trained in previous run
        if global_step - CONFIG["resume_from_step"] < 0:
            global_step += 1
            continue

        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        cs = reward_model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        ).logits
        rs = reward_model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        ).logits

        loss = -torch.nn.functional.logsigmoid(
            cs - rs
        ).mean() / CONFIG["grad_accum"]

        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % CONFIG["grad_accum"] == 0:
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            steps_done_this_run += 1

            if global_step % CONFIG["log_every"] == 0:
                print(f"Step {global_step}/{total_steps} | Loss: {accum_loss/CONFIG['log_every']:.4f}")
                accum_loss = 0.0

            if global_step % CONFIG["save_every"] == 0:
                reward_model.save_pretrained(f"{CONFIG['output_dir']}_step{global_step}")
                print(f"💾 Checkpoint saved at step {global_step}")

    acc = evaluate()
    print(f"✅ Epoch {epoch+1} accuracy: {acc*100:.1f}% (v1 was 63%)")


# ── 8. Save ──────────────────────────────────────────────────
reward_model.save_pretrained(CONFIG["output_dir"])
reward_tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\n✅ Reward model v2 saved to: {CONFIG['output_dir']}")

final_acc = evaluate()
print(f"Final accuracy: {final_acc*100:.1f}%")
if final_acc > 0.63:
    print("🎉 Better than v1! Self-play helped.")
elif final_acc >= 0.58:
    print("⚠️  Similar to v1 — still good enough for PPO.")
else:
    print("❌ Needs more data — pipeline still works though.")

print("\nNext: python phase4_ppo_v2.py")