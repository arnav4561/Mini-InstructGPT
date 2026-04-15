# ============================================================
# Mini InstructGPT — Phase 3: Reward Model Training (v4)
# Key fix: train on LAST ASSISTANT TURN only, not full conversation
# Why: chosen vs rejected only differ at the last assistant turn.
#      Truncating full conversations made both look identical.
# ============================================================

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ── 1. Config ────────────────────────────────────────────────
CONFIG = {
    "model_name":    "gpt2",
    "output_dir":    "./reward_model",
    "max_length":    128,
    "batch_size":    4,
    "grad_accum":    4,
    "epochs":        3,
    "lr":            1e-5,
    "warmup_steps":  50,
    "num_train":     5000,
    "num_eval":      200,
    "log_every":     20,
    "save_every":    200,
}

device = "cpu"
print(f"Using device: {device}")


# ── 2. Load Model ────────────────────────────────────────────
print("\nLoading reward model...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = CONFIG["max_length"]

reward_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=1,
)
reward_model.config.pad_token_id = tokenizer.eos_token_id
reward_model = reward_model.to(device)
print("Reward model loaded.")


# ── 3. Load & Filter Dataset ─────────────────────────────────
INAPPROPRIATE_PATTERNS = [
    r'\bfuck\b', r'\bshit\b', r'\bcunt\b', r'\bfucker\b',
    r'\bporn\b', r'\bnude\b', r'\bsuicide\b', r'\bcum\b',
    r'\bcocaine\b', r'\bheroin\b', r'\brake\b', r'\bdick\b',
    r'cuss word', r'swear word', r'\bwhore\b', r'\bslut\b',
]

def is_clean(sample):
    full_text = (sample["chosen"] + " " + sample["rejected"]).lower()
    for pattern in INAPPROPRIATE_PATTERNS:
        if re.search(pattern, full_text):
            return False
    return True

def extract_last_turn(text):
    """
    🧠 KEY FIX: Extract only the last assistant turn.
    chosen vs rejected ONLY differ at the last assistant response.
    Training on full conversations caused truncation to make
    both chosen and rejected look identical (same first 128 tokens).
    Now we extract just the part that's actually different.
    """
    parts = text.split("\n\nAssistant:")
    if len(parts) > 1:
        # Get last human turn + last assistant response
        last_human = parts[-2].split("\n\nHuman:")[-1].strip()
        last_assistant = parts[-1].strip()
        return f"Human: {last_human}\n\nAssistant: {last_assistant}"
    return text

print("\nLoading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
filtered = dataset.filter(is_clean)
train_data = filtered.select(range(CONFIG["num_train"]))
eval_data  = filtered.select(range(CONFIG["num_train"], CONFIG["num_train"] + CONFIG["num_eval"]))
print(f"Train: {len(train_data):,} | Eval: {len(eval_data):,}")

# Quick sanity check on extraction
sample = train_data[0]
print(f"\nExtracted last turn example:")
print(f"CHOSEN:   {extract_last_turn(sample['chosen'])[:200]}")
print(f"REJECTED: {extract_last_turn(sample['rejected'])[:200]}")
print()


# ── 4. Tokenize — Last Turn Only ─────────────────────────────
def tokenize_pair(sample):
    chosen_text   = extract_last_turn(sample["chosen"])
    rejected_text = extract_last_turn(sample["rejected"])

    chosen = tokenizer(
        chosen_text,
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
        return_tensors="pt"
    )
    rejected = tokenizer(
        rejected_text,
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "chosen_input_ids":        chosen["input_ids"].squeeze(),
        "chosen_attention_mask":   chosen["attention_mask"].squeeze(),
        "rejected_input_ids":      rejected["input_ids"].squeeze(),
        "rejected_attention_mask": rejected["attention_mask"].squeeze(),
    }

print("Tokenizing last turns only...")
tok_train = train_data.map(tokenize_pair, remove_columns=["chosen","rejected"], batched=False)
tok_eval  = eval_data.map(tokenize_pair,  remove_columns=["chosen","rejected"], batched=False)
tok_train.set_format(type="torch")
tok_eval.set_format(type="torch")

train_loader = DataLoader(tok_train, batch_size=CONFIG["batch_size"], shuffle=True,  pin_memory=False)
eval_loader  = DataLoader(tok_eval,  batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=False)

total_steps = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
print(f"Total optimizer steps: {total_steps}")


# ── 5. Optimizer & Scheduler ─────────────────────────────────
optimizer = AdamW(reward_model.parameters(), lr=CONFIG["lr"])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CONFIG["warmup_steps"],
    num_training_steps=total_steps
)


# ── 6. Eval Function ─────────────────────────────────────────
def evaluate():
    reward_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            chosen_score = reward_model(
                input_ids=batch["chosen_input_ids"].to(device),
                attention_mask=batch["chosen_attention_mask"].to(device)
            ).logits
            rejected_score = reward_model(
                input_ids=batch["rejected_input_ids"].to(device),
                attention_mask=batch["rejected_attention_mask"].to(device)
            ).logits
            correct += (chosen_score > rejected_score).sum().item()
            total += chosen_score.shape[0]
    reward_model.train()
    return correct / total


# ── 7. Training Loop ─────────────────────────────────────────
print("\nStarting training...")
print("~30-45 mins on CPU. ☕")

reward_model.train()
global_step = 0
accum_loss = 0.0
optimizer.zero_grad()

for epoch in range(CONFIG["epochs"]):
    print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
    for step, batch in enumerate(train_loader):

        chosen_score = reward_model(
            input_ids=batch["chosen_input_ids"].to(device),
            attention_mask=batch["chosen_attention_mask"].to(device)
        ).logits

        rejected_score = reward_model(
            input_ids=batch["rejected_input_ids"].to(device),
            attention_mask=batch["rejected_attention_mask"].to(device)
        ).logits

        # Bradley-Terry Loss
        loss = -torch.nn.functional.logsigmoid(
            chosen_score - rejected_score
        ).mean() / CONFIG["grad_accum"]

        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % CONFIG["grad_accum"] == 0:
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % CONFIG["log_every"] == 0:
                avg_loss = accum_loss / CONFIG["log_every"]
                print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")
                accum_loss = 0.0

            if global_step % CONFIG["save_every"] == 0:
                reward_model.save_pretrained(f"{CONFIG['output_dir']}_step{global_step}")
                print(f"💾 Checkpoint saved at step {global_step}")

    acc = evaluate()
    print(f"✅ Epoch {epoch+1} accuracy: {acc*100:.1f}% (target: >60%)")


# ── 8. Save ──────────────────────────────────────────────────
reward_model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\n✅ Reward model saved to: {CONFIG['output_dir']}")

# ── 9. Final Accuracy ────────────────────────────────────────
print("\n" + "="*60)
final_acc = evaluate()
print(f"FINAL ACCURACY: {final_acc*100:.1f}%")
if final_acc >= 0.65:
    print("🎉 Reward model working well — ready for Phase 4!")
elif final_acc >= 0.55:
    print("⚠️  Partial success — good enough for Phase 4 demo.")
else:
    print("❌ Still struggling — but let's move to Phase 4 anyway.")
    print("   A weak reward model will still demonstrate the pipeline.")

print("\nNext: Phase 4 — PPO Training (RL Fine-Tuning)")
