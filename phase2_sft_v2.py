# ============================================================
# Mini InstructGPT v2 — Phase 2: SFT on GPT-2 Medium (345M)
# Uses plain PyTorch loop — no SFTTrainer (avoids CUDA bug)
# ============================================================

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# ── 1. Config ────────────────────────────────────────────────
CONFIG = {
    "model_name":    "gpt2-medium",
    "output_dir":    "./sft_model_v2",
    "max_length":    256,      # reduced from 512 to save VRAM
    "batch_size":    1,
    "grad_accum":    16,       # effective batch = 16
    "epochs":        1,
    "lr":            2e-5,
    "warmup_steps":  100,
    "num_samples":   20000,
    "log_every":     50,
    "save_every":    500,
}

# ── 2. Load Model ────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = CONFIG["max_length"]

print(f"\nLoading {CONFIG['model_name']}...")
model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"]).to(device)
model.config.pad_token_id = tokenizer.eos_token_id
model.gradient_checkpointing_enable()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ── 3. Load & Filter Dataset ─────────────────────────────────
INAPPROPRIATE_PATTERNS = [
    r'\bfuck\b', r'\bshit\b', r'\bcunt\b', r'\bfucker\b',
    r'\bporn\b', r'\bnude\b', r'\bsuicide\b', r'\bcum\b',
    r'\bcocaine\b', r'\bheroin\b', r'\brake\b', r'\bdick\b',
    r'cuss word', r'swear word', r'\bwhore\b', r'\bslut\b',
]

def is_clean(sample):
    full_text = (sample["chosen"] + " " + sample["rejected"]).lower()
    return not any(re.search(p, full_text) for p in INAPPROPRIATE_PATTERNS)

print("\nLoading and filtering dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
filtered = dataset.filter(is_clean)
train_data = filtered.select(range(CONFIG["num_samples"]))
print(f"Training on: {len(train_data):,} samples")


# ── 4. Tokenize ──────────────────────────────────────────────
def tokenize(sample):
    tokens = tokenizer(
        sample["chosen"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing...")
tokenized = train_data.map(tokenize, remove_columns=["chosen", "rejected"], batched=False)
tokenized.set_format(type="torch")

dataloader = DataLoader(
    tokenized,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    pin_memory=False,
)
total_steps = (len(dataloader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
print(f"Total optimizer steps: {total_steps}")


# ── 5. Optimizer & Scheduler ─────────────────────────────────
optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CONFIG["warmup_steps"],
    num_training_steps=total_steps
)


# ── 6. Training Loop — Plain PyTorch ─────────────────────────
print(f"\nStarting SFT training on {CONFIG['model_name']}...")
print("~60-90 minutes on RTX 3050. ☕☕")

model.train()
global_step = 0
accum_loss = 0.0
optimizer.zero_grad()

for epoch in range(CONFIG["epochs"]):
    print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
    for step, batch in enumerate(dataloader):

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / CONFIG["grad_accum"]
        loss.backward()
        accum_loss += loss.item()

        del input_ids, attention_mask, labels, outputs

        if (step + 1) % CONFIG["grad_accum"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % CONFIG["log_every"] == 0:
                avg_loss = accum_loss / CONFIG["log_every"]
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | VRAM: {vram:.2f}GB")
                accum_loss = 0.0

            if global_step % CONFIG["save_every"] == 0:
                model.save_pretrained(f"{CONFIG['output_dir']}_step{global_step}")
                print(f"💾 Checkpoint saved at step {global_step}")


# ── 7. Save ──────────────────────────────────────────────────
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\n✅ SFT v2 model saved to: {CONFIG['output_dir']}")


# ── 8. Quick Test ────────────────────────────────────────────
print("\n" + "="*60)
print("GPT-2 MEDIUM SFT OUTPUT:")
print("="*60)

model.eval()
prompt = "Human: What is the capital of France?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\nNext: Self-Play Reward Model")
