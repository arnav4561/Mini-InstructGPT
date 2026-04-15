# ============================================================
# Mini InstructGPT — Phase 1: Setup & Data Exploration
# Optimized for: NVIDIA RTX 3050 Laptop (4GB VRAM), Windows
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ── 2. Verify GPU ────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Expected:
# Using device: cuda
# GPU: NVIDIA GeForce RTX 3050 Laptop GPU
# VRAM available: 4.0 GB


# ── 3. Load GPT-2 ────────────────────────────────────────────
# We use float16 (half precision) to save VRAM on your 4GB GPU
# float32 (default) uses 2x the memory — we can't afford that here

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # half precision — saves ~250MB VRAM
).to(device)

# GPT-2 has no pad token by default — we reuse eos token as pad
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print(f"\nModel loaded: {MODEL_NAME}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"VRAM used after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# Expected: ~0.25 GB — GPT-2 is small, plenty of room left


# ── 4. Load Anthropic HH-RLHF Dataset ───────────────────────
# 🧠 YOU should understand this structure deeply!
# Each sample has:
#   "chosen"   → full conversation where assistant was helpful
#   "rejected" → full conversation where assistant was harmful/unhelpful
# The model learns to prefer "chosen" over "rejected"

print("\nLoading HH-RLHF dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
print(f"Total training samples: {len(dataset):,}")


# ── 5. Explore the Data (🧠 DO THIS YOURSELF) ────────────────
# Read these outputs carefully — understand the format!

print("\n" + "="*60)
print("SAMPLE — CHOSEN (helpful assistant response):")
print("="*60)
print(dataset[0]["chosen"])

print("\n" + "="*60)
print("SAMPLE — REJECTED (unhelpful/harmful response):")
print("="*60)
print(dataset[0]["rejected"])

# 🧠 Answer these before moving to Phase 2:
# Q1: What does the conversation format look like?
#     (How are Human and Assistant turns separated?)
# Q2: How does "chosen" differ from "rejected"?
#     Is the difference subtle or obvious?
# Q3: Where does the Human turn end and Assistant turn begin?
#     This matters for tokenization in Phase 2!


# ── 6. Tokenization ──────────────────────────────────────────
# 🧠 Understand this — interviewers ask about tokenization!
# Key decision: MAX_LENGTH=512 (GPT-2 supports 1024 max)
# Why 512? To stay within 4GB VRAM during training

MAX_LENGTH = 512

def tokenize_sample(sample):
    """
    Tokenizes both chosen and rejected responses.
    Used in Phase 3 (Reward Model) for pairwise comparison.

    🧠 Key question: why padding="max_length"?
    → Batches need equal length tensors. Padding fills shorter
      sequences with pad tokens up to MAX_LENGTH.
    """
    chosen_tokens = tokenizer(
        sample["chosen"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    rejected_tokens = tokenizer(
        sample["rejected"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "chosen_input_ids":          chosen_tokens["input_ids"].squeeze(),
        "chosen_attention_mask":     chosen_tokens["attention_mask"].squeeze(),
        "rejected_input_ids":        rejected_tokens["input_ids"].squeeze(),
        "rejected_attention_mask":   rejected_tokens["attention_mask"].squeeze(),
    }

# Tokenize a small subset first to verify everything works
print("\nTokenizing a small subset (1000 samples)...")
small_dataset = dataset.select(range(1000))
tokenized_dataset = small_dataset.map(tokenize_sample, batched=False)
tokenized_dataset.set_format(type="torch")

print(f"Sample keys: {list(tokenized_dataset[0].keys())}")
print(f"chosen_input_ids shape: {tokenized_dataset[0]['chosen_input_ids'].shape}")
# Expected: torch.Size([512])


# ── 7. Baseline Test — GPT-2 BEFORE any fine-tuning ─────────
# 🧠 SAVE this output! After RLHF training, you'll compare
#    to show how much the model improved. This is your baseline.

print("\n" + "="*60)
print("BASE GPT-2 OUTPUT (no fine-tuning):")
print("="*60)

prompt = "Human: What is the capital of France?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
# 🧠 Notice: base GPT-2 doesn't follow instructions well.
#    It might continue the text randomly instead of answering.
#    Your job over Phases 2-4 is to fix exactly this!


# ── 8. Save for Phase 2 ──────────────────────────────────────
tokenized_dataset.save_to_disk("hh_rlhf_tokenized")
print("\n✅ Phase 1 complete! Dataset saved to: hh_rlhf_tokenized/")
print("Next: Phase 2 — Supervised Fine-Tuning (SFT)")


# ============================================================
# 🧠 BEFORE moving to Phase 2, make sure you can answer:
#
# 1. Why do we set pad_token = eos_token for GPT-2?
# 2. Why torch_dtype=float16? What's the tradeoff?
# 3. Why does base GPT-2 respond poorly to instructions?
# 4. What is the structural difference between
#    chosen and rejected samples in HH-RLHF?
# ============================================================
