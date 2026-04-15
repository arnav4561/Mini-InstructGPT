# ============================================================
# Mini InstructGPT v2 — Phase 4: PPO Training
# Fix: optimizer states on CPU (saves ~1.4GB VRAM)
# GPT-2 Medium is too large for full GPU training on 4GB
# ============================================================

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW

# ── 1. Config ────────────────────────────────────────────────
CONFIG = {
    "sft_model_path":    "/kaggle/input/datasets/arnavx/mini-instructgpt-models/sft_model_v2/sft_model_v2",
    "reward_model_path": "/kaggle/input/datasets/arnavx/mini-instructgpt-models/reward_model_v2/reward_model_v2",
    "output_dir":        "/kaggle/working/ppo_model_v2",
    "max_prompt_len":    96,
    "max_new_tokens":    48,
    "batch_size":        1,
    "lr":                1e-6,
    "kl_coef":           0.7,
    "deflection_penalty_coef": 0.4,
    "num_steps":         120,
    "log_every":         10,
    "num_prompts":       500,
}

INAPPROPRIATE_PATTERNS = [
    r'\bfuck\b', r'\bshit\b', r'\bcunt\b', r'\bfucker\b',
    r'\bporn\b', r'\bnude\b', r'\bsuicide\b', r'\bcum\b',
    r'\bcocaine\b', r'\bheroin\b', r'\brake\b', r'\bdick\b',
    r'cuss word', r'swear word', r'\bwhore\b', r'\bslut\b',
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.cuda.empty_cache()


# ── 2. Load Actor (GPU) with gradient checkpointing ─────────
print("\nLoading actor on GPU...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["sft_model_path"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

actor = AutoModelForCausalLM.from_pretrained(CONFIG["sft_model_path"]).to(device)
actor.gradient_checkpointing_enable()  # trades compute for memory
actor.train()
print(f"Actor loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")


# ── 3. Reference Model (CPU) ─────────────────────────────────
print("\nLoading reference model on CPU...")
reference = AutoModelForCausalLM.from_pretrained(CONFIG["sft_model_path"])
for param in reference.parameters():
    param.requires_grad = False
reference.eval()
reference = reference.to("cpu")
print(f"Reference on CPU. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")


# ── 4. Reward Model (CPU) ────────────────────────────────────
print("\nLoading reward model v2 on CPU...")
reward_tokenizer = AutoTokenizer.from_pretrained(CONFIG["reward_model_path"])
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["reward_model_path"])
reward_model.eval()
reward_model = reward_model.to("cpu")
print(f"All models loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")


# ── 5. CPU Optimizer ─────────────────────────────────────────
# 🧠 KEY FIX: move model params to CPU for optimizer
#    AdamW stores exp_avg and exp_avg_sq (same size as params)
#    For 345M model that's ~1.4GB just for optimizer states
#    By keeping params on CPU during optimizer.step() we avoid OOM

class CPUOffloadOptimizer:
    """
    Optimizer that keeps optimizer states on CPU.
    During step: move grads to CPU → update → move params back to GPU
    This saves ~1.4GB VRAM for GPT-2 Medium.
    """
    def __init__(self, model, lr):
        self.model = model
        # Create CPU copies of params for optimizer
        self.cpu_params = [
            p.detach().cpu().float().requires_grad_(True)
            for p in model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(self.cpu_params, lr=lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, gpu_params):
        # Copy gradients from GPU to CPU params
        for cpu_p, gpu_p in zip(self.cpu_params, gpu_params):
            if gpu_p.grad is not None:
                cpu_p.grad = gpu_p.grad.detach().cpu().float()

        # Clip gradients on CPU
        torch.nn.utils.clip_grad_norm_(self.cpu_params, 1.0)

        # Optimizer step on CPU
        self.optimizer.step()

        # Copy updated params back to GPU
        with torch.no_grad():
            for cpu_p, gpu_p in zip(self.cpu_params, gpu_params):
                gpu_p.copy_(cpu_p.to(device).to(gpu_p.dtype))

gpu_params = [p for p in actor.parameters() if p.requires_grad]
optimizer = CPUOffloadOptimizer(actor, lr=CONFIG["lr"])


# —— 6. Load Prompts (safe PPO prompt pool, no HH sampling) ——
from datasets import load_dataset
import random

print("\nLoading safe PPO prompts...")
d = load_dataset("databricks/databricks-dolly-15k", split="train")

SAFE_CATS = {"open_qa", "closed_qa", "information_extraction", "summarization"}

prompts = []
for x in d:
    if x["category"] in SAFE_CATS:
        q = x["instruction"].strip()
        if q:
            prompts.append(f"Human: {q}\n\nAssistant:")

random.shuffle(prompts)
prompts = prompts[:CONFIG["num_prompts"]]

if len(prompts) == 0:
    raise ValueError("No safe prompts loaded.")

print(f"Loaded {len(prompts)} safe prompts")


# ── 7. Helpers ───────────────────────────────────────────────
DEFLECTION_PATTERNS = [
    r"\bcan i ask\b",
    r"\bis there anything else\b",
    r"\bcan you clarify\b",
    r"\bwhat do you mean\b",
    r"\bi['’]?m not sure\b",
    r"\bit depends\b",
]

def deflection_penalty(response_text):
    t = response_text.lower()
    penalty = 0.0
    for p in DEFLECTION_PATTERNS:
        if re.search(p, t):
            penalty += 0.25
    if len(response_text.split()) < 8:
        penalty += 0.20
    return penalty

def get_reward_score(prompt_text, response_text):
    full_text = prompt_text + response_text
    tokens = reward_tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    with torch.no_grad():
        score = reward_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        ).logits.squeeze().item()
    return score

def compute_kl(actor_logits, ref_logits, response_ids):
    actor_lp = F.log_softmax(actor_logits.cpu().float(), dim=-1)
    ref_lp   = F.log_softmax(ref_logits.float(), dim=-1)
    actor_tlp = actor_lp.gather(-1, response_ids.cpu().unsqueeze(-1)).squeeze(-1)
    ref_tlp   = ref_lp.gather(-1, response_ids.cpu().unsqueeze(-1)).squeeze(-1)
    return (actor_tlp - ref_tlp).mean()


# ── 8. PPO Loop ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Starting PPO v2 — {CONFIG['num_steps']} steps")
print(f"Actor GPU | Reference CPU | Optimizer CPU offloaded")
print(f"{'='*60}\n")

step_rewards = []
step_kls = []

for step in range(CONFIG["num_steps"]):
    prompt_text = prompts[step % len(prompts)]

    # Tokenize
    enc = tokenizer(
      prompt_text,
      return_tensors="pt",
      truncation=True,
      max_length=CONFIG["max_prompt_len"],
    )
    prompt_ids = enc["input_ids"].to(device)
    prompt_mask = enc["attention_mask"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Generate
    with torch.no_grad():
        full_ids = actor.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = full_ids[:, prompt_len:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    if not response_text.strip() or response_ids.shape[1] == 0:
        continue

    # Score
    reward_score = get_reward_score(prompt_text, response_text)
    step_rewards.append(reward_score)

    # Reference forward on CPU
    full_attn = torch.ones_like(full_ids, device=full_ids.device)
    with torch.no_grad():
        ref_logits = reference(
            input_ids=full_ids[:, :-1].cpu(),
            attention_mask=full_attn[:, :-1].cpu(),
        ).logits[:, prompt_len - 1:-1, :]

    # Actor forward on GPU
    actor_out = actor(
        input_ids=full_ids[:, :-1],
        attention_mask=full_attn[:, :-1],
    )
    actor_logits = actor_out.logits[:, prompt_len - 1:-1, :]

    # KL on CPU
    kl = compute_kl(actor_logits, ref_logits, response_ids[:, :actor_logits.shape[1]])
    step_kls.append(kl.item())

    kl_raw = kl.item()
    kl_pen = max(0.0, kl_raw)   # prevent negative KL from boosting reward

    # Total reward
    total_reward = (
        reward_score
        - CONFIG["kl_coef"] * kl_pen
        - CONFIG["deflection_penalty_coef"] * deflection_penalty(response_text)
    )

    # Policy gradient loss
    log_probs = F.log_softmax(actor_logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1, response_ids[:, :actor_logits.shape[1]].unsqueeze(-1)
    ).squeeze(-1).mean()

    loss = -total_reward * token_log_probs

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(gpu_params)

    torch.cuda.empty_cache()

    if (step + 1) % CONFIG["log_every"] == 0:
        recent_reward = sum(step_rewards[-10:]) / len(step_rewards[-10:])
        recent_kl = sum(step_kls[-10:]) / len(step_kls[-10:])
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Step {step+1:3d}/{CONFIG['num_steps']} | "
              f"Reward: {recent_reward:+.3f} | "
              f"KL: {recent_kl:.3f} | "
              f"VRAM: {vram:.2f}GB")

        if (step + 1) % 50 == 0:
            print(f"  Prompt:   {prompt_text}")
            print(f"  Response: {response_text[:150]}\n")


# ── 9. Save ──────────────────────────────────────────────────
actor.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\n✅ PPO model v2 saved to: {CONFIG['output_dir']}")


# ── 10. Final Comparison ─────────────────────────────────────
print("\n" + "="*60)
print("FINAL COMPARISON — Base vs PPO v1 vs PPO v2")
print("="*60)

def generate(mdl, tok, prompt, max_new=60):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            use_cache=False,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

torch.cuda.empty_cache()

base_tok = AutoTokenizer.from_pretrained("gpt2")
base_tok.pad_token = base_tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

v1_tok = AutoTokenizer.from_pretrained("/kaggle/input/datasets/arnavx/mini-instructgpt-models/ppo_model/ppo_model", local_files_only=True)
v1_tok.pad_token = v1_tok.eos_token
ppo_v1 = AutoModelForCausalLM.from_pretrained("/kaggle/input/datasets/arnavx/mini-instructgpt-models/ppo_model/ppo_model", local_files_only=True).to(device)

test_prompts = [
    "Human: What is the capital of France?\n\nAssistant:",
    "Human: How do I make a cup of tea?\n\nAssistant:",
    "Human: Can you explain what gravity is?\n\nAssistant:",
    "Human: What are some tips for studying better?\n\nAssistant:",
]

for prompt in test_prompts:
    print(f"\n📝 {prompt}")
    print(f"  🔴 Base GPT-2 : {generate(base_model, base_tok, prompt)[:150]}")
    print(f"  🟡 PPO v1     : {generate(ppo_v1, v1_tok, prompt)[:150]}")
    print(f"  🟢 PPO v2     : {generate(actor, tokenizer, prompt)[:150]}")

print("\n" + "="*60)
print("🎉 Mini InstructGPT v2 Complete!")
print("="*60)
print("\nWhat you built:")
print("  ✅ GPT-2 Medium SFT (345M, 20k samples)")
print("  ✅ Self-play data generation (2000 pairs)")
print("  ✅ Reward model v2 — 76.5% accuracy (was 63%)")
print("  ✅ PPO v2 with CPU-offloaded optimizer")
print("  ✅ Novel: self-play inspired by SPIN paper")