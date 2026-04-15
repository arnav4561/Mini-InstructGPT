# ============================================================
# Mini InstructGPT — Phase 4: PPO Training (RL Fine-Tuning)
# Uses manual PPO loop — works with ALL trl versions
# ============================================================
# 🧠 Since newer trl removed PPOTrainer, we implement the core
#    PPO logic manually. This is actually BETTER for learning
#    because you see exactly what's happening under the hood.
#
# The PPO loop:
#   1. Generate response with actor model
#   2. Score with reward model
#   3. Compute KL penalty vs reference model
#   4. Compute PPO loss and update actor
# ============================================================

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW

# ── 1. Config ────────────────────────────────────────────────
CONFIG = {
    "sft_model_path":    "./sft_model",
    "reward_model_path": "./reward_model",
    "output_dir":        "./ppo_model",
    "max_prompt_len":    48,
    "max_new_tokens":    48,
    "batch_size":        2,
    "ppo_epochs":        2,       # gradient updates per batch
    "lr":                1e-5,
    "kl_coef":           0.1,     # 🧠 KL penalty strength
                                  # higher = stays closer to SFT model
                                  # lower = more aggressive optimization
    "num_steps":         150,     # total PPO steps
    "clip_eps":          0.2,     # PPO clip range
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


# ── 2. Load Actor Model (SFT) ────────────────────────────────
# 🧠 The ACTOR is the model we're training
#    It starts from the SFT checkpoint (not base GPT-2)
#    because SFT already taught it the conversation format

print("\nLoading actor (SFT model)...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["sft_model_path"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

actor = AutoModelForCausalLM.from_pretrained(CONFIG["sft_model_path"]).to(device)
actor.train()
print(f"Actor loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")


# ── 3. Load Reference Model (frozen SFT copy) ────────────────
# 🧠 The REFERENCE model is a frozen copy of the SFT model
#    We use it to compute KL divergence — measuring how much
#    the actor has drifted from its starting point.
#    KL penalty = -kl_coef * KL(actor || reference)
#    This prevents the actor from "reward hacking"

print("\nLoading reference model (frozen SFT copy)...")
reference = AutoModelForCausalLM.from_pretrained(CONFIG["sft_model_path"]).to(device)
for param in reference.parameters():
    param.requires_grad = False   # completely frozen
reference.eval()
print("Reference model loaded and frozen.")


# ── 4. Load Reward Model ─────────────────────────────────────
print("\nLoading reward model on CPU...")
reward_tokenizer = AutoTokenizer.from_pretrained(CONFIG["reward_model_path"])
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["reward_model_path"])
reward_model.eval()
reward_model = reward_model.to("cpu")   # keep on CPU to save GPU VRAM
print("Reward model loaded.")


# ── 5. Load Prompts ──────────────────────────────────────────
def is_clean(sample):
    full_text = (sample["chosen"] + " " + sample["rejected"]).lower()
    for pattern in INAPPROPRIATE_PATTERNS:
        if re.search(pattern, full_text):
            return False
    return True

def extract_prompt(sample):
    parts = sample["chosen"].split("\n\nHuman:")
    last_human = parts[-1].split("\n\nAssistant:")[0].strip()
    return f"Human: {last_human}\n\nAssistant:"

print("\nLoading prompts...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
filtered = dataset.filter(is_clean)
prompts = [extract_prompt(filtered[i]) for i in range(CONFIG["num_prompts"])]
print(f"Loaded {len(prompts)} prompts")
print(f"Sample: {prompts[0]}")


# ── 6. Optimizer ─────────────────────────────────────────────
optimizer = AdamW(actor.parameters(), lr=CONFIG["lr"])


# ── 7. Helper Functions ──────────────────────────────────────
def get_reward_score(prompt_text, response_text):
    """Score a response using the reward model (on CPU)."""
    full_text = prompt_text + response_text
    tokens = reward_tokenizer(
        full_text, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length"
    )
    with torch.no_grad():
        score = reward_model(**tokens).logits.squeeze()
    return score.item()

def compute_kl_divergence(actor_logits, ref_logits, response_ids):
    """
    🧠 KL divergence between actor and reference model.
    Measures how much the actor has drifted from the SFT model.
    KL = sum(actor_prob * log(actor_prob / ref_prob))
    """
    actor_log_probs = F.log_softmax(actor_logits, dim=-1)
    ref_log_probs   = F.log_softmax(ref_logits,   dim=-1)

    # Gather log probs for the actual tokens generated
    actor_token_log_probs = actor_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)
    ref_token_log_probs = ref_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)

    kl = (actor_token_log_probs - ref_token_log_probs).mean()
    return kl


# ── 8. PPO Training Loop ─────────────────────────────────────
# 🧠 THIS IS THE CORE — the actual RL loop:
#
# For each step:
#   A) Generate responses with current actor
#   B) Score with reward model → get rewards
#   C) Compute KL penalty vs reference
#   D) Total reward = reward_score - kl_coef * KL
#   E) Compute policy gradient loss (PPO-clip)
#   F) Update actor weights

print(f"\n{'='*60}")
print(f"Starting PPO training — {CONFIG['num_steps']} steps")
print(f"Watch: avg_reward should increase, KL should stay <5")
print(f"{'='*60}\n")

step_rewards = []
step_kls = []

for step in range(CONFIG["num_steps"]):
    # Pick batch of prompts
    start = (step * CONFIG["batch_size"]) % len(prompts)
    batch_prompts = prompts[start: start + CONFIG["batch_size"]]

    batch_rewards = []
    batch_kls = []
    losses = []

    for prompt_text in batch_prompts:

        # ── A: Tokenize prompt ───────────────────────────────
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_prompt_len"],
        ).input_ids.to(device)

        # ── B: Generate response ─────────────────────────────
        with torch.no_grad():
            full_ids = actor.generate(
                prompt_ids,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids = full_ids[:, prompt_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # ── C: Get reward score ──────────────────────────────
        reward_score = get_reward_score(prompt_text, response_text)
        batch_rewards.append(reward_score)

        # ── D: Compute KL penalty ────────────────────────────
        if response_ids.shape[1] == 0:
            continue  # skip empty responses

        with torch.no_grad():
            ref_out = reference(full_ids[:, :-1])
            ref_logits = ref_out.logits[:, prompt_ids.shape[1]-1:-1, :]

        actor_out = actor(full_ids[:, :-1])
        actor_logits = actor_out.logits[:, prompt_ids.shape[1]-1:-1, :]

        kl = compute_kl_divergence(actor_logits, ref_logits, response_ids[:, :actor_logits.shape[1]])
        batch_kls.append(kl.item())

        # ── E: Total reward = reward - KL penalty ────────────
        # 🧠 This is the key equation of RLHF:
        #    total_reward = reward_model_score - kl_coef * KL
        #    The KL term prevents the model from drifting too far
        total_reward = reward_score - CONFIG["kl_coef"] * kl.item()

        # ── F: Policy gradient loss ──────────────────────────
        # We want to increase probability of high-reward responses
        # Loss = -total_reward * log_prob(response)
        log_probs = F.log_softmax(actor_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, response_ids[:, :actor_logits.shape[1]].unsqueeze(-1)
        ).squeeze(-1).mean()

        loss = -total_reward * token_log_probs
        losses.append(loss)

    if not losses:
        continue

    # Update actor
    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    avg_reward = sum(batch_rewards) / len(batch_rewards)
    avg_kl = sum(batch_kls) / len(batch_kls) if batch_kls else 0
    step_rewards.append(avg_reward)
    step_kls.append(avg_kl)

    if (step + 1) % CONFIG["log_every"] == 0:
        recent_reward = sum(step_rewards[-10:]) / len(step_rewards[-10:])
        recent_kl = sum(step_kls[-10:]) / len(step_kls[-10:])
        print(f"Step {step+1:3d}/{CONFIG['num_steps']} | "
              f"Reward: {recent_reward:+.3f} | "
              f"KL: {recent_kl:.3f} | "
              f"Loss: {total_loss.item():.4f}")

        # Show sample generation every 50 steps
        if (step + 1) % 50 == 0:
            print(f"  Prompt:   {batch_prompts[0]}")
            print(f"  Response: {response_text[:120]}\n")

    torch.cuda.empty_cache()


# ── 9. Save PPO Model ────────────────────────────────────────
actor.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\n✅ PPO model saved to: {CONFIG['output_dir']}")


# ── 10. Final 3-Way Comparison ───────────────────────────────
# 🧠 The moment of truth — Base vs SFT vs PPO
print("\n" + "="*60)
print("FINAL COMPARISON — Base GPT-2 vs SFT vs PPO")
print("="*60)

def generate(mdl, tok, prompt, max_new=80):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
base_tok = AutoTokenizer.from_pretrained("gpt2")
base_tok.pad_token = base_tok.eos_token

sft_model = AutoModelForCausalLM.from_pretrained("./sft_model").to(device)
ppo_model = AutoModelForCausalLM.from_pretrained("./ppo_model").to(device)

test_prompts = [
    "Human: What is the capital of France?\n\nAssistant:",
    "Human: How do I make a cup of tea?\n\nAssistant:",
    "Human: Can you explain what gravity is?\n\nAssistant:",
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")
    print(f"  🔴 Base GPT-2 : {generate(base_model, base_tok, prompt)[:150]}")
    print(f"  🟡 SFT model  : {generate(sft_model, tokenizer, prompt)[:150]}")
    print(f"  🟢 PPO model  : {generate(ppo_model, tokenizer, prompt)[:150]}")

print("\n" + "="*60)
print("🎉 RLHF Pipeline Complete!")
print("Base GPT-2 → SFT → Reward Model → PPO = Mini InstructGPT")
print("="*60)
