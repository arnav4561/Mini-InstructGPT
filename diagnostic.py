# ============================================================
# Diagnostic — check if chosen/rejected labels are flipped
# ============================================================

import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("./reward_model")
tokenizer.pad_token = tokenizer.eos_token

reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model")
reward_model.eval()

# Load a few real samples and check scores
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

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
filtered = dataset.filter(is_clean)

print("Checking 10 real samples from dataset:\n")
correct = 0
for i in range(10):
    sample = filtered[5000 + i]  # use samples outside training set

    chosen_tokens = tokenizer(
        sample["chosen"], truncation=True, max_length=128,
        padding="max_length", return_tensors="pt"
    )
    rejected_tokens = tokenizer(
        sample["rejected"], truncation=True, max_length=128,
        padding="max_length", return_tensors="pt"
    )

    with torch.no_grad():
        chosen_score   = reward_model(**chosen_tokens).logits.item()
        rejected_score = reward_model(**rejected_tokens).logits.item()

    passed = chosen_score > rejected_score
    correct += int(passed)
    print(f"Sample {i+1}: chosen={chosen_score:.3f} | rejected={rejected_score:.3f} | {'✅' if passed else '❌'}")
    print(f"  CHOSEN:   {sample['chosen'][-100:].strip()}")
    print(f"  REJECTED: {sample['rejected'][-100:].strip()}")
    print()

print(f"Correct: {correct}/10")
print(f"\nIf consistently ❌: labels are likely flipped during training")
print(f"If mixed ✅/❌: model just needs more training")
