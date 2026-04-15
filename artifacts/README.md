# Artifacts

This folder contains evaluation outputs exported from the Kaggle runs.

## Files

- `eval_100_prompts.jsonl` - benchmark prompt set used for both models
- `outputs_7b_dpo.jsonl` - generated outputs for Qwen2.5-7B DPO
- `outputs_gpt2_ppo_v2.jsonl` - generated outputs for GPT-2 PPO v2
- `comparison_full_100.csv` - merged comparison for all 100 prompts
- `comparison_sample_20.csv` - 20-row sample from the full comparison

## Reported Metrics

- Qwen (7B DPO) deflection rate: `2.0%`
- GPT-2 PPO v2 deflection rate: `88.0%`
- Qwen (7B DPO) direct-answer rate: `97.0%`
- GPT-2 PPO v2 direct-answer rate: `1.0%`
