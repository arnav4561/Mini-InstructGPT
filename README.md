# Mini-InstructGPT

End-to-end LLM alignment project: started with a GPT-2 RLHF pipeline (SFT + reward model + PPO), diagnosed reward-hacking behavior, then scaled to a 7B QLoRA + DPO setup with significantly better response quality.

## Highlights

- Built full RLHF-style pipeline from scratch with GPT-2 and iterated through PPO v2.
- Identified and mitigated reward-hacking patterns in small-model PPO training.
- Scaled to `Qwen2.5-7B-Instruct` with QLoRA + DPO on Kaggle.
- Benchmarked both systems on a fixed 100-prompt evaluation set.

## Results

| Metric | GPT-2 PPO v2 | Qwen2.5-7B DPO |
|---|---:|---:|
| Deflection rate | 88.0% | 2.0% |
| Direct-answer rate | 1.0% | 97.0% |

## Project Phases

- `Phase 1` Setup + data prep (`phase1_setup.py`)
- `Phase 2` Supervised fine-tuning (`phase2_sft_v2.py`)
- `Phase 3` Reward model training (`phase3_reward_model.py`, `selfplay_reward_model.py`)
- `Phase 4` PPO / PPO v2 (`phase4_ppo.py`, `phase4_ppo_v2.py`)
- `Scale-up` 7B QLoRA + DPO with fixed-prompt evaluation (artifacts in `artifacts/`)

## Repository Structure

- `phase1_setup.py` - setup and preprocessing helpers
- `phase2_sft_v2.py` - supervised fine-tuning
- `phase3_reward_model.py` - reward model training
- `selfplay_reward_model.py` - reward model refinement flow
- `phase4_ppo.py` - PPO training
- `phase4_ppo_v2.py` - PPO v2 training
- `diagnostic.py` - diagnostics/debug checks
- `make_zips.py` - packaging helper
- `artifacts/` - evaluation outputs and comparison tables
- `notebooks/` - exported Kaggle notebooks used in training/evaluation

## Evaluation Artifacts

- `artifacts/eval_100_prompts.jsonl` - fixed benchmark prompt set
- `artifacts/outputs_7b_dpo.jsonl` - Qwen2.5-7B DPO outputs
- `artifacts/outputs_gpt2_ppo_v2.jsonl` - GPT-2 PPO v2 outputs
- `artifacts/comparison_full_100.csv` - full side-by-side comparison
- `artifacts/comparison_sample_20.csv` - short sample view

## Reproducibility Notes

- Training/evaluation runs were executed on Kaggle notebooks.
- This repo stores code + exported evaluation outputs.
- Large checkpoints and model binaries are excluded by `.gitignore`.
