# Mini-InstructGPT

Mini-InstructGPT is an end-to-end alignment project that starts with a GPT-2 RLHF pipeline and scales to a 7B QLoRA + DPO setup.

## Project Goals

- Build an InstructGPT-style training pipeline from scratch.
- Understand reward modeling and PPO behavior in practice.
- Reduce deflective responses and improve direct-answer quality.

## What Was Built

- GPT-2 pipeline:
- SFT training scripts
- Reward model training
- PPO / PPO v2 training loops
- 7B scale-up:
- QLoRA-style adapter workflow (Qwen2.5-7B-Instruct base)
- DPO training data preparation and training flow
- Evaluation:
- Fixed-prompt comparisons between GPT-2 PPO v2 and 7B DPO

## Key Result Snapshot

- Qwen (7B DPO) deflection rate: `2.0%`
- GPT-2 PPO v2 deflection rate: `88.0%`
- Qwen (7B DPO) direct-answer rate: `97.0%`
- GPT-2 PPO v2 direct-answer rate: `1.0%`

## Repository Layout

- `phase1_setup.py` - initial setup and data prep helpers
- `phase2_sft_v2.py` - supervised fine-tuning stage
- `phase3_reward_model.py` - reward model training
- `phase4_ppo.py` - PPO training stage
- `phase4_ppo_v2.py` - PPO v2 iteration
- `selfplay_reward_model.py` - reward-model refinement script
- `diagnostic.py` - debugging/diagnostics utilities
- `make_zips.py` - artifact packaging helper

## Notes

- Large checkpoints, model weights, and zip artifacts are excluded via `.gitignore`.
- This repository tracks code and workflow, not full model binaries.
