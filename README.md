# Mini-InstructGPT

Mini-InstructGPT is an end-to-end alignment project that started with a GPT-2 RLHF-style pipeline and then scaled to a 7B QLoRA + DPO setup on Kaggle.

The main objective was not only to train a model, but to debug real alignment failure modes (especially reward hacking / deflective behavior) and show measurable improvement after redesigning the pipeline.

## Final Outcome

| Metric | GPT-2 PPO v2 | Qwen2.5-7B DPO |
|---|---:|---:|
| Deflection rate | 88.0% | 2.0% |
| Direct-answer rate | 1.0% | 97.0% |

Benchmark setup:
- Same fixed 100 prompts for both models.
- Outputs saved in `artifacts/` and compared side-by-side.

## Why This Project Matters

- Demonstrates practical RLHF pipeline construction, not only fine-tuning.
- Documents real training pathologies in small-model PPO.
- Shows a justified method pivot (PPO -> DPO) with clear empirical gains.

## Project Timeline

### Phase 1: Setup + Data Prep
- Script: `phase1_setup.py`
- Outcome: base project scaffolding and dataset preprocessing pipeline.

### Phase 2: Supervised Fine-Tuning (SFT)
- Script: `phase2_sft_v2.py`
- Outcome: initial instruction-following model baseline for downstream alignment.

### Phase 3: Reward Modeling
- Scripts: `phase3_reward_model.py`, `selfplay_reward_model.py`
- Outcome: trained reward model and self-play-driven refinement attempts.
- Key observation: improved reward model metrics did not fully prevent exploit patterns.

### Phase 4: PPO / PPO v2
- Scripts: `phase4_ppo.py`, `phase4_ppo_v2.py`
- Outcome: RL policy optimization runs with reward model guidance.
- Key issue observed: policy increasingly produced hedgy/deflective outputs despite higher reward trends.

### Scale-Up Phase: 7B QLoRA + DPO
- Notebooks: `notebooks/migptv2.ipynb`, `notebooks/7b-qlora-dpo-first-success.ipynb`
- Base model: `Qwen2.5-7B-Instruct`
- Outcome: large drop in deflection and strong rise in direct answering on the fixed benchmark.

## Repository Structure

- `phase1_setup.py` - setup and data preparation helpers.
- `phase2_sft_v2.py` - SFT training stage.
- `phase3_reward_model.py` - reward model training.
- `selfplay_reward_model.py` - reward model refinement flow.
- `phase4_ppo.py` - PPO training loop.
- `phase4_ppo_v2.py` - PPO v2 iteration.
- `diagnostic.py` - diagnostics/debug checks.
- `make_zips.py` - packaging helper for model artifacts.
- `notebooks/` - exported Kaggle notebooks used in training/evaluation.
- `artifacts/` - benchmark prompts, model outputs, and comparison CSVs.

## Notebooks Included

- `notebooks/7b-qlora-dpo-first-success.ipynb`
- `notebooks/migptv2.ipynb`
- `notebooks/migptv2-test.ipynb`
- `notebooks/mini-instructgpt-7b-inference.ipynb`
- `notebooks/mini-instructgpt-model-comparison.ipynb`

## Artifacts Included

- `artifacts/eval_100_prompts.jsonl` - fixed prompt set used for evaluation.
- `artifacts/outputs_7b_dpo.jsonl` - Qwen2.5-7B DPO responses.
- `artifacts/outputs_gpt2_ppo_v2.jsonl` - GPT-2 PPO v2 responses.
- `artifacts/comparison_full_100.csv` - full merged comparison.
- `artifacts/comparison_sample_20.csv` - sample subset for quick review.

## How To Reproduce (High-Level)

1. Run setup/data prep.
2. Train SFT baseline.
3. Train reward model.
4. Run PPO / PPO v2 experiments.
5. Scale up with QLoRA + DPO on 7B base model.
6. Evaluate both models on the same 100-prompt set.
7. Compute deflection/direct-answer metrics from exported outputs.

## Limitations

- Large model checkpoints are excluded from git.
- Kaggle runtime constraints affected some experiment design choices.
- The benchmark here focuses on behavioral quality; broader factual benchmarks can be added.

## Reproducibility Notes

- Training/evaluation were executed on Kaggle.
- This repository tracks code + exported evaluation artifacts.
- Generated checkpoints and zipped binaries are excluded by `.gitignore` to keep the repo lightweight.
