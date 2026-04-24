# gemma4-rlvr — direct_debit_explainer RLVR mock

A mock RLVR / GRPO training loop that fine-tunes Gemma 4 against the
same I/O contract as the production `direct_debit_explainer` API
(gemini-2.5-pro today).

- Input: `DDExplainerPromptInput` (customer `account_context` + `latest_dd_change`)
- Output: `DirectDebitExplainerResponse` — `List[TriggerExplanation]`, each
  picking one of 7 closed-enum `Trigger` values.
- Reward signal: 7 verifiable checks covering schema validity, trigger F1
  vs a scenario-first oracle, and three production failure categories
  identified from the LangSmith `direct_debit_faithfulness` evaluator
  reports (incorrect previous-DD amount, hallucinated tariffs/rates,
  misused "underpayment" language).

## Layout

```
dd_explainer_data_generator.py     # synthetic data (schemas, generator, oracle)
dd_explainer_rewards.py            # 7 reward functions + REWARD_FUNCS bundle
train.py                           # typer CLI: train / infer / regress
config/                            # Pydantic settings (WandbSettings, TrainSettings)
configs/                           # hydra YAMLs: train / smoke / train_long
.env.example                       # template for WANDB_API_KEY / HF_TOKEN
data/                              # generated JSONL datasets (gitignored except previews)
.error_analysis_cache/             # scrubbed LangSmith trace snapshots
Gemma4_A100_billing_example.ipynb  # notebook walkthrough (same content as the CLI)
METRICS.md                         # W&B dashboard metrics to watch
```

## Quickstart

```bash
# 1. Set up secrets
cp .env.example .env
# Fill in WANDB_API_KEY and (optionally) HF_TOKEN.

# 2. Generate the synthetic training dataset (one-off, ~20s for 5500 rows)
python dd_explainer_data_generator.py -n 5500

# 3. Train
python train.py train                                                    # defaults: configs/train.yaml
python train.py train -c smoke --model-name unsloth/gemma-4-E2B-it       # quick sanity
python train.py train -c train_long -d "first real 600-step run"         # with notes

# 4. Inference / regression
python train.py infer   --lora-path gemma_4_lora --target-trigger "Change in usage"
python train.py regress --lora-path gemma_4_lora --n-rows 20
```

## Config override precedence

CLI flag > `configs/<config-name>.yaml` > `${oc.env:VAR, default}` >
Pydantic defaults (`config/base.py`). Environment variables (`.env` or
shell-exported) are picked up by the YAML at compose time via Hydra's
`oc.env` resolver.

`.env` always trumps a committed YAML for secrets (API keys), while YAML
pins `wandb.mode` per-experiment so smoke runs cannot accidentally log.

## W&B experiment tracking

`train.py` auto-registers panel groupings and summary aggregations via a
`WandbMetricDefsCallback` when wandb is enabled. See
[METRICS.md](METRICS.md) for the priority list of metrics to monitor and
the at-a-glance dashboard recipe.

Project: https://wandb.ai/chaleong/gemma4-rlvr
