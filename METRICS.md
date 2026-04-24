# Metrics to watch

When `wandb.enabled` (the `train` and `train_long` configs), `train.py`
installs `WandbMetricDefsCallback` which sets panel groupings and summary
aggregations automatically. This document is the reference for which
metrics matter and why.

## Priority order

### Top-level run-health signals

| Metric | Watch for | Why it matters |
|---|---|---|
| `train/reward` | trending **up** | Primary GRPO scalar — weighted sum of the 7 rewards. Should climb steadily after warmup. |
| `train/rewards/reward_triggers_match_ground_truth/mean` | rising to **+6…+10** | Main correctness signal — F1 vs oracle, scaled to [-2, +10]. This is what proves GRPO is learning the domain rules. |
| `train/kl` | **< 0.5**, stable | KL-divergence from reference policy. Any upward creep past 1 is a red flag — the run we aborted earlier hit 406 at step 12 (policy collapse). |

### Production failure-reward targets (the RLVR "did it help?" numbers)

The three rewards lifted directly from the LangSmith `direct_debit_faithfulness`
error-analysis reports. These are the metrics the demo is optimising.

| Metric | Baseline | Target |
|---|---|---|
| `train/rewards/reward_previous_dd_amount_correct/mean` | 0 (model not citing) | **+2** (cited and correct) |
| `train/rewards/reward_no_hallucinated_facts/mean` | -3 (hallucinating) | **+1** (grounded in input) |
| `train/rewards/reward_underpayment_language_constrained/mean` | +0.5 | stays **+0.5** |

### Shape convergence — should settle fast (within ~50-100 steps)

| Metric | Target |
|---|---|
| `train/rewards/reward_schema_valid/mean` | → **+1** (JSON always parses) |
| `train/rewards/reward_triggers_in_enum/mean` | → **+1** (all triggers are valid enum values) |
| `train/rewards/reward_explanations_well_formed/mean` | → **+0.5** (header ≤10 words, 1-3 sentences) |

### Safety / reward-hacking watchers

| Metric | Alarm threshold | What it means |
|---|---|---|
| `train/completions/min_length` | drops to **1** repeatedly | Model learned to emit EOS immediately to game the shape rewards. |
| `train/completions/clipped_ratio` | **> 0.15** | Generations hitting `max_completion_length` — raise the cap. |
| `train/grad_norm` | **NaN** or **> 10** | LR too high / numerical blow-up. |
| `train/clip_ratio/high_mean` or `low_mean` | **> 0.3** sustained | Policy drifting past trust region — reduce LR or tighten epsilon. |

### Throughput diagnostics (only for tuning)

- `train/step_time` — seconds per optimizer step. E4B baseline ~89 s with xFormers; optimized (bs=12 num_gen=6 max_comp=384) ~76 s. With FA2 expect another ~20 %.
- `profiling/Time taken: transformers.generate` — dominates step time (~86 % in the smoke run). This is where FA2 helps.
- `profiling/Time taken: _calculate_rewards` — sub-millisecond (regex Python).

## At-a-glance workspace layout

`train.py:WandbMetricDefsCallback` step-wise logs running aggregates under
a dedicated **`train/summary/*`** namespace, which appears as its own
collapsible section in the workspace tab between `train/rewards/` and
`train/completions/`. Pin that section at the top — it's the one-glance
view of the whole run.

| Panel (under `train/summary/`) | What it shows |
|---|---|
| `best_reward` | Running max of the GRPO scalar — should climb monotonically. |
| `best_reward_triggers_match_ground_truth` | Running max of the F1 vs oracle. The headline correctness number. |
| `best_reward_no_hallucinated_facts` | Running max of the production-failure check #2. |
| `best_reward_previous_dd_amount_correct` | Running max of failure check #1. |
| `best_reward_underpayment_language_constrained` | Running max of failure check #3. |
| `best_reward_schema_valid` / `best_reward_triggers_in_enum` / `best_reward_explanations_well_formed` | Shape rewards — saturate fast. |
| `current_kl` / `current_loss` / `current_grad_norm` | Live numerical-health signals. |
| `min_completion_length` | Running min — drops to 1 = reward-hacking caught. |

If `best_reward` and `best_reward_triggers_match_ground_truth` trend up
and `current_kl` stays below 0.5, the rest is tuning detail.

## What the callback does

`train.py:WandbMetricDefsCallback.on_train_begin` calls
`wandb.define_metric(...)` to:

- Set `train/global_step` as the x-axis for every `train/*` metric (so
  plots render correctly regardless of the log order transformers sends).
- Surface the run-best value in the summary panel for:
  - `train/reward` — best single-step reward
  - `train/rewards/*/mean` — best per-reward mean
- Surface the run-final value for:
  - `train/kl` — where policy ended up
  - `train/loss`, `train/grad_norm` — numerical health
  - `train/completions/clipped_ratio` — truncation burden
- Surface the run-min for:
  - `train/completions/min_length` — lowest ever, catches reward-hacking

To add more summaries, extend the callback in `train.py`.

## Verified runs

- Smoke (E2B, 2 steps, wandb online): https://wandb.ai/chaleong/gemma4-rlvr/runs/29w0ye7o
- E4B optimisation (5 steps @ bs=12 num_gen=6 max_completion_length=384): https://wandb.ai/chaleong/gemma4-rlvr/runs/vy9hdapi — 2.6× throughput over baseline (bs=8 num_gen=4 max_comp=512).
- Project: https://wandb.ai/chaleong/gemma4-rlvr
