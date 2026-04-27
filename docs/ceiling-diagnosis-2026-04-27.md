# v2 ceiling diagnosis — 2026-04-27

**Status:** complete after 14 experiments across 4 sweeps (`v2_baseline`, `v2_lr_explore`, `v2_step_time_relax`, `v2_cap_neg_tails`). Conclusion: the ~9.6 mean_total / ~7.45 f1_triggers ceiling is **data-side**, not RL-side. Pivoting to data regen.

## What we tried

| sweep | experiments | hypothesis | outcome |
|---|---|---|---|
| `v2_baseline` | E0-E3 (4 iters) | v2 stack delivers; explore lr / beta / lora_rank | E1 `lr=5e-6` champion, mean_total=**9.611**. ceiling first observed. |
| `v2_lr_explore` | E4-E9 (6 iters, 3 useful) | longer training / ng=24 / max_completion=512 / β=0.03 / kitchen-sink can break ceiling | None beat E1. `max_completion=512` *hurt*; `β=0.03` marginal; `ng=24` triage-killed twice (false positive at SLOW_MEAN_S=90). |
| `v2_step_time_relax` | E10-E12 (3 iters) | retry the missed long-run + seed-variance + ng=24 with relaxed triage | Long-run plateaued at 9.41 (lower than E1). seed=1337 dropped to 9.02 (variance ±0.6 across seeds). ng=24 finished cleanly at 8.98. |
| `v2_cap_neg_tails` | E13-E14 (in progress) | rubric was masking learning gains; cap -3 fail penalties to -1 | E13 anchor: mean_total=9.46, f1=7.39, no_halluc PASS% **dropped** 66.8 → 64.4. Softer penalty = weaker gradient = worse, not better. |

## The signal that nailed it

f1_triggers is stuck at 7.39-7.50 across **every** experiment. That's the single observation that ruled out RL-side fixes:

- E0 (anchor)              → f1=6.88
- E1 (champion lr=5e-6)    → f1=7.45
- E10 (long-run)           → f1=6.92
- E12 (ng=24)              → f1=6.88
- E13 (cap-neg-tails)      → f1=7.39
- variance across runs     → ±0.4

When 14 different RL configurations all converge to the same f1 ceiling, the optimisation isn't the problem — the dataset is the problem.

## The data-distribution finding

`data/dd_dataset_20260424T174610Z_5500rows.jsonl`:

```
trigger count distribution (n=5501)
  1 triggers: 4772 (86.7%)
  2 triggers:  728 (13.2%)
  3+ triggers:   0  (0.0%)
```

**86.7% of training rows are 1-trigger.** For 1-trigger examples, F1 is binary: 1.0 if the model picks the correct trigger, 0.0 otherwise. There is no in-between gradient signal.

### Why this caps f1 at exactly ~7.45

If 1-trigger F1 ≈ 0.85 (model has saturated single-trigger accuracy) and 2-trigger F1 ≈ 0.40 (typical for under-trained multi-label):

```
weighted F1 = 0.867 × 0.85 + 0.132 × 0.40 = 0.79
reward      = 0.79 × 12 - 2 = 7.48
```

Matches observed ceiling almost exactly. The model has learned what this dataset can teach.

### Why reward shaping couldn't help (in retrospect)

Capping the -3 hallucination penalty in `v2_cap_neg_tails` softened the gradient on a dimension where the model **was already doing fine** (66.8% pass rate). With less pressure on hallucination, the model spent capacity elsewhere — and hallucination pass rate got worse, not better. Lesson: don't soften rubrics when the underlying ceiling is data composition.

## What unlocking it should look like

Adding 3+ trigger examples does three things at once:

1. **Continuous F1 surface** — a 3-trigger example can score F1 in {0.0, 0.4, 0.67, 0.8, 1.0}. Real partial-credit gradient instead of binary.
2. **Recall transfer** — multi-trigger learning improves 2-trigger F1 too. The 13% slice currently underperforms; it should converge upward.
3. **Richer context for other rubrics** — multi-trigger examples typically have more contract changes / amount data / tariff names. May finally give `prev_amount_correct` (stuck at 0 across all 14 runs) something to attempt.

### Target distribution

| trigger count | current | target |
|---|---|---|
| 1 | 86.7% | 40% (sanity baseline) |
| 2 | 13.2% | 30% |
| 3 | 0% | 20% |
| 4+ | 0% | 10% |

Approx 5500 rows total (preserves dataset size; substitutes hard for easy).

## Decision

Pivot to data regen. Next sweep launches against the new distribution under the original (uncapped) rubric — the rubric was working as intended; only the data was the bottleneck.

If f1_triggers moves past 7.5 on the new distribution, the diagnosis is confirmed. If not, the next levers are LoRA `target_modules` expansion to MLP (capacity for `prev_amount` arithmetic) or base-model swap.

## Reference rows

| experiment | rubric | mean_total | f1_triggers | no_halluc PASS% | well_formed | prev_amount mean |
|---|---|---|---|---|---|---|
| E1 (champion) | 2026-04-26-soften-well-formed | 9.611 | 7.45 | 66.8% | -0.001 | -0.01 |
| E10 (long-run) | 2026-04-26-soften-well-formed | 9.41 | 6.92 | (lift in raw mean) | 0.155 | 0.0 |
| E13 (capped) | 2026-04-27-cap-neg-tails | 9.463 | 7.39 | 64.4% | -0.01 | +0.012 |

W&B project: <https://wandb.ai/chaleong/gemma4-rlvr>
