# `v2_data_regen` — does multi-trigger data lift f1 past 7.5?

**Schedule:** [`configs/schedules/v2_data_regen.yaml`](../../../../configs/schedules/v2_data_regen.yaml)
**Config:** `train_v2_80gb`
**Chassis:** `unsloth/gemma-4-E4B-it` · LoRA r=128 · max_seq=8192 · num_generations=16
**Hardware:** A100 PCIe 80GB
**Iterations:** E15-E16 (2 iters)
**Started:** 2026-04-27 19:54 UTC · **Finished:** 2026-04-27 23:00 UTC (3h 06min)

## Hypothesis

The first 14 experiments capped f1 at ~7.45 across every reward variant tried.
The dataset distribution analysis showed 86.7% of training rows were 1-trigger
— for which F1 is binary (1.0 if correct trigger picked, 0.0 otherwise) — so
the model had no continuous F1 gradient to learn against. Regenerate with the
realistic 1-3 trigger skew (45/30/20/4/1) and see whether f1 finally moves.

**Falsifiable:** if E15/E16 stay at f1≈7.45 on the new data, the ceiling is not
data-side and the hypothesis is wrong.

## Schedule

```yaml
common_overrides:
  - --eval-batch-size
  - 32
  - --eval-regression-n
  - 100
  - --patience
  - 8
  - --plateau-window
  - 10
  - --plateau-delta
  - 0.05

iters:
  # E15: anchor — single short run on the new dataset, compare to E1's 9.611.
  - config: train_v2_80gb
    overrides: ["--learning-rate", "5.0e-6"]
    description: "lr=5e-6 + new dataset (v2.0.0 distribution) — anchor vs E1's 9.611"

  # E16: long-run — lets a slower-learning multi-trigger curve play out.
  - config: train_v2_80gb
    overrides:
      - --learning-rate
      - 5.0e-6
      - --max-steps
      - 160
      - --patience
      - 15
      - --plateau-window
      - 15
    description: "lr=5e-6 + new dataset + max_steps=160 — long-run on multi-trigger"
```

## Pre-launch comparisons

| anchor | mean_total | f1 | no_halluc | data |
|---|---|---|---|---|
| E1 (mean_total champ) | **9.611** | 7.45 | -0.328 | old (87% 1-trigger) |
| E10 (long-run, old data) | 9.41 | 6.92 | (—) | old |

## Results

| iter | exp | steps | runtime | mean_total | f1_triggers | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/2 anchor | E15 | 25 | 71.6m | 8.778 | 7.343 | -0.896 | -0.123 | 7.3% |
| 2/2 long-run | E16 | 35 | 87.2m | 9.082 | **⭐ 7.523** | -0.920 | +0.026 | 13.7% |

Both ran natural-exit on plateau; neither EARLY_KILLed.

## Verdict

**Hypothesis confirmed for f1, broken for mean_total.** E16 at f1=7.523 is the
first config across 16 experiments to pass 7.5, and the gain happens precisely
where predicted (multi-trigger gradient signal). But mean_total stayed at 9.082
— under E1's 9.611 — because the harder dataset triggered a NEW failure mode:
the model hallucinates more on multi-fact contexts (no_halluc -0.33 → -0.92).

So the ceiling has two causes, not one:

- ✓ **Data-side cause confirmed** — f1 was data-bound. The 86.7% 1-trigger
  distribution capped the absolute ceiling.
- ✗ **Reward-side cause uncovered** — with multi-fact data the reward
  landscape has a new equilibrium that costs no_halluc.

`pass_all` collapsed from 24.8% (E1) to 13.7% (E16) because the threshold for
no_halluc=`+1` is now rarely met.

## Next move

Push reward-side. The follow-up sweep ([`v2_no_halluc_weighted.md`](v2_no_halluc_weighted.md))
weights `reward_no_hallucinated_facts` ×2 and ×3 to bracket the f1 ↔ no_halluc
trade ridge: does stronger penalty gradient pull no_halluc up without dropping
f1 below 7.5?
