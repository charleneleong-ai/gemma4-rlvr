# `v2_no_halluc_weighted` — does ×2/×3 reward weighting lift no_halluc?

**Schedule:** [`configs/schedules/v2_no_halluc_weighted.yaml`](../../../../configs/schedules/v2_no_halluc_weighted.yaml)
**Config:** `train_v2_80gb`
**Chassis:** `unsloth/gemma-4-E4B-it` · LoRA r=128 · max_seq=8192 · num_generations=16
**Hardware:** A100 PCIe 80GB
**Iterations:** E17-E20 (4 iters)
**Started:** 2026-04-27 23:50 UTC · **Finished:** 2026-04-28 05:20 UTC (5h 30min)

## Hypothesis

E16 broke the f1 ceiling but no_halluc collapsed to -0.92 on the multi-trigger
data. The simplest reward-side fix is to put more gradient pressure on the
hallucination dimension. Weight `reward_no_hallucinated_facts` ×2 (and ×3 to
bracket) and see whether the model retreats from f1 to gain no_halluc — or
whether stronger penalty gradient lifts no_halluc *without* costing f1.

**Falsifiable:** if no_halluc stays in the -0.88..-0.92 band across both
weights, weighting is not the right lever.

## Schedule

Brackets the trade ridge at 4 points: `{weight=2, weight=3} × {anchor 80 steps,
long-run 160 steps}`.

```yaml
common_overrides: [--eval-batch-size, 32, --eval-regression-n, 100,
                   --patience, 8, --plateau-window, 10, --plateau-delta, 0.05]

iters:
  - {config: train_v2_80gb,
     overrides: [--learning-rate, 5.0e-6, --no-halluc-weight, 2.0],
     description: "no_halluc×2 (anchor — vs E15's 8.78/7.34/-0.90)"}
  - {config: train_v2_80gb,
     overrides: [--learning-rate, 5.0e-6, --no-halluc-weight, 2.0,
                 --max-steps, 160, --patience, 15, --plateau-window, 15],
     description: "no_halluc×2 + max_steps=160 (vs E16's 9.08/7.52/-0.92)"}
  - {config: train_v2_80gb,
     overrides: [--learning-rate, 5.0e-6, --no-halluc-weight, 3.0],
     description: "no_halluc×3 (anchor — bracket vs ×2)"}
  - {config: train_v2_80gb,
     overrides: [--learning-rate, 5.0e-6, --no-halluc-weight, 3.0,
                 --max-steps, 160, --patience, 15, --plateau-window, 15],
     description: "no_halluc×3 + max_steps=160 (bracket long-run)"}
```

The CLI plumbing for `--no-halluc-weight` is a thin factory in
[`dd_explainer_rewards.make_weighted_no_halluc`](../../../../dd_explainer_rewards.py)
that multiplies every `reward_no_hallucinated_facts` score by the weight; the
function name is preserved for W&B logging.

## Pre-launch comparisons

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| E15 (×1, anchor) | 8.78 | 7.34 | -0.90 |
| E16 (×1, long-run) | 9.08 | 7.52 | -0.92 |

## Results

| iter | exp | weight | steps | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/4 anchor | E17 | ×2 | 28 | 8.819 | 7.372 | -0.908 | -0.062 | 9.9% |
| **2/4 long** | **E18** | **×2** | **35** | **⭐ 9.324** | **⭐ 7.745** | -0.888 | +0.005 | 13.9% |
| 3/4 anchor | E19 | ×3 | 28 | 8.855 | 7.354 | -0.900 | -0.029 | 9.0% |
| 4/4 long | E20 | ×3 | 36 | 9.333 | 7.676 | -0.884 | -0.102 | 8.0% |

All ran natural-exit on plateau.

## Verdict

**Hypothesis half-confirmed, half-refuted.**

✓ **f1 lifted to a new branch high (7.745 at E18)** — but the gain was driven
by *longer training*, not the weight: ×3 long-run (E20) hit f1=7.676, while ×2
anchor (E17) only hit 7.372. Compare like-for-like with E16 (×1, long): the
×2 weight added ~0.22 f1 over the unweighted long-run. Modest, not zero.

✗ **no_halluc plateau confirmed.** Across all four iters, no_halluc lands in
the -0.884..-0.908 band — indistinguishable from E15/E16's -0.90/-0.92. The
×3 weight gave no improvement over ×2. Weighting is not the right lever.

E20 is interesting: `prev_amount_correct=+0.146` (first non-zero across all
20 experiments), suggesting longer training under stronger no_halluc pressure
does eventually start scoring on the previous-amount dimension. Not enough to
move pass_all (8.0%).

## Next move

Try reward *shape*, not weight. The plateau looks consistent with two
mechanisms inside GRPO:

1. The binary +1/-3 reward gives no slope between K-of-N partial wins, so
   "3 of 4 hallucinated" and "0 of 4 valid" both score -3.
2. GRPO's group-normalised advantage has no within-group spread when all 16
   generations hallucinate identically — gradient direction is undefined.

The follow-up sweep ([`v2_granular_no_halluc.md`](v2_granular_no_halluc.md))
replaces the binary reward with `-1 + 2·n_valid/n_total` (per-fact partial
credit, range [-1, +1]) to give a smooth slope and force within-group spread.

E18 stands as the branch champion. Save its adapter for downstream evaluation.
