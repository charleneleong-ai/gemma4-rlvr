# `v2_granular_no_halluc` — does per-fact granular reward lift no_halluc?

**Schedule:** [`configs/schedules/v2_granular_no_halluc.yaml`](../../../configs/schedules/v2_granular_no_halluc.yaml)
**Config:** `train_v2_80gb` (Gemma 4 4B, A100 PCIe 80GB)
**Iterations:** E21-E22 (2 iters)
**Started:** 2026-04-28 07:55 UTC · **Finished:** 2026-04-28 10:20 UTC (2h 25min)

## Hypothesis

E17-E20 confirmed that *weighting* the binary `reward_no_hallucinated_facts`
does not lift no_halluc. The plateau looks consistent with two mechanisms
inside GRPO:

1. **Binary reward has no slope** between K-of-N partial wins — "3 of 4
   hallucinated" and "0 of 4 valid" both score -3.
2. **GRPO group-normalised advantage has no within-group spread** when all 16
   generations hallucinate identically — gradient direction is undefined.

Replace the +1/-3 binary with `-1 + 2·(n_valid / n_total)` (granular per-fact
partial credit, range [-1, +1]). Smooth slope between the extremes; same +1
ceiling for "no citations" so the no-cite escape hatch isn't worse than today.

**Falsifiable:** if no_halluc stays in the -0.88..-0.92 band even with the
granular slope, the failure is *not* the reward shape — it is something
upstream (data, encoder grounding, base-model capability).

## Schedule

```yaml
common_overrides: [--eval-batch-size, 32, --eval-regression-n, 100,
                   --patience, 8, --plateau-window, 10, --plateau-delta, 0.05]

iters:
  - {config: train_v2_80gb,
     overrides: [--learning-rate, 5.0e-6, --no-halluc-mode, granular],
     description: "granular×1 (anchor — vs E15's 8.78/7.34/-0.90)"}
  - {config: train_v2_80gb,
     overrides: [--learning-rate, 5.0e-6, --no-halluc-mode, granular,
                 --max-steps, 160, --patience, 15, --plateau-window, 15],
     description: "granular×1 + max_steps=160 (vs E18's 9.32/7.745/-0.89)"}
```

CLI plumbing: `--no-halluc-mode {binary, granular}` selects the base reward
function; `--no-halluc-weight` (kept from the previous sweep) wraps either
variant. See
[`dd_explainer_rewards.reward_no_hallucinated_facts_granular`](../../../dd_explainer_rewards.py).

## Pre-launch comparisons

| anchor | mean_total | f1 | no_halluc | reward |
|---|---|---|---|---|
| E15 (×1 binary, anchor) | 8.78 | 7.34 | -0.90 | binary |
| E16 (×1 binary, long) | 9.08 | 7.52 | -0.92 | binary |
| E18 (×2 binary, long — branch champ) | 9.32 | 7.745 | -0.888 | binary ×2 |

## Results

| iter | exp | mode | steps | mean_total | f1 | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|---|
| 1/2 anchor | E21 | granular×1 | 19 | 8.605 | 7.121 | -0.884 | -0.098 | 6.8% |
| 2/2 long | E22 | granular×1 | 31 | 8.933 | 7.459 | -0.896 | -0.083 | 10.7% |

Both natural-exit on plateau.

## Verdict

**Hypothesis falsified.** Granular reward stayed in the same -0.88..-0.92 band
as both binary×1 and binary×2 — shape gave no measurable lift.

The training-time reward log gives the mechanism: granular reward mean=1.0
with std=0 across the early steps. The model **learned to dodge the penalty
entirely by citing nothing**. With `n_total=0` the granular formula returns
+1 (full credit, no hallucination opportunity), so the model collapses to a
"don't cite anything" attractor — same failure mode binary had, expressed
differently.

f1 also dropped slightly (7.459 vs E18's 7.745), confirming that the granular
slope cost recall without gaining no_halluc.

## Next move

**Reward-side levers are exhausted.** Three sweeps (E15-E22) on the new data
have now mapped the landscape: weighting (×1, ×2, ×3) and shape (binary,
granular) all converge to the same equilibrium.

The pivot is to **stop trying to teach the LLM not to hallucinate** and add a
gate in front of it that decides whether it should answer at all. Plan in
[`docs/ceiling-diagnosis-2026-04-27.md`](../../ceiling-diagnosis-2026-04-27.md#next-move--encoder-outlier-suppression).

E18 (binary×2, 160 steps) remains the branch champion at 9.324 / 7.745 / -0.888.
