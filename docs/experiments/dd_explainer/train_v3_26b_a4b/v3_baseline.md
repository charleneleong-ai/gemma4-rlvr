# `v3_baseline` — does 26B-A4B (MoE) clear the v2 ceiling at the same decode cost?

**Schedule:** [`configs/schedules/v3_baseline.yaml`](../../../../configs/schedules/v3_baseline.yaml)
**Config:** `train_v3_26b_a4b`
**Chassis:** `unsloth/gemma-4-26B-A4B-it` · LoRA r=256 · max_seq=8192 · num_generations=16
**Hardware:** A100 PCIe 80GB
**Iterations:** 2 iters (anchor + long-run)
**Migration plan:** [`docs/v3-26b-a4b-migration.md`](../../../v3-26b-a4b-migration.md)
**Started:** <UTC timestamp> · **Finished:** <UTC timestamp> (<duration>)

## Hypothesis

22 v2 experiments mapped a multi-knob ceiling at `mean_total ≈ 9.32` (E18 champion). The encoder-outlier gate ([`v0_gate.md`](../encoder_outlier/v0_gate.md)) falsified the OOD-routing approach: hallucination is *uniform* across in-distribution rows, not concentrated on weird tail. That points the lever at base-model capacity — a 4B dense model can't reliably ground 5-10 facts of nested JSON simultaneously.

**Hypothesis:** swapping to `gemma-4-26B-A4B-it` (Mixture-of-Experts: 26B total, 4B active per token) gives the experts enough parameters to (a) do the JSON-traversal arithmetic that 4B never could (`prev_amount_correct=0` across 22 v2 experiments), and (b) hold more facts in working memory simultaneously, breaking the uniform `no_halluc` plateau at -0.88..-0.92.

**Falsifiable:** if `v3_baseline` long-run lands at `mean_total < 9.5`, the 6.5× parameter bump is not enough on this task — pivot to non-Gemma family (Qwen 3.5-9B, Llama 3.1-8B) or accept that the rubric weighting itself (f1 max=10 vs no_halluc max=1) is the structural ceiling.

## Schedule

```yaml
common_overrides:
- --eval-batch-size
- '16'
- --eval-regression-n
- '100'
- --patience
- '8'
- --plateau-window
- '10'
- --plateau-delta
- '0.05'
iters:
- config: train_v3_26b_a4b
  overrides:
  - --learning-rate
  - '1.0e-5'
  description: "26B-A4B v3 baseline anchor (vs v2 E15 8.78/7.34/-0.90 \u2014 same\
    \ data)"
- config: train_v3_26b_a4b
  overrides:
  - --learning-rate
  - '1.0e-5'
  - --max-steps
  - '160'
  - --patience
  - '15'
  - --plateau-window
  - '15'
  description: 26B-A4B v3 baseline long-run (vs v2 E18 champion 9.324/7.745/-0.888)
```

`common_overrides`: `--eval-batch-size 16 --eval-regression-n 100 --patience 8 --plateau-window 10 --plateau-delta 0.05`

## Pre-launch comparisons

Direct vs the v2 champion E18 (binary×2 long-run, multi-trigger data, same rubric):

| anchor | model | mean_total | f1 | no_halluc | pass_all | prev_amount |
|---|---|---|---|---|---|---|
| **E1 (v2 champ on old data)** | E4B-it | **9.611** | 7.45 | -0.328 | 24.8% | 0 |
| **E18 (v2 champ on new data)** | E4B-it | **9.324** | **7.745** | -0.888 | 13.9% | 0 |
| E15 (v2 anchor, new data) | E4B-it | 8.778 | 7.343 | -0.896 | 7.3% | 0 |
| **E6 (v3 anchor, lr=2e-6)** | 26B-A4B | — | — | — | — | — (EARLY_KILL @ step 8, KL=9.96) |
| **E7 (v3 long, lr=2e-6)** | 26B-A4B | **7.625** | 6.250 | -1.000 | **0.0%** | 0 |
| E8 (v3 follow-up, lr=5e-6) | 26B-A4B | — | — | — | — | — (EARLY_KILL @ step 2, KL=8207) |

Target: long-run `mean_total ≥ 10.0`. Threshold: `< 9.5` → falsified.

## Results

Three iterations across three learning rates. The full LR landscape:

| iter | exp | LR | warmup | grad_clip | beta | result | notes |
|---|---|---|---|---|---|---|---|
| v3_baseline 1/2 (anchor) | E6 | 2e-6 | 30 | 0.5 | 0.1 | EARLY_KILL @ step 8 | KL=9.96, |grad|=53→1028 spike |
| v3_baseline 2/2 (long) | E7 | 2e-6 | 30 | 0.5 | 0.1 | KEEP, 30 steps, plateau-stop | mean_total=7.625, **1.7 below E18** |
| v3_followup 1/1 (anchor) | E8 | **5e-6** | 15 | 0.5 | 0.1 | EARLY_KILL @ step 2 | **KL=8,207**, grad_norm=1.3M |

A pre-sweep attempt at the original lr=1e-5 (no grad-clip, beta=0.04) also diverged at step 2 (KL=13.07). The full landscape is decisive — see Verdict.

## Verdict

**Falsified.** The 26B-A4B MoE base does not break the v2 ceiling on this task. Three sub-claims:

- ✗ **Long-run `mean_total ≥ 9.5`** — E7 landed at 7.625, **1.7 below E18's 9.324**. The bump in capacity didn't translate to better task performance under available training budget.
- ✗ **`prev_amount_correct > 0`** — still 0 in E7. The 26B parameter pool didn't unlock JSON-traversal arithmetic in 30 steps of fine-tuning.
- ✗ **`no_halluc ≥ -0.5`** — E7 hit -1.0, *worse* than E18's -0.888. Bigger model wasn't a uniform-hallucination fix.
- ✓ **Step time ≤ 200s sustained** — actually achieved (95-130s/step typical), but irrelevant given the policy never trained meaningfully.

### Why — a structural gradient pathology

The decisive evidence is the LR landscape: there is no working LR for this base + GRPO + LoRA combination.

- **lr ≥ 5e-6**: the experts' LoRA path explodes within 2 steps. KL goes from 0.03 → 8,207 between steps 1 and 2; grad_norm hits 1.3M. No stability fix (lower beta, tighter grad-clip) helped at this LR.
- **lr = 2e-6**: stable enough to *not* diverge in iter 2 (30 steps clean), but trains so slowly that the model never moves materially from the base policy. Plateau hit at f1=6.25, below the E18 baseline that v2's lr=5e-6 reached in v2-recipe time.

This is a **structural pathology of MoE LoRA on this task**, not a tuning problem:
- Each token in a rollout activates a subset of experts (top-k routing). The LoRA delta on each expert sees only the tokens routed to it.
- During GRPO advantage updates, the rare experts get a high-magnitude noisy gradient (small denominator, big numerator from advantage variance).
- 4-bit quantization of the base + bf16 LoRA means each expert's update is also numerically noisier than dense LoRA.
- Net: experts get whip-saw pushes. Either lr is low enough to dampen them (and the model doesn't learn) or high enough to overshoot every step (and the policy diverges).

The dense 4B (E4B-it) doesn't have this pathology because the base path is dense — every parameter sees every token. v2's lr=5e-6 worked for 22 experiments because dense LoRA averages gradients across all tokens.

## Next move

Pivot. The 26B-A4B variant is not the right base for GRPO on this task at this hardware budget. Three doors:

1. **Accept E18 as v2 champion, ship that for production** (recommended). 22 experiments + the encoder-gate work + the v3 falsification all point at E18 being the practical ceiling for this task on the Gemma 4 family at L4-fittable sizes.
2. **Try Gemma 4 31B dense at INT4 inference** — 31B BF16 won't fit L4 even at INT4 (~16 GB plus KV) but might be viable at A100 inference if Cloud Run constraint relaxes.
3. **Try non-Gemma family** — Qwen 3.5-9B or Llama 3.1-8B. Both fit L4 INT8. Project constraint said Gemma-only — needs a re-decision.

What WON'T fix this:
- More LoRA capacity (r=256, r=512). The pathology is per-expert noise, not capacity.
- Different reward weighting. The model can't even learn the unweighted rubric stably.
- Longer training. lr=2e-6 for 160 steps would gain maybe 0.5 mean_total — not enough to reach 9.5.

Cross-sweep synthesis lands in [`docs/v3-26b-a4b-migration.md`](../../../v3-26b-a4b-migration.md) (the migration plan now reads as a postmortem) and the project-wide [`docs/ceiling-diagnosis-2026-04-27.md`](../../../ceiling-diagnosis-2026-04-27.md).
