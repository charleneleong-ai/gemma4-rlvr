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
| **v3_baseline anchor** (this sweep, iter 1) | 26B-A4B | — | — | — | — | — |
| **v3_baseline long-run** (this sweep, iter 2) | 26B-A4B | — | — | — | — | — |

Target: long-run `mean_total ≥ 10.0`. Threshold: `< 9.5` → falsified.

## Results

_No results.jsonl rows found yet — fill in once the sweep finishes._

## Verdict

_Pending — sweep launches after smoke validation lands._

Sub-claims to evaluate when results land:

- ✓/✗ Long-run `mean_total ≥ 9.5` — base-model swap moved the ceiling at all.
- ✓/✗ Long-run `mean_total ≥ 10.0` — swap was the right move; v3 is the new champion.
- ✓/✗ `prev_amount_correct > 0` — 26B's working memory is large enough for nested-JSON arithmetic.
- ✓/✗ `no_halluc ≥ -0.5` — uniform hallucination genuinely was a capacity problem.
- ✓/✗ Step time ≤ 200s sustained — 2-3× v2 budget holds; sweep wall-clock stays under 12h.

## Next move

_Pending — depends on verdict._

If `mean_total ≥ 10`: ship v3 as the new champion. Skip further sweeps; the base-model swap was the answer. Cross-link to a new `docs/ceiling-diagnosis-26b-a4b.md` rolling up the v2→v3 transition.

If `mean_total ∈ [9.5, 10]`: re-run the v2 reward sweeps (binary×2, granular) on top of v3 — they may hit different optima with bigger working memory. New schedules: `v3_no_halluc_weighted`, `v3_granular_no_halluc`.

If `mean_total < 9.5`: investigate. Either LoRA r=256 is too small (try r=512), 4-bit quantization at training is too lossy for MoE experts (try BF16 base + smaller seq length), or the rubric itself is the bottleneck (rubric reweighting to f1×0.5 + no_halluc×3 — the encoder-gate writeup notes this regime).

Cross-sweep synthesis goes in [`docs/v3-26b-a4b-migration.md`](../../../v3-26b-a4b-migration.md), not this writeup.
