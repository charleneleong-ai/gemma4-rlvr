# `v0_pipeline` — two-stage classifier + LLM explainer

**Schedule:** ad-hoc — `scripts/two_stage_eval.py` is a one-shot A/B harness, not an autoresearch sweep.
**Config slot:** `dd_explainer/two_stage` (no Gemma fine-tune — this reuses E18's adapter unchanged).
**Chassis:** `BAAI/bge-small-en-v1.5` (Stage 1 frozen encoder, 33M, 384-d) + `unsloth/gemma-4-E4B-it` + E18 LoRA r=128 (Stage 2 explainer)
**Hardware:** A100 PCIe 80GB
**Driving doc:** [`docs/v3-26b-a4b-migration.md`](../../../v3-26b-a4b-migration.md#postmortem-added-2026-04-29-after-falsification) (the v3 falsification that pointed back at the architectural lever).

## Hypothesis

22 v2 experiments + the encoder-outlier-gate falsification (PR #8) + the v3 base-model-swap falsification (PR #10) all converged on the same diagnosis: the dd_explainer task asks one 4B model to do two structurally different jobs at once — pick triggers (discrete, 7-way label) AND explain them (free-form prose). The f1 ↔ no_halluc trade ridge that capped mean_total at 9.6 came from those two jobs fighting inside one reward sum.

**Hypothesis:** splitting the task into two stages eliminates the trade ridge entirely.

- **Stage 1**: a small frozen encoder + linear head predicts the trigger set from raw input JSON. f1 by construction.
- **Stage 2**: E18's adapter writes the prose given the predicted triggers as a templated response skeleton.

**Falsifiable:** if two-stage `mean_total` doesn't clear 9.6 (the v2 ceiling), or if Stage 1's classifier rubric reward stays below E18's f1=7.745, the architectural decoupling didn't help.

## Build sequence

1. **Stage 1 classifier** (`scripts/train_trigger_classifier.py`) — multi-label classifier mirroring the encoder-outlier gate but with 6-d sigmoid head (`No_triggers_identified` handled as default rule).
2. **Stage 2 wrapper** (`dd_explainer_two_stage.py`) — `TwoStageClassifier.load()` for inference, plus `build_two_stage_prompt()` that injects predicted triggers into the user message as a "RESPONSE TEMPLATE" with pre-filled trigger fields.
3. **A/B harness** (`scripts/two_stage_eval.py`) — generate completions twice on same heldout (vanilla E18 prompt vs Stage-1-injected prompt), score both with the existing rubric.

## Stage 1 classifier — variant exploration

| variant | encoder | input dim | macro_f1 | rubric reward | exact_match | vs E18 (7.745) |
|---|---|---|---|---|---|---|
| v0 | bge-small (33M, 384-d) | 393 (384 + 9 numeric) | 0.768 | 6.730 | 0.266 | -1.02 |
| v1 | bge-base (109M, 768-d) | 777 (768 + 9 numeric) | 0.792 | 7.008 | 0.298 | -0.74 |
| v2 | qwen3-embedding-0.6B (600M, 1024-d) | 1033 (1024 + 9 numeric) | 0.797 | 7.051 | 0.302 | -0.69 |
| **v3** | **bge-small (33M)** | **399 (384 + 9 numeric + 6 discriminator)** | **0.915** | **8.767** | **0.638** | **+1.02** ⭐ |

**Headline finding:** the *cheapest* encoder won, with the *right input features*. The v0/v1/v2 progression showed that bigger encoders give modest lift (+0.32 rubric across 18× capacity scale-up). Inspecting `dd_explainer_data_generator.py` revealed the over-predicting classes (Manual reduction / Exemption Expiry / Change in usage) are defined by isolated booleans the linear head couldn't pull out of any-size JSON embedding. Surfacing them as 6 explicit features (`is_amount_manually_reduced`, `dd_reason_is_customer_request`, `is_exemption`, `exemption_expired`, `abs_electricity_change_percent`, `abs_gas_change_percent`) lifted rubric +1.7 in one shot.

**Per-trigger lift v0 → v3:**

| trigger | v0 F1 | v3 F1 | Δ |
|---|---|---|---|
| First DD review since account start | 1.000 | 1.000 | 0 |
| Missed/bounced DD payments | 0.985 | 0.973 | -0.012 |
| **Change in usage** | 0.541 | **1.000** | **+0.459** |
| **Exemption Expiry** | 0.679 | **0.978** | **+0.299** |
| **Manual reduction** | 0.616 | **0.871** | **+0.255** |
| Change in unit rates | 0.787 | 0.668 | -0.119 (regression — head reweighted away from bge-similarity once booleans dominated; a 2-layer MLP head likely fixes) |

## End-to-end A/B (n=1000, E18 adapter, 4-bit inference)

**Setup:**
- Same heldout split as v2 sweeps (seed=42 → 1000 rows)
- Both passes generate via the same Gemma E18 4-bit adapter, batched at 64 (after batch=128 OOM'd on the longer two-stage prompts)
- Total wall-clock: ~50 min (both phases ran end-to-end)

**Result:**

| metric | vanilla | **two-stage** | Δ |
|---|---|---|---|
| **mean_total** | 8.354 | **10.293** | **+1.939** ⭐ |
| f1_triggers | 6.645 | 8.619 | **+1.974** |
| no_halluc | -0.804 | -0.840 | -0.036 |
| well_formed | +0.032 | +0.054 | +0.022 |
| pass_all | 11.9% | 14.1% | +2.2% |
| prev_amount_correct | 0 | 0 | 0 |

(Note: vanilla at 8.354 vs E18's published 9.324 because both passes use 4-bit inference for memory headroom. The relative delta is honest; full-precision two-stage would land higher.)

**Stage 1 prediction distribution** on the heldout:
- Change in unit rates: 41.4%
- Manual reduction: 32.0%
- Missed/bounced DD payments: 31.0%
- Exemption Expiry: 27.9%
- First DD review since account start: 19.2%
- Multi-trigger rate: 60.6%

## Verdict

✓ **Two-stage breaks the v2 ceiling decisively.** mean_total = 10.293, the **first config across 26 experiments** to clear 10.0. v2 ceiling at ~9.6 (E1's full-precision champion) is now broken.

✓ **f1 lift is real and large** (+1.974). Stage 1's rubric=8.77 translates almost 1:1 to end-to-end f1=8.62 — the templating works as designed.

✗ **The "decoupling lifts no_halluc" claim is weak.** n=20 smoke saw +0.20; n=1000 saw -0.036 (within noise). The trade ridge isn't fully eliminated by giving the LLM pre-chosen triggers — the LLM still hallucinates supporting facts in the explanation prose at roughly the same rate. **Most of the lift is from f1, not the rubric trade-off being resolved.**

✓ **`pass_all` improves +2.2 points** — the rubric's binary thresholds now pass for more rows because f1 reliably hits its target.

## Next move

The architectural decoupling is the right move. Two paths to compound the win:

1. **Productionise this** — the gain is large enough (+1.9 mean_total, ~+20%) that v0_pipeline becomes the deployment target. Wire Stage 1 + Stage 2 into the inference path; ship.
2. **Fresh GRPO on Stage 2 only** — train a *new* LoRA with the f1 reward removed and the no_halluc / well_formed weights bumped. The LLM's job is now narrower (explain pre-given triggers), so GRPO should converge to a no_halluc-optimised policy without the f1 trade. Could push mean_total another +0.5 to +1.0.
3. **2-layer MLP head on Stage 1** — addresses the `Change in unit rates` regression and may push classifier rubric from 8.77 toward 9.5.

(2) is the highest-leverage follow-up. (3) is cheap and worth doing before (2) since it raises the ceiling on what Stage 2 sees as input.

## Out of scope for this PR

- Production deploy (Cloud Run + L4 wiring) — separate PR.
- Stage 2 GRPO retrain (option 2 above) — separate PR; would supersede E18.
- 2-layer MLP head — separate small PR before any further sweeps.
- Threshold-per-class tuning on Stage 1 — calibration set + per-class threshold optimisation.

---

## Update 2026-04-30: option 3 (MLP head) + v4_mlp + E18 = new champion at 10.816

After the v0_pipeline writeup landed, two follow-up branches:

1. **v4 GRPO retrain** with the v3 linear classifier (PR #11 commit `3e68693`) — fresh LoRA from base, GRPO on Stage-1-injected prompts. Hit `mean_total = 10.804` (E25, 30 steps long-run). f1=8.793, pass_all=12.9%. **+0.51 over v0_pipeline, but no_halluc still flat at -0.844.**

2. **Option 3: 2-layer MLP head** on Stage 1 (commit `4876881`) — `Linear(399, 128) → GELU → Dropout → Linear(128, 6)` instead of a single linear layer. **Stage 1 macro_f1 jumped 0.915 → 0.990, rubric reward 8.767 → 9.840, exact_match 0.638 → 0.956.** 5/6 triggers at F1=1.000.

The decisive question: with a near-perfect classifier, do we still need the GRPO retrain? Re-ran the n=1000 A/B with `v4_mlp + E18` (no retrain):

| metric | vanilla E18 | v0_pipeline (v3 linear) | E25 (v4 GRPO retrain w/ v3 linear) | **v4_mlp + E18 (no retrain)** ⭐ |
|---|---|---|---|---|
| mean_total | 8.354 | 10.293 | 10.804 | **10.816** |
| f1_triggers | 6.645 | 8.619 | 8.793 | **9.108** |
| no_halluc | -0.804 | -0.840 | -0.844 | -0.848 |
| well_formed | +0.032 | +0.054 | — | **+0.062** |
| pass_all | 11.9% | 14.1% | 12.9% | **22.4%** |

**v4_mlp + E18 (3 min Stage 1 train, no Stage 2 retrain) ties E25 (3 min + 90 min GRPO) on mean_total** and dominates on f1 (+0.315) and pass_all (+9.5 pts). The cheaper architectural path wins.

### Final verdict

✓ **v4_mlp + E18 is the new champion** at `mean_total = 10.816`. First config to clear `pass_all = 20%`.

✓ **Stage 1 quality is the dominant lever, not Stage 2 retraining.** The MLP-head + 6 discriminator features lifted classifier rubric from 6.73 (v0) to 9.84 (v4) — far exceeding what GRPO retraining achieved on top of a weaker classifier. This shifts the project's optimization budget: future improvements should focus on Stage 1 (better calibration, threshold-per-class, more discriminator features) not on Stage 2 fine-tuning.

✗ **no_halluc plateau confirmed across four architectures** (v2 sweeps with f1 in gradient, v0_pipeline with classifier-templated triggers, E25 with GRPO retrain on Stage-1-injected prompts, v4_mlp with near-perfect Stage 1). The trade-ridge hypothesis was wrong; the plateau is a 4B-base capability bound. Fixing it requires structural change (RAG, larger base) — not reward shaping.
