# Encoder-outlier gate — plan & scope

**Status:** scoping (no code yet)
**Branch:** `feat/encoder-outlier-gate`
**Driving doc:** [`docs/ceiling-diagnosis-2026-04-27.md`](ceiling-diagnosis-2026-04-27.md#next-move--encoder-outlier-suppression)

## Why this branch exists

22 experiments and three sweeps confirmed reward-side levers are exhausted: `no_halluc` plateau'd at -0.88..-0.92 across binary×{1,2,3} and granular×1. The branch champion is **E18** at mean_total=9.324 / f1=7.745 / no_halluc=-0.888.

Pushing further on reward shaping doesn't move no_halluc — the equilibrium is structural. **The fix is to stop trying to teach the LLM not to hallucinate and add a pre-flight gate** that decides whether the model should answer at all. OOD inputs route to a structured "insufficient context" response without running Gemma.

## Hypothesis

If a small frozen encoder can separate in-distribution account contexts from obvious OOD ones at AUROC ≥ 0.95, then a meaningful fraction of E18's `no_halluc` penalty comes from inputs the LLM genuinely can't ground — and gating those inputs out at eval time will move mean_total past 9.5 *without re-training Gemma*.

**Falsifiable:** if AUROC < 0.85 on a basic mutation set (the model can't even tell an empty `contract_history` apart from a real one), the failure is the LLM's grounding capability, not data drift — pivot to base-model swap or much larger LoRA rank.

## Build plan

### 1. Outlier dataset (`scripts/build_outlier_set.py`)

Mutate ~100 real account contexts in obvious ways. These are positive OOD examples:

| Mutation | Failure it triggers |
|---|---|
| Drop `contract_history` entirely | LLM has no tariff names to cite — fabricates one |
| Replace tariff names with gibberish (`"zXq42_payg"`) | LLM ignores the gibberish, invents a real name |
| Set numeric `dd_amount` / rate fields to NaN / negative | LLM hallucinates plausible-looking numbers |
| Empty `dd_change_history` on a context that needs prior amount | LLM invents a previous amount |
| Wildly inconsistent dates (end < start) | Reasoning failure, often invents a remediation |
| Truncate `payment_history` to 1 row | LLM extrapolates a pattern from one data point |

For negatives: 100 in-distribution rows sampled from `data/dd_dataset_20260427T201521Z_5500rows.jsonl` with a fixed seed. 50/50 split for train, plus a held-out 50/50 split for AUROC.

### 2. Encoder candidates (`scripts/train_outlier_encoder.py`)

Test in parallel:

- **`bge-small-en-v1.5`** (33M params, 384-d) — frozen, fit a logistic regression on the embeddings.
- **`gemma-embed-300m`** — frozen, fit a logistic regression.
- **`bge-small-en-v1.5` + LoRA** — fine-tune at r=8 on the 200-row binary task.

Each gets a single AUROC number on the heldout split. Cheapest first: frozen-bge before any LoRA.

Input to the encoder is the JSON-stringified `account_context` (truncated to ~2k tokens — encoders are 512-token-bound, so we pre-summarise critical fields rather than feed raw JSON).

### 3. Integration (`dd_explainer_eval.py` — new file)

Pre-flight check in the eval harness:
1. For each heldout row, run the encoder.
2. If OOD score > threshold (calibrated to 95th percentile on the train set), short-circuit to a structured "insufficient context" response.
3. Otherwise, run E18's adapter as today.
4. Score with the existing rubric. The gated rows count as "no hallucination" (model didn't generate anything to hallucinate).

This gives a clean A/B against ungated E18:
- **Ungated (control):** E18's published 9.324 / 7.745 / -0.888.
- **Gated (treatment):** same adapter, same rubric, same heldout — only the gate flips OOD rows out of LLM scope.

If gated mean_total > 9.5, we have the lift.

## Success criteria

| Metric | Threshold | What it tells us |
|---|---|---|
| Encoder AUROC on heldout | ≥ 0.95 | Gate is reliable enough to wire into eval |
| Gated mean_total | ≥ 9.5 | Encoder gate breaks the v2 ceiling |
| Gated f1_triggers | ≥ 7.6 | Gate doesn't filter so aggressively that f1 drops |
| Gated no_halluc | ≥ -0.5 | Gate is doing the work it was supposed to do |

If AUROC ≥ 0.95 but mean_total only nudges to 9.35-9.4, the failure is not data drift — most of the no_halluc cost comes from in-distribution inputs the LLM hallucinates against. Pivot to base-model swap.

## Out of scope for this branch

- Production routing — this is just train + eval gating to validate the hypothesis.
- Re-training Gemma on a gated set — do that *after* the gate works.
- Real-traffic OOD examples — synthetic mutations only for v0; production OOD comes later if AUROC clears.

## Build sequence

1. ~~Branch off main~~ ✓ done
2. **Outlier mutator** (`scripts/build_outlier_set.py`) — emit `data/outlier_set_v0.jsonl` with 200 rows (100 OOD + 100 in-dist), labelled, with the mutation type recorded for diagnosis.
3. **Encoder training** (`scripts/train_outlier_encoder.py`) — three candidates × AUROC report.
4. **Eval integration** (`dd_explainer_eval.py`) — gated A/B against E18.
5. **Writeup** (`docs/experiments/encoder_outlier/v0_gate.md` — note the new top-level `encoder_outlier` since this isn't a `train_v2_80gb` sweep) once results land.

The first three steps should land as separate small PRs against this branch so each can be reviewed independently.
