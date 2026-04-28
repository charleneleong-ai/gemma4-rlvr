# `v0_gate` — encoder-outlier OOD gate, plan & scope

**Status:** scoping (no code yet)
**Branch:** `feat/encoder-outlier-gate`
**Task:** `dd_explainer` · **Config slot:** `encoder_outlier` (no Gemma training run — this is a frozen-encoder + linear-head pipeline)
**Driving doc:** [`docs/ceiling-diagnosis-2026-04-27.md`](../../../ceiling-diagnosis-2026-04-27.md#next-move--encoder-outlier-suppression)

## Why this branch exists

22 experiments and three sweeps confirmed reward-side levers are exhausted: `no_halluc` plateau'd at -0.88..-0.92 across binary×{1,2,3} and granular×1. The branch champion is **E18** at mean_total=9.324 / f1=7.745 / no_halluc=-0.888.

Pushing further on reward shaping doesn't move no_halluc — the equilibrium is structural. **The fix is to stop trying to teach the LLM not to hallucinate and add a pre-flight gate** that decides whether the model should answer at all. OOD inputs route to a structured "insufficient context" response without running Gemma.

## Hypothesis

If a small frozen encoder can separate in-distribution account contexts from obvious OOD ones at AUROC ≥ 0.95, then a meaningful fraction of E18's `no_halluc` penalty comes from inputs the LLM genuinely can't ground — and gating those inputs out at eval time will move mean_total past 9.5 *without re-training Gemma*.

**Falsifiable:** if AUROC < 0.85 on a basic mutation set (the model can't even tell an empty `contract_history` apart from a real one), the failure is the LLM's grounding capability, not data drift — pivot to base-model swap or much larger LoRA rank.

## Build plan

### 1. Outlier dataset (`scripts/build_outlier_set.py`)

Mutate ~100 real account contexts. The set covers **two flavours of OOD**: synthetic (clearly broken JSON / impossible values) and **domain-realistic** (legitimate but tail-extreme — high-debt customers, big DD jumps that production sees but training doesn't).

| Mutation | Flavour | Failure it triggers |
|---|---|---|
| Drop `contract_history` entirely | synthetic | LLM has no tariff names to cite — fabricates one |
| Replace tariff names with gibberish (`"zXq42_payg"`) | synthetic | LLM ignores the gibberish, invents a real name |
| Set numeric `dd_amount` / rate fields to negative | synthetic | LLM hallucinates plausible-looking numbers |
| Empty `dd_change_history` on a context that needs prior amount | synthetic | LLM invents a previous amount |
| Wildly inconsistent dates (end < start) | synthetic | Reasoning failure, often invents a remediation |
| Truncate `payment_history` to 1 row | synthetic | LLM extrapolates a pattern from one data point |
| **Very large debt** (×4-8 multiplier on `dd_amount` — ~£500-£1700) | **realistic** | Synthetic distribution has p99=£211 / max=£219; production sees high-usage / commercial / arrears customers. LLM fabricates explanations for unfamiliar magnitudes. |
| **Significant increase** (`dd_amount_change` £150-£300) | **realistic** | Synthetic max is ±£40; real price-cap moves and tariff resets can jump 5×. LLM invents a cause that doesn't match context. |

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

## v0 results — frozen bge-small-en-v1.5 + linear head

**Setup:** `BAAI/bge-small-en-v1.5` (33M params, frozen), mean-pooled L2-normalised
sentence embedding, `nn.Linear(384, 1)` head, `BCEWithLogitsLoss`, AdamW @ lr=1e-2,
100 epochs, 100 train / 100 heldout (50/50 stratified by `is_outlier`),
seed=42. ~30s end-to-end on the A100. Reproduce with
`uv run python scripts/train_outlier_encoder.py`.

| split | AUROC |
|---|---|
| train | 0.805 |
| **heldout (overall)** | **0.717** |

The overall metric is dragged down by the harder mutations — see the
per-flavour breakdown below for the load-bearing signal.

### Per-mutation heldout AUROC

| mutation | flavour | AUROC | encoder picks up |
|---|---|---|---|
| `drop_contract_history` | synthetic | **1.000** | empty array → JSON shape change |
| `empty_dd_change_history` | synthetic | **1.000** | empty array → JSON shape change |
| `broken_dates` | synthetic | **1.000** | `1970-01-01` strings stand out vs. real dates |
| `large_debt` | **realistic** | 0.697 | partial — £740 is lexically uncommon vs. £140-ish norm |
| `gibberish_tariff_names` | synthetic | 0.616 | `zxq42_payg` reads unlike English brand names |
| `negative_numerics` | synthetic | 0.517 | ✗ no "DD should be positive" prior |
| `significant_increase` | **realistic** | 0.507 | ✗ £200 jump is lexically normal, structurally fine |
| `truncate_payment_history` | synthetic | 0.450 | ✗ shorter array is structurally legal |

### Verdict (v0)

✓ **Structural OOD is solved.** Empty arrays and broken dates score AUROC = 1.000
without any feature engineering — the pretrained encoder already separates them.

~ **Domain-magnitude OOD is partially solved.** `large_debt` lands at
0.697 — the encoder has some weak signal that £740 is unusual (token
frequency on big numbers in pretraining), but it's not reliable enough
to gate on alone.

✗ **Semantic / domain-numeric OOD is not solved.** `significant_increase`
(0.51) and `truncate_payment_history` (0.45) are at-or-below chance.
A £200 dd_amount_change is well-formed text; a 1-row payment_history is
well-formed JSON. The pretrained encoder has no domain rule for either.

This isn't the falsifying threshold from the original plan — that was
"can the encoder tell an empty `contract_history` from a real one" and
the answer there is yes, perfectly. The v0 result rules in **structural
gating** but rules out the simplest possible **domain-aware gating**
(both synthetic-numeric and realistic-magnitude flavours).

### Next move (v1 candidates)

1. **Pre-summarise the input** — extract critical numerics (`dd_amount`, `n_payments`, `n_contracts`) into a short natural-language sentence the encoder can score. e.g.
   `"DD amount £-141.20, recommended £-145.02, 1 payment record, 1 contract."`
   Negative numbers are unusual *prose* even when they're not unusual JSON.
2. **Add manual numeric features alongside the embedding** — concatenate `[embedding, sign(dd_amount), n_payments, n_contracts]` before the linear head. ~5 extra dims, but encodes the domain rules the encoder lacks.
3. **Try `gemma-embed-300m`** — bigger encoder, more numeric awareness in pretraining. Cheap to swap (one CLI flag).

(1) is the cheapest, most principled fix. (2) is the most reliable — explicit features bypass the encoder's blind spot entirely. Worth running both and comparing.

The encoder gate as currently specified will still wire in — it'll just gate **structural OOD only** for v0. Whether that lifts mean_total past 9.5 depends on what fraction of E18's `no_halluc` failures come from structural vs. semantic OOD. The eval-integration step (build sequence #4) measures that directly.

## Out of scope for this branch

- Production routing — this is just train + eval gating to validate the hypothesis.
- Re-training Gemma on a gated set — do that *after* the gate works.
- Real-traffic OOD examples — synthetic mutations only for v0; production OOD comes later if AUROC clears.

## Build sequence

1. ~~Branch off main~~ ✓ done
2. **Outlier mutator** (`scripts/build_outlier_set.py`) — emit `data/outlier_set_v0.jsonl` with 200 rows (100 OOD + 100 in-dist), labelled, with the mutation type recorded for diagnosis.
3. **Encoder training** (`scripts/train_outlier_encoder.py`) — three candidates × AUROC report.
4. **Eval integration** (`dd_explainer_eval.py`) — gated A/B against E18.
5. **Writeup** — backfill the Results / Verdict / Next move sections of *this* doc once results land.

The first three steps should land as separate small PRs against this branch so each can be reviewed independently. Cross-sweep synthesis (multi-knob equilibria, branch champions, etc.) belongs in [`docs/ceiling-diagnosis-2026-04-27.md`](../../../ceiling-diagnosis-2026-04-27.md), not here.
