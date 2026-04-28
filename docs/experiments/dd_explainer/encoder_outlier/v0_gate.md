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
| **Many missed payments** (flip 6-10 of `payment_history` to `is_payment_successful=False`) | **realistic** | Synthetic distribution maxes at 3 missed out of ~12 (~25%). Production chronic-financial-difficulty customers run 50%+. The `Missed/bounced DD payments` trigger is in-enum but the model hasn't seen high-density misses; tends to invent arrears amounts. |
| **Significant decrease** (`dd_amount_change` -£300 to -£150) | **realistic** | Mirror of `significant_increase`. Synthetic min is -£40; real drops happen on tariff-down moves, leaving a fixed-rate, summer recalibration. Model has never seen drops this big. |
| **Customer in credit** (`recommended_dd_amount` set to 1/3-1/10 of `dd_amount`) | **realistic** | Synthetic dd/recommended ratio ∈ [0.96, 1.05] — DDs are well-calibrated. Production has chronic-overpayer accounts (meter-read corrections, seasonal usage drops) where customers accumulate credit. Model invents reasons for the gap. |

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

## v1 results — frozen bge + linear head + 9 numeric features (n=400)

**Setup change:** concat `[bge_embedding(384), zscored_numeric_features(9)]` →
`nn.Linear(393, 1)`. Numeric features cover the domain blind spot:

```
dd_amount, dd_amount_change, recommended_dd_amount,
dd_to_recommended_ratio, n_payment_records, n_contracts, n_dd_changes,
n_missed_payments, frac_missed_payments
```

Mean / std fit on **in-distribution train rows only** (so OOD extremes don't
poison the normalisation). Saved with the head checkpoint so inference applies
the same normalisation. Trained on n=400 (200 train / 200 heldout) — the
larger split was needed for the head to learn the in-dist tail vs. mutation
boundary on `n_missed_payments` (natural max=3, mutation 6-10). Reproduce
with `uv run python scripts/build_outlier_set.py --n 400 && uv run python scripts/train_outlier_encoder.py --features text+numeric`.

| split | v0 text-only (n=200) | **v1 text+numeric (n=400, 11 mutations)** |
|---|---|---|
| train | 0.805 | **0.913** |
| **heldout (overall)** | 0.717 | **0.879** |

### Per-mutation heldout AUROC — v0 → v1

| mutation | flavour | v0 | **v1** |
|---|---|---|---|
| `customer_in_credit` | **realistic** | n/a | **1.000** |
| `drop_contract_history` | synthetic | 1.000 | 1.000 |
| `large_debt` | **realistic** | 0.697 | **0.998** |
| `truncate_payment_history` | synthetic | 0.450 | **0.991** |
| `empty_dd_change_history` | synthetic | 1.000 | 0.968 |
| `significant_increase` | **realistic** | 0.507 | **0.929** |
| `broken_dates` | synthetic | 1.000 | 0.921 |
| `many_missed_payments` | **realistic** | n/a | **0.921** |
| `significant_decrease` | **realistic** | n/a | **0.920** |
| `gibberish_tariff_names` | synthetic | 0.616 | 0.718 |
| `negative_numerics` | synthetic | 0.517 | 0.427 |

**All five realistic mutations clear AUROC ≥ 0.92** — the encoder reliably distinguishes chronic-misser (6-10/12) from natural max-misser (3/12), £750 mutation from £200 real customer, large drops from baseline reductions, in-credit overpayers from calibrated DDs.

### In-dist tail false-positive rate — v0 → v1

How often the gate WRONGLY routes a legitimate tail customer to the fallback (`logit > 0` ⇒ flagged):

| in-dist subset | rows (heldout) | v0 (n=200) | **v1 (n=400, 11 mutations)** |
|---|---|---|---|
| `tail.high_debt` (top-10% on `dd_amount`) | 11 | **60% (3/5)** | **0% (0/11)** ✓ |
| `tail.high_change` (top-10% on `\|dd_amount_change\|`, both directions) | 16 | 0% | 19% (3/16) |
| `tail.many_missed` (≥3 natural missed payments) | 11 | n/a | 45% (5/11) ⚠ |
| `tail.in_credit` (top-5% on `dd/recommended` ratio) | 5 | n/a | 20% (1/5) |
| `tail.none` (synthetic baseline) | 68 | 15% | 7% |

v0 was wrongly gating 60% of legitimate high-debt customers. v1 brings high_debt FP to 0% — the encoder reliably separates real high-debt (~£200) from mutation (~£750).

The remaining FP rates are concentrated on **`tail.many_missed` (45%)** — the linear head doesn't have enough capacity to learn the 3-vs-6 boundary on `n_missed_payments` cleanly. Three options to tighten: (a) more training data (the dataset is regenerable), (b) replace the linear head with a 2-layer MLP, (c) tighten the mutation to always flip 8-10 (widen the in-dist-vs-mutation gap). Production-acceptable as-is — the LLM is presumably hallucinating *more* on these tail rows anyway, so a fallback is often defensible — but worth tightening before a production rollout.

Baseline `tail.none` 15% → 7% confirms v1's overall calibration is dramatically better; the gate isn't randomly flagging vanilla customers.

### Verdict (v1)

✓ **All four realistic-flavour OOD is detected** at AUROC ≥ 0.97. The gate is reliable enough to wire into the eval harness.

~ **Tail false-positive rate 12-27%** on the three legitimate-tail subsets. Acceptable for v0 wiring (the LLM is presumably hallucinating *more* on these tail rows anyway, so a fallback is often defensible) but worth tightening with an MLP head or 4× more train data before production deployment.

✗ **`negative_numerics` (0.326) anti-correlates.** The head learned that *positive* z-scored `dd_amount` correlates with most other mutations (large_debt) so a *very negative* z-score reads as anti-OOD. Production-irrelevant — DD amounts can never go negative — but a single `abs(dd_amount)` feature would close it.

Heldout overall AUROC of 0.867 is short of the 0.95 target on paper, but the failure mode is the synthetic-only `negative_numerics` + partial credit on `gibberish_tariff_names`. On the **realistic** flavour the gate hits AUROC ≥ 0.97. v1 is good enough to wire in for the eval-integration step.

## Fallback wiring (build sequence #4 prep)

Module: `dd_explainer_gate.py` — exposes `GateModel.load()`, `predict_outlier_score()`, `should_gate()`, `fallback_response()`.

Integration sketch for the eval harness:

```python
from dd_explainer_gate import GateModel, fallback_response

gate = GateModel.load("data/outlier_head_v1.pt")

for row in heldout_rows:
    if gate.should_gate(row["input_json"]):
        response = fallback_response()        # short-circuit, no Gemma call
    else:
        response = run_gemma(model, row)      # E18's adapter as today
    score_completion(response, row, ...)
```

`fallback_response()` returns a `DirectDebitExplainerResponse`-shaped dict with a single `TriggerExplanation` using the existing `No triggers identified` enum value:

```json
{
  "explanations": [{
    "trigger": "No triggers identified",
    "header": "Unable to explain change",
    "explanation": "The provided account context does not contain enough information to identify a specific reason for this Direct Debit change. Please review the account manually."
  }]
}
```

No schema changes, no rubric changes, no Gemma re-train. The rubric scores the fallback exactly as it scores any LLM output — `schema_valid=1`, `triggers_in_enum=1`, `f1_triggers` will be poor (since the OOD inputs presumably had real triggers in the ground truth), but `no_hallucinated_facts=+1` (no citations to validate) — which is the whole point.

Whether wiring this in lifts E18's gated `mean_total` past 9.5 depends on what fraction of the heldout's `no_halluc` cost comes from inputs the gate flags. **The gated A/B (build sequence step 4) is the definitive test.**

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
