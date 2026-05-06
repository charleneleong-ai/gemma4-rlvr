# Templated Trigger Backfill Retrospective

Inference-time post-processing on the existing `data/eval_e28_v5_templates_n1000.per_row.jsonl`
dump. For every Stage-1-predicted trigger absent from the LLM's output,
append a templated explanation entry rendered from `extract_trigger_grounding` /
`extract_valid_facts`. Gated on Stage-1 v5 sigmoid probability `>= 0.9`.
No regeneration, no retrain. Pure rescore.

## Headline — last reachable lift before retrain

| metric | wf-fix champion (PR #32) | **+ trigger backfill (gated)** | Δ |
|---|---:|---:|---:|
| mean_total | 15.897 | **15.967** | **+0.070** |
| f1_triggers (mean) | 9.897 | **9.967** | **+0.070** |
| no_halluc (row-level) | 1.000 | 1.000 | preserved |
| no_halluc (slot-only) | 1.000 | 1.000 | preserved |
| well_formed (mean) | 0.500 | 0.500 | preserved |
| **pass_all_pct** | 97.7% | **99.1%** | **+1.4 pp** |
| pass_all (count) | 977 / 1000 | **991 / 1000** | +14 |

**991 / 1000 rows now pass all 7 rubric components.**

vs PR-F champion (14.014): **+1.953 mean_total, ~8.5x pass_all (11.7% → 99.1%).**

mean_total is now at **99.8 % of theoretical ceiling (16.0)**.

## Diagnosis — what was failing

PR #32's ceiling analysis showed 23/1000 residual failures, **all on
`f1_triggers` only**. Per-row breakdown:

| failure cause | count |
|---|---:|
| Stage-1 v5 misclassified | 0 / 23 |
| Stage-1 right, **LLM dropped a trigger** | **23 / 23** |
| Parse / other | 0 / 23 |

So Stage-1 v5 was perfect on every failing row — Stage-2 was emitting
N-1 explanations when Stage-1 said there were N triggers. Pattern was
consistent: the LLM tends to keep "salient" triggers (Manual reduction,
Exemption Expiry, Missed/bounced) and drop "metadata" triggers (First
DD review, Change in usage / unit rates) when the two are paired:

| stage-1 combo | rows failing | dropped |
|---|---:|---|
| Change in usage + First DD review | 11 / 23 | First DD review |
| Other (3-4-trigger combos) | 12 / 23 | usually the lower-salience one |

Every failing row scored `f1 = 6.0` (1 of 2 GT triggers emitted) instead of 10.0.

## The fix

`backfill_missing_triggers(parsed, stage1_triggers, stage1_probs, grounding,
valid_facts, confidence_threshold=0.9)` in `dd_explainer_template_renderer.py`:

```python
emitted = {e["trigger"] for e in parsed["explanations"] ...}
for trigger in stage1_triggers:
    if trigger in emitted:
        continue
    if stage1_probs is not None and stage1_probs[trigger] < threshold:
        continue                       # confidence gate
    rendered = render_lonely_explanation(trigger, grounding, valid_facts)
    if rendered is None:
        continue                       # no template → skip
    explanations.append(_build_backfill_entry(trigger, rendered, valid_facts))
```

Reuses the four existing lonely-trigger renderers (`first_dd_review`,
`missed_payments`, `manual_reduction`, `exemption_expiry`). Slot citation
fields (`tariff_cited` / `prev_amount_cited` / `rate_change_pct_cited`)
are populated from `valid_facts` so the backfilled entry is rubric-complete
by construction.

## The confidence gate — production safety

Stage-1 v5's predictions on this eval are 100 % exact-match because its
3 boolean features are deterministic predicates over `input_json`. So on
the eval the gate is not load-bearing (ungated rescore produces identical
results).

But on truly out-of-distribution production data (different bill formats,
edge cases), Stage-1 false-positives would otherwise be **forced** into
the output by the backfill — lowering `no_halluc` and `f1` (extra trigger).
The gate caps that risk.

Empirical calibration on the eval:

| max trigger prob (per-row) | n rows |
|---|---:|
| max ≥ 0.99 | ~970 |
| 0.9 ≤ max < 0.99 | ~30 |
| 0.5 ≤ max < 0.9 | **0** |
| max < 0.5 (no-trigger fallback) | ~3 |

The [0.5, 0.9) band is empty — v5's MLP head is essentially binary, so a
0.9 threshold is a free safety net at no cost on the current distribution.

## Why we stopped at 0.07 mean_total

| residual fail | n |
|---|---:|
| Drops "Change in unit rates" | 6 |
| Drops "Change in usage" | 3 |

All 9 remaining failures drop `Change in usage` or `Change in unit rates`,
which have no renderer in the lonely-trigger set. To close the last 0.03
mean_total we'd need:

1. **Add usage / unit-rate templates** — doable, ~50 lines per renderer
   that cite the correct anchors (rate_change_pct values from valid_facts,
   usage figures from `account_context.usage_history`). Estimated impact:
   ~+0.03 mean_total → ceiling.
2. **Or retrain Stage-2 with a trigger-completeness reward** — addresses
   the root cause (LLM dropping triggers) instead of patching outputs.
   Higher cost, would also make the backfill unnecessary.

Both belong in a follow-up; the cost-benefit at 99.1 % is already lopsided
in favour of stopping.

## Tests

12 new tests in `test_template_renderer.py`:

- No-op when no Stage-1 trigger is missing.
- Appends one entry per missing trigger with all slot fields populated.
- Gate blocks low-confidence predictions; threshold boundary inclusive (>=).
- `stage1_probs=None` bypasses gate (legacy path).
- Skips when no renderer or no grounding available.
- Idempotent (running twice doesn't duplicate entries).
- Handles malformed input (no `explanations` key, non-list, etc.).
- Falls back gracefully when `valid_facts` lacks a tariff name.

All 29 renderer tests pass.

## Cumulative roadmap

| step | mechanism | mean_total | pass_all |
|---|---|---:|---:|
| PR-F champion | LMFE force-populate + rubric reform | 14.014 | 11.7 % |
| E28 (PR-G) | retrain w/ --constrain-facts | 14.396 | 25.8 % |
| + lonely-trigger templates | inference-time | 14.684 | 44.1 % |
| + `_TARIFF_RE` regex fix | rubric reform | 14.765 | 48.3 % |
| + No-triggers fallback template | inference-time | 14.787 | 49.7 % |
| + Stage-1 v5 (3 boolean features) | classifier retrain | 15.634 | 69.1 % |
| + well_formed regex fix | rubric reform | 15.897 | 97.7 % |
| **+ trigger backfill (gated)** (this) | **inference-time** | **15.967** | **99.1 %** |

70%+ of cumulative lift came from non-LoRA changes — templates + 3 rubric
fixes + Stage-1 classifier features + this backfill. Same broader pattern:
*outside* the model is where the wins are.

## Ceiling analysis — what's left at 15.967

| sub-rubric | mean (max) | pass_count |
|---|---:|---:|
| schema_valid | 1.000 (1.0) | 1000 / 1000 |
| in_enum | 1.000 (1.0) | 1000 / 1000 |
| f1_triggers | 9.967 (10.0) | 991 / 1000 |
| prev_amount_correct | 2.000 (2.0) | 1000 / 1000 |
| no_hallucinated_facts | 1.000 (1.0) | 1000 / 1000 |
| no_hallucinated_facts_slots | 1.000 (1.0) | 1000 / 1000 |
| underpayment_ok | 0.500 (0.5) | 1000 / 1000 |
| well_formed | 0.500 (0.5) | 1000 / 1000 |
| **mean_total** | **15.967** (16.0) | **991 / 1000** |

99.8 % of theoretical max. Production-ready as-is.
