# well_formed Rubric Fix Retrospective

Decimal-aware sentence splitter for `reward_explanations_well_formed`. Pure
inference-time rescore on the existing E28+v5+all per-row dump
(`data/eval_e28_v5_templates_n1000.per_row.jsonl`). No regeneration, no
retrain. Bug fix only.

## Headline — biggest single lift since rubric reform

| metric | E28 + v5 + all (prior champion) | **+ well_formed fix** | Δ |
|---|---:|---:|---:|
| mean_total | 15.634 | **15.897** | **+0.263** |
| f1_triggers (mean) | 9.897 | 9.897 | preserved |
| no_halluc (row-level) | 1.000 | 1.000 | preserved |
| no_halluc (slot-only) | 1.000 | 1.000 | preserved |
| **well_formed (mean)** | 0.237 | **0.500** ✓ perfect | **+0.263** |
| **pass_all_pct** | 69.1% | **97.7%** | **+28.6 pp** |
| pass_all (count) | 691 | **977** | +286 |

**977/1000 rows now pass all 7 rubric components.** Only 23 rows still drop
points anywhere (those are the residual f1<10 rows from Stage-1 misses on
exotic 4-trigger combos — separate ceiling).

vs PR-F champion (14.014): **+1.883 mean_total, ~8.4× pass_all.**

## Diagnosis

The E28+v5+all retrospective showed `well_formed` was the only sub-rubric
not at ceiling: mean 0.237 with pass_count 714/1000. The failing 286 rows
averaged -0.42 (near all-fail floor).

Per-row analysis on the per_row dump showed:

| failure mode | count |
|---|---:|
| header > 10 words | **0** |
| > 4 sentences | **287** (all of them) |
| 0 sentences | 0 |

Failures clustered on **2 specific trigger combos**:

| stage-1 trigger combo | fail rate |
|---|---:|
| Change in usage (alone) | **99.3%** (141/142) |
| Change in unit rates (alone) | **100%** (104/104) |
| Change in unit rates + Change in usage | 95.0% (19/20) |

Inspecting full prose for these rows revealed they were **3-sentence prose
ending cleanly with periods**, NOT 5-sentence sprawl. Example failure:

> "Your Direct Debit has been adjusted because of changes in the unit rates
> for your tariff Better Energy Fixed. Specifically, the electricity unit
> rate saw a change of -3.73% since the last review. This change resulted
> in the new DD amount being lower than the previous amount of £60.29."

3 real sentences, but the rubric counted 5. The sentence splitter regex
`re.split(r"[.!?]+", text)` was matching **decimal periods inside numbers**
("£60.29" → "60" / "29", "12.38%" → "12" / "38"), inflating the count.

Trigger-specific because these prose templates always cite a percentage
AND a previous-DD £-amount → 2-3 spurious decimal "sentences" added to
the real 3.

## The fix

```python
# Before:
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

# After:
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?=\s|$)")
```

The lookahead requires the punctuation to be followed by whitespace or
end-of-string. Skips decimals like "£174.64" / "12.38%" / "v2.5" / dates
like "2025-07-29." (where the trailing period IS sentence-ending — has
whitespace or EOS after).

`RUBRIC_VERSION` bumped to `2026-05-06-well-formed-decimal-aware-split` so
prior eval JSONs don't silently mix scoring rules.

## Tests

- **Positive case**: 3-sentence prose with `£60.29` and `-3.73%` decimals →
  `well_formed = +0.5` (full pass). Pre-fix, this scored `-0.5`.
- **Negative case**: 5 actual sentences (each ending with period+space) →
  `well_formed = -0.5`. Confirms the splitter still rejects genuinely too-long prose.
- All 27 prior tests pass.

## Why this matters — and why it took until now

This bug was latent the entire project. Every prior eval since the rubric
was introduced has been silently undercounting well_formed pass rates on
prose containing £-amounts or %-amounts.

Why it didn't surface earlier:
1. Pre-PR-F, the rubric capped at 3 sentences and the LLM-generated prose was
   already 4-5 actual sentences, so well_formed mostly failed for *real*
   reasons — the decimal artifact was masked.
2. PR-F's 3 → 4 cap moved the threshold to where the 3-real-sentence prose
   would pass IF the count were correct, but the bug then made it fail by
   counting 5.
3. The bug only became diagnosable once the OTHER rubric components were
   maxed out (no_halluc, f1, pass_all on slots). With those clean, the
   per-row dump showed unambiguously that "Change in usage" rows were
   failing well_formed despite having clean 3-sentence prose.

## Cumulative roadmap

| step | mechanism | mean_total | pass_all |
|---|---|---:|---:|
| PR-F champion | LMFE force-populate + rubric reform | 14.014 | 11.7% |
| E28 (PR-G) | retrain w/ --constrain-facts | 14.396 | 25.8% |
| + lonely-trigger templates | inference-time | 14.684 | 44.1% |
| + `_TARIFF_RE` regex fix | rubric reform | 14.765 | 48.3% |
| + No-triggers fallback template | inference-time | 14.787 | 49.7% |
| + Stage-1 v5 (3 boolean features) | classifier retrain | **15.634** | **69.1%** |
| **+ well_formed regex fix** (this) | **rubric reform** | **15.897** | **97.7%** |

70%+ of the cumulative lift came from non-LoRA changes (templates + 3 rubric
fixes + Stage-1 classifier features). Matches the broader pattern: when
the eval rubric is mismatched to the actual evaluation goal, fixing the
rubric is higher-leverage than training the model harder.

## Ceiling analysis — what's left at 15.897

| sub-rubric | mean (max) | pass_count |
|---|---:|---:|
| schema_valid | 1.000 (1.0) | 1000/1000 |
| in_enum | 1.000 (1.0) | 1000/1000 |
| f1_triggers | 9.897 (10.0) | 977/1000 |
| prev_amount_correct | 2.000 (2.0) | 1000/1000 |
| no_hallucinated_facts | 1.000 (1.0) | 1000/1000 |
| no_hallucinated_facts_slots | 1.000 (1.0) | 1000/1000 |
| underpayment_ok | 0.500 (0.5) | 1000/1000 |
| well_formed | 0.500 (0.5) | 1000/1000 |
| **mean_total** | **15.897** (16.0) | **977/1000** |

**Only f1_triggers is below ceiling** (23 rows have f1<10). These are
exotic 4-trigger combos where Stage-1 v5 still misclassifies one trigger
or the LLM fails to emit all 4 explanations. mean_total ceiling is 16.0;
we're at 15.897 (99.4% of theoretical max).

Closing the last 0.103 would need either:
- Deeper Stage-1 features for 4-trigger discrimination (~+0.05)
- Templated explanation list for high-trigger-count rows (skip LLM,
  programmatically generate one explanation per Stage-1 trigger using
  the template library) (~+0.05)

But these are diminishing returns — the headline is **97.7% pass_all** is
production-ready as-is.
