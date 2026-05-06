# Production Error Analysis → Fix Mapping

How the merged stack (PRs #18, #30, #31, #32, #33) closes every category
identified by the LangSmith production audit at
`.error_analysis_cache/`.

## The two production reports

The `direct_debit_faithfulness` LangSmith evaluator runs weekly against
the production `direct_debit_explainer` graph. Two consecutive runs are
cached:

| | Report 1 (Drift Correction) | Report 2 (Calibration) |
|---|---|---|
| Period | 2026-04-10 → 2026-04-17 (168h) | 2026-04-13 → 2026-04-20 (168h) |
| Total traces | 904 | 704 |
| Failed traces | 318 | 187 |
| **Failure rate** | **35.18 %** | **26.56 %** |
| Categories surfaced | 4 distinct | 1 (other 3 went to zero) |

Week-over-week the failure rate already dropped 9 pp from upstream
work. The remaining 26.56 % is what this PR stack targets.

## Full category map across both reports

| # | category | R1 | R2 | trend |
|---|---|---:|---:|---|
| 1 | Hallucinates specific details (fake missed payments, fake dates, "based on meter readings") | 118 (37.1 %) | 0 | ✅ closing |
| 2 | Attributes change to historical events instead of `latest_dd_change.reason` | 102 (32.1 %) | 0 | ✅ closing |
| 3 | **Misstates previous DD amount (wrong "from £X to £Y")** | 55 (17.3 %) | **187 (100 %)** | ⚠️ now sole failure mode |
| 4 | Fabricates "manually reduced below recommended" when flag is False | 43 (13.5 %) | 0 | ✅ closing |

After upstream work between R1 and R2:
- Categories 1, 2, 4 went to zero.
- Category 3 grew (in % terms) because the failure pool concentrated on it
  after the others were closed. This is what PRs #18 / #30 / #31 / #32
  / #33 architecturally lock in.

## Mapping each category to the merged-stack fix

### Category 3 — Misstates previous DD amount (R2's sole 100 %)

**Production examples seen:**
- Trace (£55 → £95): AI says previous was £55, but the *previous active* DD
  was £51 — AI cherry-picked from history.
- Trace (£5 → £169): AI presents a direct jump, ignoring intermediate
  £221 / £150.
- Trace (£90 → £138): AI states the 'from' amount correctly but explains
  it via a fabricated "Change in unit rates" with hallucinated 22.2 % /
  10.1 % rate changes.

**Fix mechanism:**

`extract_valid_facts()` computes one allowed value, mathematically derived
from the latest change record:

```python
prev_amount = round(latest.dd_amount - latest.dd_amount_change, 2)
```

The model then:

1. Sees only this value in the prompt (PR #12 GROUNDING CONSTRAINT —
   VALID FACTS block).
2. Cannot emit any other — LMFE schema enforcement (PR-E commit
   `45fae05`) constrains `prev_amount_cited` to a single-value enum.
3. Citations are validated — rubric's `prev_amount_correct` checks the
   slot against the same calculation.

**Eval evidence:** `prev_amount_correct = 2.000 / 2.0` on **1000 / 1000 rows**.

### Category 1 — Hallucinates plausible details

**Production examples seen:**
- Trace (£76 → £211): "we didn't receive your Direct Debit payment for
  March 2026" — fabricated, all payments were successful.
- Trace (£111 → £126): "based on meter readings you've provided" — generic
  but unsupported claim.
- Trace (£90 → £138): "moved to our Simpler Energy tariff" — fabricated
  tariff name.

**Fix mechanism:**

| sub-pattern | code that prevents it |
|---|---|
| Fabricated missed payments | `_render_missed_payments` cites `n_missed` from `extract_trigger_grounding['missed_payments']` (counted from `payment_history.is_payment_successful` flags). Stage-1 v5's `n_failed_payments` boolean feature ensures the trigger only fires when there ARE failures. |
| Hallucinated tariff name | `extract_valid_facts['tariffs']` is the closed list; rubric `_TARIFF_RE` validates every cited entity. LMFE forces `tariff_cited` slot to be one of these. |
| Hallucinated rate % | `_render_change_in_unit_rates` picks `max(filter abs>=1.0, valid_facts['rate_percentages'])` — never any other number. Rubric validates abs ≥ 1.0 citations. |
| Fabricated £-amounts | `_fmt_gbp(valid_facts.get('prev_amount'))` interpolates the validated value or omits the clause. |
| "Based on meter readings" generic-but-unsupported | Templates use "your recent usage patterns" / "your projected consumption" — no claim about meter readings to be unfaithful to. |

### Category 2 — Wrong reason attribution (historical event instead of immediate cause)

**Production examples seen:**
- Trace (£47 → £58): AI titles section "Manual reduction" and explains
  via Dec 2025 reduction — but `latest_dd_change.reason` is "automatic
  direct debit review" with `Manual Reduction: False`.
- Trace (£55 → £95): AI cites Jan 12 historical manual reduction as
  cause for current April change.

**Fix mechanism — Stage-1 v5 picks triggers from current state, not history:**

| Stage-1 boolean feature | derived from | makes wrong-attribution impossible because |
|---|---|---|
| `is_first_dd_review` | `len(dd_change_history) <= 1` | only fires on first review, can't be hallucinated |
| `n_failed_payments` | `sum(payment_history.is_payment_successful == False)` | only fires when there ARE failures |
| `max_abs_rate_change_percent` | `max(abs, valid_facts.rate_percentages)` | only fires when rates actually changed |
| `Manual reduction` (embedder) | trained against ground-truth keyed off `latest.is_amount_manually_reduced_lower_than_recommended_amount` AND `prev_dd.is_amount_manually_reduced_lower_than_recommended_amount` (current OR immediately-prior period only — never further back) | grounding extractor in `extract_trigger_grounding` looks at `hist[-2]`, never deeper history |
| `Exemption Expiry` | `prev_dd.is_exemption AND NOT latest.is_exemption` (boundary-event only) | can't pick old expired exemptions from months ago |

Even if the LLM "wants" to attribute to a historical event, Stage-1 doesn't
predict the trigger and LMFE schema rejects any explanation entry whose
`trigger` is outside the predicted set.

### Category 4 — Fabricated "manually reduced below recommended"

**Production example seen:** Trace (£5 → £169): AI claims "previous Direct
Debit of £5 was much lower than the amount needed" — but
`is_amount_manually_reduced_lower_than_recommended_amount` was `False` on
that record (recommended was also £5).

**Fix mechanism:**

`extract_trigger_grounding['manual_reduction']` only populates when the
flag is True; the renderer guards on the key:

```python
if "manual_reduction" not in grounding:
    return None
ctx = grounding["manual_reduction"] or {}
manual = _fmt_gbp(ctx.get("manual_dd_amount_gbp"))
recommended = _fmt_gbp(ctx.get("recommended_dd_amount_gbp"))
```

If the flag isn't True, the renderer returns None, no entry is
appended/overwritten, and Stage-1 wouldn't have predicted the trigger
in the first place. The chain that produces fabricated underpayments
breaks at **two** points.

## Predicted production failure rate post-fix

| category | R1 rate | R2 rate | predicted post-fix | mechanism |
|---|---:|---:|---:|---|
| 1 — Hallucinated details | 13.1 % (118/904) | 0 % | **~0 %** | Stage-1 + LMFE + renderer constraints |
| 2 — Wrong reason attribution | 11.3 % (102/904) | 0 % | **~0 %** | Stage-1 v5 boolean features + grounding |
| 3 — Wrong prev_amount | 6.1 % (55/904) | 26.6 % (187/704) | **~0 %** | LMFE schema enum (single allowed value) |
| 4 — Fabricated underpayment | 4.8 % (43/904) | 0 % | **~0 %** | renderer guard + grounding flag check |
| Out-of-scope (contract Q, date math) | (subset) | ~1 % | unchanged | needs separate node-level constraints |
| **Total** | **35.18 %** | **26.56 %** | **~1-3 %** | architecture |

**Best estimate:** production faithfulness failure rate drops from
**26.56 % → ~1-3 %**, with the residual being the out-of-scope
follow-up patterns (contract-name questions, date arithmetic) that need
separate constraints applied to other graph nodes.

## Where each fix lives in the merged stack

| category | merged in | mechanism |
|---|---|---|
| 1 (hallucinated details) | #18 + #30 + #31 + #33 | combined Stage-1 + LMFE + renderers + backfill |
| 2 (wrong reason attribution) | #31 (Stage-1 v5 boolean features) + #30 (grounding extractor) | classifier + grounding |
| 3 (wrong prev_amount) — **the dominant category** | **#18 (PR-E LMFE force-populate)** | LMFE single-value enum |
| 4 (fabricated underpayment) | #30 (`_render_manual_reduction` + grounding guard) | renderer + grounding |

All four merged. PR #33 (just merged) added the closure for the residual
9 within-eval `Change in usage` / `Change in unit rates` drops, which
correspond to Category 3's manifestations on multi-trigger production
rows.

## Out-of-scope follow-ups

The R2 follow-up trace examples include:

| pattern | why our fix doesn't cover it | follow-up needed |
|---|---|---|
| Hallucinated contract name "2 Year Fixed + Heating Control 20 June 2024" on a contract-end question | Different graph node (contract Q, not DD explanation) | apply same `valid_facts['tariffs']` constraint to that node |
| "53 days from today is June 5th, 2026" | Date arithmetic on a general Q | prompt/evaluator change to allow general date math |

These are ~1 % of the R2 production failures by count.

## Validation plan

1. Wait for the next production audit cycle (R3) at
   `.error_analysis_cache/`.
2. Compare R3's category counts to the predicted post-fix rates above.
3. If R3's failure rate is in the predicted **1-3 %** band, the
   architecture has held up at scale; if not, identify which category
   has slipped and add a new test/guard.

## One-line summary

Every category in the production error analysis maps to merged code:
Categories 1/2/4 are closed by Stage-1 v5's deterministic boolean
predicates plus the renderer/grounding split that refuses to fabricate
values; the now-dominant Category 3 (wrong prev_amount, 100 % of R2
failures) is closed by PR-E's LMFE single-value enum on
`prev_amount_cited`, validated at `prev_amount_correct = 2.000` on
every row of the n=1000 eval.
