# `v2_slot_grounded` — structured slot grounding for citation-eligible facts

**Schedule:** ad-hoc — `scripts/two_stage_eval.py --enforce-slots` (PR-B) is the A/B harness.
**Config slot:** `dd_explainer/two_stage` (no Gemma fine-tune; reuses E18's adapter).
**Chassis:** `BAAI/bge-small-en-v1.5` Stage 1 + 2-layer MLP head + `unsloth/gemma-4-E4B-it` E18 LoRA + slot-enforced JSON decoding (PR-B).
**Hardware:** A100 PCIe 80GB.
**Branch:** `feat/slot-grounded-schema` (PR-A foundation; PR-B will add the decoder).
**Driving doc:** [`v1_constrained.md`](v1_constrained.md) — partial-win on no_halluc (-0.848 → -0.732); didn't clear the -0.5 huge-win threshold.

## Hypothesis

PR #12 proved the no_halluc plateau is **constraint-adherence-bound**, not absolute capability bound (prompt-time allowed-list moved no_halluc from -0.848 to -0.732). The remaining gap to -0.5: at production context lengths the prompt-time "GROUNDING CONSTRAINT — VALID FACTS" block scrolls out of attention, so the LLM partially reverts to fabricating facts.

**Hypothesis:** if we move citation-eligible facts (tariff names, rate change percentages) out of free prose and into **structured slots** with values constrained by JSON schema enforcement, hallucinated facts become mechanically impossible — the LLM cannot emit slot values outside the allowed-list, and the rendered prose substitutes those slot values verbatim.

This is "logit masking" in the sense the production frameworks (Outlines, lm-format-enforcer) actually use it: token-level masking on the structured slot fields, not on free prose. Free prose stays free; only the slots are masked.

**Falsification thresholds at n=1000:**
- `no_halluc_prose ≥ -0.3` → **huge win**, ship as new champion (slots help even with E18 untrained on the new schema).
- `no_halluc_prose ≥ -0.5` → **partial win**, ship; still need a Stage 2 retrain (PR-C) for the model to fully use slots.
- `no_halluc_prose < -0.7` → **failed at PR-B**; E18 doesn't know how to populate slots without retraining. Pivot to PR-C (Stage 2 GRPO retrain on the new schema).

The two reported numbers:
- **`no_hallucinated_facts`** (legacy, regex on rendered prose) — apples-to-apples vs PR #12 baselines. THIS is the verdict number.
- **`no_hallucinated_facts_slots`** (new, structured slot membership) — diagnostic. With slot enforcement at decode time, this is +1.0 by construction. Tells us whether slot validity translates to clean prose.

## Implementation phases

### Phase 1 — schema foundation (this PR-A)

Schema, rubric, prompt-template additions only. No model retraining, no inference-time decoder change. Backwards-compatible: old-format completions (no slot fields) score under the legacy regex path.

1. **`TriggerExplanation`** in `dd_explainer_data_generator.py:210` — adds two optional fields:
   - `tariff_cited: Optional[str]` — verbatim tariff name from input contract_history.
   - `rate_change_pct_cited: Optional[float]` — verbatim rate change percentage.
   Both default `None` so unchanged completions still validate.

2. **`_render_explanation`** in `dd_explainer_rewards.py` — substitutes slot values into `{tariff_cited}` / `{rate_change_pct_cited}` placeholders in the explanation prose. The user-visible string IS the rendered string.

3. **`reward_no_hallucinated_facts`** unchanged — still scans `_extract_text(parsed)` which now substitutes slots first, then runs regex. So existing experiments that emit no slots get exactly the same score; new slot-aware completions get scored against the rendered prose.

4. **`reward_no_hallucinated_facts_slots`** — new, slot-only validator. Returns +1.0 if every populated slot is in the allowed-list, fail otherwise. For old-format completions falls back to the granular legacy score so the metric is meaningful in mixed cohorts.

5. **`SYSTEM_PROMPT`** at `dd_explainer_data_generator.py:315` — augmented to instruct the model to populate slots verbatim and reference them as `{tariff_cited}` / `{rate_change_pct_cited}` placeholders inside the explanation prose.

6. **`tests/test_slot_rubric.py`** — 7 tests covering old-format passthrough, new-format slot validation, slot-valid-but-prose-extra-hallucination edge case, render substitution.

7. **`RUBRIC_VERSION`** bumped to `2026-05-02-slot-grounded`. The slot diagnostic is added to `_aggregate_scores` as a side metric — does NOT enter `mean_total` or `pass_all` so v1_constrained / v0_pipeline / E18 baselines stay directly comparable.

### Phase 2 — slot enforcement at decode time (PR-B)

Add `lm-format-enforcer` (or Outlines) to `pyproject.toml`. Build a `LogitsProcessor` that constrains:
- `tariff_cited` ∈ `extract_valid_facts(input_json)["tariffs"]`
- `rate_change_pct_cited` ∈ `extract_valid_facts(input_json)["rate_percentages"]`

Test at n=20 smoke, then n=1000 A/B vs PR #12's `--constrain-facts` baseline (mean_total=10.961, no_halluc=-0.732).

### Phase 3 — Stage 2 GRPO retrain (PR-C, conditional on PR-B verdict)

If PR-B clears `no_halluc_prose ≥ -0.5` on E18 alone, we ship and skip PR-C. If E18 doesn't know how to populate slots well (the regression in well_formed exceeds the slot lift), retrain a new adapter that learns the schema. Sweep recipe will live at `configs/schedules/v2_slot_grounded.yaml`.

## Verdict

**TBD** — Phase 1 (this PR) ships the foundation. PR-B will run the n=1000 A/B and post the verdict here.

## What this PR ships (PR-A)

- Schema additions + rubric updates (backwards-compatible).
- Updated prompt template instructing slot population.
- 7 unit tests; pytest infrastructure added to `pyproject.toml`.
- `RUBRIC_VERSION` bump.

No baselines change. Mean_total comparable across all prior experiments. The slot diagnostic exposes a new side-metric without affecting the canonical scores.

## Why this isn't a metric play

The honest concern: would slot-only scoring "launder" hallucinations by validating slots while ignoring prose? The two-number design addresses this:
- `no_hallucinated_facts` (rendered prose) is what we compare against PR #12's baseline.
- `no_hallucinated_facts_slots` (slot validity) is the diagnostic.

If the rendered prose still contains hallucinations (the LLM cites extra tariffs outside the slots), the prose number penalises it. The slot number being +1.0 alone wouldn't ship — we need the prose number to clear the threshold.

## Next move

Open PR-A. Once merged, build PR-B (slot decoder + n=1000 A/B). PR-C only if PR-B doesn't clear -0.5 on E18.
