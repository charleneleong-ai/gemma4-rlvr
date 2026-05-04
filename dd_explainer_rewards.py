"""Verifiable reward functions for GRPO on the direct_debit_explainer task.

Four groups:
 1. Shape   — `reward_schema_valid`, `reward_triggers_in_enum`
 2. Correctness — `reward_triggers_match_ground_truth` (F1 vs generator GT)
 3. Production failure categories (LangSmith error-analysis reports):
    - `reward_previous_dd_amount_correct`
    - `reward_no_hallucinated_facts`
    - `reward_underpayment_language_constrained`
 4. Well-formed — `reward_explanations_well_formed`

`REWARD_FUNCS` is the canonical list to pass to `GRPOTrainer(reward_funcs=...)`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from dd_explainer_data_generator import DirectDebitExplainerResponse, Trigger


# Bumped whenever a reward function's scoring formula changes — written into
# `_aggregate_scores` output so charts/results.jsonl don't silently mix
# rubric versions across runs. Format: YYYY-MM-DD-shortdesc.
RUBRIC_VERSION = "2026-05-04-inaction-loophole-fix-v2"

# Reverted to the uncapped rubric (matching E1 champion) for the data-regen
# experiment. Rationale: E14 showed that capping the no_halluc penalty makes
# f1 a trade-off variable — the model retreats from f1 to gain ground on
# no_halluc. To isolate whether the new multi-trigger data lifts f1's
# absolute ceiling, we need the original gradient pressure restored.
# (See docs/ceiling-diagnosis-2026-04-27.md "Decision" section.)
NO_HALLUC_FAIL_SCORE = -3.0
PREV_AMOUNT_FAIL_SCORE = -3.0


# =============================================================================
# Shared helpers
# =============================================================================


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[str]:
    """Return the first JSON object substring in `text`, stripping markdown fences."""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_response(text: str) -> Optional[DirectDebitExplainerResponse]:
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return DirectDebitExplainerResponse.model_validate_json(raw)
    except Exception:
        return None


def _render_explanation(e) -> str:
    """Substitute structured slot values into `{tariff_cited}` / `{rate_change_pct_cited}` /
    `{prev_amount_cited}` placeholders in the explanation prose. Old-format
    completions (slots None) pass through unchanged.

    The user-visible prose IS the rendered string — slot enforcement only
    helps no_halluc / prev_amount_correct if the rendered output actually
    contains the valid slot value, so this is what the regex rubric scans.
    """
    text = f"{e.header} {e.explanation}"
    if getattr(e, "tariff_cited", None):
        text = text.replace("{tariff_cited}", e.tariff_cited)
    rate = getattr(e, "rate_change_pct_cited", None)
    if rate is not None:
        # Format with the precision the rubric expects (1 decimal); strip leading "+"
        text = text.replace("{rate_change_pct_cited}", f"{rate:+.1f}".lstrip("+"))
    prev_amount = getattr(e, "prev_amount_cited", None)
    if prev_amount is not None:
        # Format as "£<value>" with up to 2 decimals; the prev_amount rubric
        # regex expects a "£<num>" pattern preceded by "previous|before|was|
        # prior|old" cue words, so the placeholder is typically used like
        # "previous DD was {prev_amount_cited}".
        text = text.replace("{prev_amount_cited}", f"£{float(prev_amount):.2f}")
    return text


def _extract_text(parsed: Optional[DirectDebitExplainerResponse]) -> str:
    if parsed is None:
        return ""
    return " ".join(_render_explanation(e) for e in parsed.explanations)


def _f1(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gt)
    return 2 * prec * rec / (prec + rec)


# =============================================================================
# Shape: schema + enum validity
# =============================================================================


def reward_schema_valid(completions, **kwargs) -> List[float]:
    """+1.0 if the completion parses as `DirectDebitExplainerResponse`, else -2.0."""
    return [1.0 if parse_response(c[0]["content"]) is not None else -2.0 for c in completions]


def reward_triggers_in_enum(completions, **kwargs) -> List[float]:
    """+1.0 if every `explanations[i].trigger` is a valid `Trigger` enum value, else -1.0.

    Inaction loophole fix (PR-F audit, 2026-05-04): empty `explanations: []`
    returns -1.0 instead of +1.0. Previously `all(...)` over an empty list was
    vacuously True, so the model could earn +1 here (and +1 from
    `reward_schema_valid`) by emitting `{"explanations": []}` — a 2-point
    free ride for outputting nothing on a task where every dataset row has
    1-2 ground-truth triggers (audit: 4772 rows with 1, 728 rows with 2,
    0 with 0).
    """
    valid = {t.value for t in Trigger}
    scores: List[float] = []
    for c in completions:
        parsed = parse_response(c[0]["content"])
        if parsed is None or not parsed.explanations:
            scores.append(-1.0)
            continue
        scores.append(1.0 if all(e.trigger.value in valid for e in parsed.explanations) else -1.0)
    return scores


# =============================================================================
# Correctness: F1 vs generator ground truth
# =============================================================================


def reward_triggers_match_ground_truth(
    completions, ground_truth_triggers, **kwargs
) -> List[float]:
    """F1 between predicted and GT trigger sets, scaled to [-2.0, +10.0]. Main RL signal."""
    scores: List[float] = []
    for c, gt_list in zip(completions, ground_truth_triggers):
        parsed = parse_response(c[0]["content"])
        gt = set(gt_list)
        if parsed is None:
            scores.append(-2.0 if gt else 0.0)
            continue
        pred = {e.trigger.value for e in parsed.explanations}
        scores.append(-2.0 + 12.0 * _f1(pred, gt))
    return scores


# =============================================================================
# Production failure categories (from .error_analysis_cache/*.md)
# =============================================================================


_PREV_AMOUNT_RE = re.compile(
    r"\b(?:previous|before|was|prior|old)[^.!?\n]{0,60}?£\s*(\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_TARIFF_RE = re.compile(
    r"(?:tariff|contract|plan)\s+(?:called\s+|named\s+)?['\"]?([A-Z][A-Za-z0-9 &-]{2,40})['\"]?"
)
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_UNDERPAY_RE = re.compile(r"\bunderpa(?:y|id|ying|yment)\b", re.IGNORECASE)


def reward_previous_dd_amount_correct(completions, input_json, **kwargs) -> List[float]:
    """Targets failure category: 'AI incorrectly identifies previous DD amount'.

    Two paths:
      - Slot-aware (PR-E, when `prev_amount_cited` is populated): validate the
        slot value directly against expected = latest.dd_amount - dd_amount_change.
        +2.0 if within £0.01, -3.0 otherwise. Forces every row to cite (vs the
        legacy regex which returned 0 for "no citation").
      - Legacy regex (back-compat for pre-PR-E completions): scan rendered prose
        for "previous|before|was|prior|old ... £<num>" cues. +2 / 0 / -3 as before.

    Slot path is preferred whenever any explanation populates `prev_amount_cited`.
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        parsed = parse_response(c[0]["content"])
        latest = inp["latest_dd_change"]
        expected = round(
            float(latest["dd_amount"]) - float(latest.get("dd_amount_change") or 0.0),
            2,
        )

        any_slot = False
        slot_invalid = False
        if parsed is not None:
            for e in parsed.explanations:
                slot_val = getattr(e, "prev_amount_cited", None)
                if slot_val is not None:
                    any_slot = True
                    if abs(float(slot_val) - expected) > 0.01:
                        slot_invalid = True
                        break
        if any_slot:
            scores.append(PREV_AMOUNT_FAIL_SCORE if slot_invalid else 2.0)
            continue

        # Legacy regex path — pre-PR-E completions or rows where the model
        # didn't populate the slot. Used at training time (no LMFE) to push
        # the policy toward citing instead of abstaining.
        text = _extract_text(parsed)
        cited = [float(m.group(1)) for m in _PREV_AMOUNT_RE.finditer(text)]
        if not cited:
            # Inaction loophole fix (PR-F audit, 2026-05-04): when the row
            # HAS a computable prev_amount (100% of dataset rows do), no
            # citation is always wrong. Penalise at -1 — softer than -3 for
            # an outright wrong cite (so the policy still prefers attempting
            # over guessing wildly), but firmly below 0 so abstaining loses
            # to a correct attempt at +2 within every GRPO group.
            scores.append(-1.0)
            continue
        scores.append(2.0 if all(abs(v - expected) <= 0.01 for v in cited) else PREV_AMOUNT_FAIL_SCORE)
    return scores


def reward_no_hallucinated_facts(completions, input_json, **kwargs) -> List[float]:
    """Targets failure category: 'AI fabricates rate changes / hallucinates tariff names'.

    Tariff names cited must substring-match a real `contract.tariff_name`. Cited
    %s with magnitude >= 1.0 must be within +/-0.5 of either a rate-change %
    (`contract_history.contract_rates_history.rates.change_since_previous_rate_percent`)
    OR a usage-change % (`projected_consumption_history.<fuel>.change_percent`).

    The usage-aware path (added 2026-05-03) closes the rubric bug that fired on
    every `Change in usage` row: prose about "your usage went up 12%" was being
    validated against rate-change %s only, so usage citations always failed.

    Inaction loophole fix (2026-05-04): text that cites zero facts but the input
    HAS citable tariffs/percents now scores 0.0 (neutral) instead of +1.0.
    Prevents the model from earning full credit by avoiding citations entirely.
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        text = _extract_text(parse_response(c[0]["content"]))
        if not text:
            scores.append(0.0)
            continue
        tariffs, real_pcts = _allowed_facts(inp)

        score = 1.0
        any_citation = False
        for m in _TARIFF_RE.finditer(text):
            any_citation = True
            cited = m.group(1).strip().lower()
            if not any(cited in t or t in cited for t in tariffs if t):
                score = NO_HALLUC_FAIL_SCORE
                break

        if score > 0:
            for m in _PERCENT_RE.finditer(text):
                cited_pct = float(m.group(1))
                if abs(cited_pct) < 1.0:
                    continue
                any_citation = True
                if not any(abs(cited_pct - rp) <= 0.5 for rp in real_pcts):
                    score = NO_HALLUC_FAIL_SCORE
                    break

        # Inaction loophole: text exists but cites no facts while facts
        # ARE available -> neutral (0) rather than full credit (+1).
        if score > 0 and not any_citation and (tariffs - {''} or real_pcts):
            score = 0.0

        scores.append(score)
    return scores


def reward_no_hallucinated_facts_granular(completions, input_json, **kwargs) -> List[float]:
    """Per-fact partial credit variant of `reward_no_hallucinated_facts`.

    Fixes the binary reward problem: original returns +1 / -3 for any-invalid /
    all-valid, identical for "3-of-4 valid" and "0-of-4 valid". This gives no
    within-group gradient under GRPO when most generations hallucinate.

    Granular score = -1.0 + 2.0 × (n_valid / n_total). Range [-1, +1], smooth
    between the extremes. "No citations" returns +1 only when there are no
    citable facts in the input; returns 0 when facts exist but the model cites
    none (inaction loophole fix, 2026-05-04). Empty completion returns 0.

    Like the binary variant, the allowed-% list combines rate-change %s and
    consumption-change %s (usage-aware rubric, 2026-05-03).
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        text = _extract_text(parse_response(c[0]["content"]))
        if not text:
            scores.append(0.0)
            continue
        tariffs, real_pcts = _allowed_facts(inp)

        cited_tariffs = [m.group(1).strip().lower() for m in _TARIFF_RE.finditer(text)]
        cited_pcts = [
            float(m.group(1)) for m in _PERCENT_RE.finditer(text)
            if abs(float(m.group(1))) >= 1.0
        ]
        n_total = len(cited_tariffs) + len(cited_pcts)
        if n_total == 0:
            # Inaction loophole: no citations but facts available -> neutral
            has_citable = bool(tariffs - {''}) or bool(real_pcts)
            scores.append(0.0 if has_citable else 1.0)
            continue
        n_valid = sum(
            1 for cited in cited_tariffs
            if any(cited in t or t in cited for t in tariffs if t)
        ) + sum(
            1 for pct in cited_pcts
            if any(abs(pct - rp) <= 0.5 for rp in real_pcts)
        )
        scores.append(-1.0 + 2.0 * (n_valid / n_total))
    return scores


def _allowed_facts(inp: Dict[str, Any]) -> tuple[set, List[float]]:
    """Pull (tariffs_lower, allowed_pcts) from input_json.

    `allowed_pcts` combines:
      - rate-change %s from contract_history.contract_rates_history.rates
      - consumption-change %s from projected_consumption_history.<fuel>.change_percent
      - the absolute value of every signed % above (so prose like
        "decreased by 12.38%" — verbal direction + positive magnitude —
        validates against an input stored as -12.38)

    The usage path (added 2026-05-03) makes the rubric accept legitimate usage
    citations on `Change in usage` rows. The abs() variant added shortly after
    closes a sign-flip bug where every Change-in-usage decrease cited as
    positive prose failed against the negative input value.
    """
    ac = inp.get("account_context", {}) or {}
    tariffs = {
        (ch.get("tariff_name", "") or "").lower()
        for ch in ac.get("contract_history", []) or []
    }
    pcts: List[float] = []
    for ch in ac.get("contract_history", []) or []:
        for rh in ch.get("contract_rates_history", []) or []:
            for rate in rh.get("rates", []) or []:
                v = rate.get("change_since_previous_rate_percent")
                if v is not None:
                    pcts.append(float(v))
    pch = ac.get("projected_consumption_history") or {}
    if isinstance(pch, dict):
        for fuel_data in pch.values():
            if not isinstance(fuel_data, dict):
                continue
            v = fuel_data.get("change_percent")
            if v is not None:
                pcts.append(float(v))
    # Accept either sign — prose like "decreased by 12%" cites a positive
    # magnitude even when the underlying delta is stored as -12.
    pcts.extend(abs(v) for v in list(pcts) if v != 0)
    return tariffs, pcts


def reward_no_hallucinated_facts_slots(completions, input_json, **kwargs) -> List[float]:
    """Slot-only variant — validates `tariff_cited` / `rate_change_pct_cited` slot
    fields directly against the input_json allowed-list. Skips prose regex.

    +1.0 if every populated slot is in the allowed-list.
    NO_HALLUC_FAIL_SCORE if any slot is invalid.
    Returns the granular legacy score if NO slot is populated (back-compat path
    for old-format completions during the migration window).

    Diagnostic intended to live alongside the legacy `reward_no_hallucinated_facts`
    so we can quantify the gap between "slot validity" and "rendered prose
    cleanliness" — only the latter is what the user reads, so the latter is
    the apples-to-apples lift number across experiments.
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        parsed = parse_response(c[0]["content"])
        if parsed is None or not parsed.explanations:
            scores.append(0.0)
            continue
        tariffs, real_pcts = _allowed_facts(inp)
        any_slot_populated = False
        invalid = False
        for e in parsed.explanations:
            if e.tariff_cited:
                any_slot_populated = True
                cited = e.tariff_cited.strip().lower()
                if not any(cited in t or t in cited for t in tariffs if t):
                    invalid = True
                    break
            if e.rate_change_pct_cited is not None:
                any_slot_populated = True
                cited_pct = float(e.rate_change_pct_cited)
                if abs(cited_pct) >= 1.0 and not any(
                    abs(cited_pct - rp) <= 0.5 for rp in real_pcts
                ):
                    invalid = True
                    break
        if not any_slot_populated:
            # Old-format completion — defer to granular prose-regex score
            scores.append(reward_no_hallucinated_facts_granular([c], [inp])[0])
            continue
        scores.append(NO_HALLUC_FAIL_SCORE if invalid else 1.0)
    return scores


def reward_underpayment_language_constrained(completions, input_json, **kwargs) -> List[float]:
    """Targets failure category: 'AI uses underpayment language inappropriately'.
    Only allowed when a prior DDChange has
    `is_amount_manually_reduced_lower_than_recommended_amount=True`.
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        text = _extract_text(parse_response(c[0]["content"]))
        if not _UNDERPAY_RE.search(text):
            scores.append(0.5)
            continue
        prior_manual = any(
            ch.get("is_amount_manually_reduced_lower_than_recommended_amount", False)
            for ch in inp["account_context"].get("dd_change_history", []) or []
        )
        scores.append(0.5 if prior_manual else -1.5)
    return scores


# =============================================================================
# Shape: explanation length
# =============================================================================


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _explanation_well_formed(e) -> bool:
    """Per-explanation predicate: header ≤ 10 words AND 1-3 sentences."""
    if len(e.header.split()) > 10:
        return False
    n_sentences = sum(
        1 for s in _SENTENCE_SPLIT_RE.split(e.explanation) if s.strip()
    )
    return 1 <= n_sentences <= 3


def reward_explanations_well_formed(completions, **kwargs) -> List[float]:
    """Row score is proportional to the fraction of explanations that pass the
    per-explanation shape check (header ≤ 10 words, 1-3 sentences).

    Range stays [-0.5, +0.5]:
    - all explanations pass → +0.5 (full credit, same as before)
    - all explanations fail → -0.5 (worst case, same as before)
    - partial pass: linear interpolation, e.g. 2/3 pass → +0.166

    Replaces an all-or-nothing AND that scored ~40% pass on this dataset
    purely because rows have 2-3 explanations and one bad one failed the
    whole row. Fractional credit gives the model gradient signal for partial
    wins instead of treating "2 good, 1 bad" identically to "all bad".
    """
    scores: List[float] = []
    for c in completions:
        parsed = parse_response(c[0]["content"])
        if parsed is None or not parsed.explanations:
            scores.append(-0.5)
            continue
        n_ok = sum(1 for e in parsed.explanations if _explanation_well_formed(e))
        n_total = len(parsed.explanations)
        scores.append(-0.5 + 1.0 * (n_ok / n_total))
    return scores


# =============================================================================
# Canonical bundle for GRPOTrainer
# =============================================================================


def make_weighted_no_halluc(weight: float, base_fn=None):
    """Factory for a weighted no_halluc reward. weight=1.0 returns base_fn
    untouched; otherwise multiplies every score by `weight`.

    base_fn defaults to `reward_no_hallucinated_facts` (binary). Pass
    `reward_no_hallucinated_facts_granular` to weight the granular variant.
    """
    if base_fn is None:
        base_fn = reward_no_hallucinated_facts
    if weight == 1.0:
        return base_fn

    def weighted(*args, **kwargs) -> List[float]:
        return [weight * s for s in base_fn(*args, **kwargs)]

    weighted.__name__ = "reward_no_hallucinated_facts"  # GRPOTrainer logs by name
    return weighted


REWARD_FUNCS = [
    reward_schema_valid,
    reward_triggers_in_enum,
    reward_triggers_match_ground_truth,
    reward_previous_dd_amount_correct,
    reward_no_hallucinated_facts,
    # reward_underpayment_language_constrained -- demoted to eval-only diagnostic
    # (2026-05-04): dead weight, 200/200 pass, 0.5 max on every config shipped.
    # Kept in score_completion() for dashboard visibility.
    reward_explanations_well_formed,
]


# =============================================================================
# Manual scoring helper (used by regress subcommand for real LangSmith traces)
# =============================================================================


def score_completion(
    completion_text: str,
    ground_truth_triggers: List[str],
    input_json: Dict[str, Any],
) -> Dict[str, float]:
    """Score a single (completion, GT, input) tuple on all 7 rewards. Returns a flat dict."""
    fake = [[{"content": completion_text}]]
    gt = [ground_truth_triggers]
    inp = [input_json]
    return {
        "schema_valid":               reward_schema_valid(fake)[0],
        "in_enum":                    reward_triggers_in_enum(fake)[0],
        "f1_triggers":                reward_triggers_match_ground_truth(fake, gt)[0],
        "prev_amount_correct":        reward_previous_dd_amount_correct(fake, inp)[0],
        "no_hallucinated_facts":      reward_no_hallucinated_facts(fake, inp)[0],
        "no_hallucinated_facts_slots": reward_no_hallucinated_facts_slots(fake, inp)[0],
        "underpayment_ok":            reward_underpayment_language_constrained(fake, inp)[0],
        "well_formed":                reward_explanations_well_formed(fake)[0],
    }
