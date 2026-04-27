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
RUBRIC_VERSION = "2026-04-27-cap-neg-tails-v2"

# Negative-tail caps for the two penalty-heavy rewards. v2_step_time_relax
# confirmed the v2 mean_total ceiling at ~9.6 is structural — the
# `no_hallucinated_facts` and `prev_amount_correct` rewards were dragging
# the mean by ~1 point each via their -3 failure mode. Capping the failure
# at -1 preserves the gradient signal (model still penalised for hallucination)
# but stops the negative tail from masking actual learning gains. f1_triggers
# (max 10) remains the dominant lever.
NO_HALLUC_FAIL_SCORE = -1.0     # was -3.0 in 2026-04-26 rubric
PREV_AMOUNT_FAIL_SCORE = -1.0   # was -3.0 in 2026-04-26 rubric


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


def _extract_text(parsed: Optional[DirectDebitExplainerResponse]) -> str:
    if parsed is None:
        return ""
    return " ".join(f"{e.header} {e.explanation}" for e in parsed.explanations)


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
    """+1.0 if every `explanations[i].trigger` is a valid `Trigger` enum value, else -1.0."""
    valid = {t.value for t in Trigger}
    scores: List[float] = []
    for c in completions:
        parsed = parse_response(c[0]["content"])
        if parsed is None:
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
    +2.0 if every cited previous-amount matches `latest.dd_amount - latest.dd_amount_change`
    within £0.01; 0 if none cited; -3.0 if any cited amount is wrong.
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        text = _extract_text(parse_response(c[0]["content"]))
        latest = inp["latest_dd_change"]
        expected = round(float(latest["dd_amount"]) - float(latest.get("dd_amount_change") or 0.0), 2)
        cited = [float(m.group(1)) for m in _PREV_AMOUNT_RE.finditer(text)]
        if not cited:
            scores.append(0.0)
            continue
        scores.append(2.0 if all(abs(v - expected) <= 0.01 for v in cited) else PREV_AMOUNT_FAIL_SCORE)
    return scores


def reward_no_hallucinated_facts(completions, input_json, **kwargs) -> List[float]:
    """Targets failure category: 'AI fabricates rate changes / hallucinates tariff names'.
    Tariff names cited must substring-match a real `contract.tariff_name`; rate %s > 1.0
    must be within ±0.5 of a real `change_since_previous_rate_percent`.
    """
    scores: List[float] = []
    for c, inp in zip(completions, input_json):
        text = _extract_text(parse_response(c[0]["content"]))
        if not text:
            scores.append(0.0)
            continue
        tariffs = {
            (ch.get("tariff_name", "") or "").lower()
            for ch in inp["account_context"].get("contract_history", []) or []
        }
        real_pcts: List[float] = []
        for ch in inp["account_context"].get("contract_history", []) or []:
            for rh in ch.get("contract_rates_history", []) or []:
                for rate in rh.get("rates", []) or []:
                    v = rate.get("change_since_previous_rate_percent")
                    if v is not None:
                        real_pcts.append(float(v))

        score = 1.0
        for m in _TARIFF_RE.finditer(text):
            cited = m.group(1).strip().lower()
            if not any(cited in t or t in cited for t in tariffs if t):
                score = NO_HALLUC_FAIL_SCORE
                break

        if score > 0:
            for m in _PERCENT_RE.finditer(text):
                cited_pct = float(m.group(1))
                if abs(cited_pct) < 1.0:
                    continue
                if not any(abs(cited_pct - rp) <= 0.5 for rp in real_pcts):
                    score = NO_HALLUC_FAIL_SCORE
                    break

        scores.append(score)
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


REWARD_FUNCS = [
    reward_schema_valid,
    reward_triggers_in_enum,
    reward_triggers_match_ground_truth,
    reward_previous_dd_amount_correct,
    reward_no_hallucinated_facts,
    reward_underpayment_language_constrained,
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
        "schema_valid":          reward_schema_valid(fake)[0],
        "in_enum":               reward_triggers_in_enum(fake)[0],
        "f1_triggers":           reward_triggers_match_ground_truth(fake, gt)[0],
        "prev_amount_correct":   reward_previous_dd_amount_correct(fake, inp)[0],
        "no_hallucinated_facts": reward_no_hallucinated_facts(fake, inp)[0],
        "underpayment_ok":       reward_underpayment_language_constrained(fake, inp)[0],
        "well_formed":           reward_explanations_well_formed(fake)[0],
    }
