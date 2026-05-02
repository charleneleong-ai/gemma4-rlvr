"""Smoke tests for the slot-grounded schema + rubric (PR-A).

Verifies that:
- Old-format completions (no slot fields) score under the legacy regex path.
- New-format completions populate the structured `tariff_cited` /
  `rate_change_pct_cited` slots and score correctly.
- The two diagnostic numbers — `no_hallucinated_facts` (rendered-prose regex,
  apples-to-apples vs PR #12) and `no_hallucinated_facts_slots` (structured
  slot membership) — diverge in the expected ways.

Run with:  uv run pytest tests/test_slot_rubric.py -v
"""

from __future__ import annotations

import json

import pytest

from dd_explainer_rewards import (
    _render_explanation,
    parse_response,
    score_completion,
)


@pytest.fixture
def input_json():
    return {
        "account_context": {
            "contract_history": [{
                "tariff_name": "Simply Fixed",
                "contract_rates_history": [
                    {"rates": [{"change_since_previous_rate_percent": 12.3}]}
                ],
            }],
        },
        "latest_dd_change": {"dd_amount": 100.0, "dd_amount_change": 10.0},
    }


@pytest.fixture
def gt():
    return ["Change in unit rates"]


def _completion(**fields) -> str:
    base = {
        "trigger": "Change in unit rates",
        "header": "Rates went up",
        "explanation": "Tariff Simply Fixed went up by 12.3%.",
    }
    base.update(fields)
    return json.dumps({"explanations": [base]})


def test_old_format_valid_citation_scores_high(input_json, gt):
    s = score_completion(_completion(), gt, input_json)
    assert s["no_hallucinated_facts"] == 1.0
    assert s["no_hallucinated_facts_slots"] == 1.0  # falls back to granular = +1


def test_old_format_hallucinated_tariff_penalised(input_json, gt):
    c = _completion(explanation="Your tariff Fake-Plan went up by 12.3%.")
    s = score_completion(c, gt, input_json)
    assert s["no_hallucinated_facts"] < 0  # legacy regex catches it
    # Slots fallback to granular: 1-of-2 facts valid → 0.0
    assert s["no_hallucinated_facts_slots"] == pytest.approx(0.0, abs=1e-3)


def test_new_format_valid_slots_pass_both(input_json, gt):
    c = _completion(
        explanation="Tariff {tariff_cited} went up by {rate_change_pct_cited}%.",
        tariff_cited="Simply Fixed",
        rate_change_pct_cited=12.3,
    )
    s = score_completion(c, gt, input_json)
    assert s["no_hallucinated_facts"] == 1.0  # rendered prose has valid citations
    assert s["no_hallucinated_facts_slots"] == 1.0


def test_new_format_invalid_slots_fail_both(input_json, gt):
    c = _completion(
        explanation="Tariff {tariff_cited} went up by {rate_change_pct_cited}%.",
        tariff_cited="Fake-Plan",
        rate_change_pct_cited=99.9,
    )
    s = score_completion(c, gt, input_json)
    assert s["no_hallucinated_facts"] < 0
    assert s["no_hallucinated_facts_slots"] < 0


def test_slot_valid_but_prose_adds_extra_halluc(input_json, gt):
    """Critical case — slots are valid but prose mentions an additional invalid fact.
    Slot-only rubric is fooled (+1); rendered-prose rubric catches it (-3).
    """
    c = _completion(
        explanation=(
            "Tariff {tariff_cited} but also tariff Fake-Plan went up by "
            "{rate_change_pct_cited}%."
        ),
        tariff_cited="Simply Fixed",
        rate_change_pct_cited=12.3,
    )
    s = score_completion(c, gt, input_json)
    assert s["no_hallucinated_facts"] < 0  # prose regex sees Fake-Plan
    assert s["no_hallucinated_facts_slots"] == 1.0  # slots are valid


def test_render_substitutes_slot_placeholders():
    parsed = parse_response(_completion(
        explanation="Rate {rate_change_pct_cited}% on {tariff_cited}.",
        tariff_cited="Simply Fixed",
        rate_change_pct_cited=-3.0,
    ))
    rendered = _render_explanation(parsed.explanations[0])
    assert "Simply Fixed" in rendered
    assert "-3.0%" in rendered
    assert "{tariff_cited}" not in rendered
    assert "{rate_change_pct_cited}" not in rendered


def test_render_old_format_unchanged():
    parsed = parse_response(_completion())
    rendered = _render_explanation(parsed.explanations[0])
    assert rendered == "Rates went up Tariff Simply Fixed went up by 12.3%."
