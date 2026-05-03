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


# Usage-aware rubric (PR-D, 2026-05-03) — _allowed_facts now also pulls
# projected_consumption_history.<fuel>.change_percent so prose like
# "your usage went up 10.15%" passes on Change in usage rows.


@pytest.fixture
def usage_input_json():
    """Row with consumption history (Change in usage scenario)."""
    return {
        "account_context": {
            "contract_history": [{
                "tariff_name": "Simply Fixed",
                "contract_rates_history": [
                    {"rates": [{"change_since_previous_rate_percent": 5.0}]}
                ],
            }],
            "projected_consumption_history": {
                "electricity": {"change_percent": 10.15, "change_kwh": 304.6},
                "gas": {"change_percent": 0.0, "change_kwh": 0.0},
            },
        },
        "latest_dd_change": {"dd_amount": 100.0, "dd_amount_change": 10.0},
    }


def test_usage_pct_citation_passes_under_usage_aware_rubric(usage_input_json):
    """A 10.15% usage citation should now PASS (PR-D fix).

    Pre-fix: 10.15% was validated against rate-change %s (only 5.0 there) and
    failed by ±0.5 distance. Post-fix: the consumption-change % is in the
    allowed-list, so the citation matches.
    """
    c = _completion(
        trigger="Change in usage",
        explanation="Your electricity usage went up by 10.15% this year.",
    )
    s = score_completion(c, ["Change in usage"], usage_input_json)
    assert s["no_hallucinated_facts"] == 1.0


def test_usage_aware_rubric_still_rejects_truly_invalid_pct(usage_input_json):
    """Numbers that match neither rate (5.0) nor consumption (10.15) still fail."""
    c = _completion(
        trigger="Change in usage",
        explanation="Your usage went up by 47.3% this year.",
    )
    s = score_completion(c, ["Change in usage"], usage_input_json)
    assert s["no_hallucinated_facts"] < 0


def test_usage_aware_rubric_still_accepts_rate_pct(usage_input_json):
    """Citing the rate %% still passes — usage-aware path is additive, not replacing."""
    c = _completion(
        trigger="Change in unit rates",
        explanation="Your tariff Simply Fixed went up by 5.0%.",
    )
    s = score_completion(c, ["Change in unit rates"], usage_input_json)
    assert s["no_hallucinated_facts"] == 1.0


def test_allowed_facts_helper_combines_rate_and_usage_pcts(usage_input_json):
    """Direct unit test on the helper — both axes present in the returned list."""
    from dd_explainer_rewards import _allowed_facts

    tariffs, pcts = _allowed_facts(usage_input_json)
    assert "simply fixed" in tariffs
    assert 5.0 in pcts                # rate-change %
    assert 10.15 in pcts               # electricity change_percent
    assert 0.0 in pcts                 # gas change_percent (kept; gated by abs() >= 1.0 at use-site)


def test_allowed_facts_no_consumption_history_still_works(input_json):
    """Rows without projected_consumption_history just return the rate %s."""
    from dd_explainer_rewards import _allowed_facts

    tariffs, pcts = _allowed_facts(input_json)
    # rate %s only (12.3); abs() variant adds 12.3 again as a duplicate, both pass-through
    assert 12.3 in pcts


@pytest.fixture
def usage_decrease_input_json():
    """Row where consumption *decreased* (negative change_percent)."""
    return {
        "account_context": {
            "contract_history": [{
                "tariff_name": "Simply Fixed",
                "contract_rates_history": [
                    {"rates": [{"change_since_previous_rate_percent": 0.0}]}
                ],
            }],
            "projected_consumption_history": {
                "electricity": {"change_percent": -12.38, "change_kwh": -304.6},
                "gas": {"change_percent": 0.0, "change_kwh": 0.0},
            },
        },
        "latest_dd_change": {"dd_amount": 100.0, "dd_amount_change": -10.0},
    }


def test_signflip_decrease_cited_as_positive_magnitude_passes(usage_decrease_input_json):
    """A decrease of -12.38% described as 'decreased by 12.38%' should pass.

    Pre-fix: input -12.38 vs cited +12.38 → |distance| = 24.76 → fail by 24.26.
    Post-fix: _allowed_facts also includes abs(-12.38) = 12.38 → pass.
    """
    c = _completion(
        trigger="Change in usage",
        explanation="Your gas usage has decreased by 12.38% this year.",
    )
    s = score_completion(c, ["Change in usage"], usage_decrease_input_json)
    assert s["no_hallucinated_facts"] == 1.0


def test_signflip_signed_negative_citation_still_passes(usage_decrease_input_json):
    """Citing the value with the original negative sign should also pass."""
    c = _completion(
        trigger="Change in usage",
        explanation="Your gas usage change_percent is -12.38%.",
    )
    s = score_completion(c, ["Change in usage"], usage_decrease_input_json)
    assert s["no_hallucinated_facts"] == 1.0


def test_signflip_truly_invalid_pct_still_rejected(usage_decrease_input_json):
    """Numbers that match neither the signed nor abs() form fail."""
    c = _completion(
        trigger="Change in usage",
        explanation="Your usage decreased by 47.3%.",
    )
    s = score_completion(c, ["Change in usage"], usage_decrease_input_json)
    assert s["no_hallucinated_facts"] < 0


def test_allowed_facts_includes_abs_for_signed_pcts(usage_decrease_input_json):
    """The helper exposes both signed and abs forms in its allowed list."""
    from dd_explainer_rewards import _allowed_facts

    _, pcts = _allowed_facts(usage_decrease_input_json)
    assert -12.38 in pcts        # original signed value preserved
    assert 12.38 in pcts          # abs() variant added so prose magnitude validates
