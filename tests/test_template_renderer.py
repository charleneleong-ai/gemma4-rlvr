"""Tests for dd_explainer_template_renderer (PR-I).

Verifies templates render only from grounding/valid_facts values, never invent
phrases that could fail the row-level no_halluc rubric, and degrade gracefully
when fields are missing.
"""

from __future__ import annotations

import pytest

from dd_explainer_template_renderer import (
    LONELY_TRIGGERS,
    NO_TRIGGERS_LABEL,
    backfill_missing_triggers,
    overwrite_explanations,
    render_for_backfill,
    render_lonely_explanation,
)


def test_lonely_triggers_set():
    assert LONELY_TRIGGERS == frozenset({
        "First DD review since account start",
        "Missed/bounced DD payments",
        "Manual reduction",
        "Exemption Expiry",
    })


def test_non_lonely_trigger_returns_none():
    out = render_lonely_explanation(
        "Change in usage",
        {"first_dd_review": {"is_first": True}},
        {"prev_amount": 80.0},
    )
    assert out is None


def test_first_dd_review_with_prev_amount():
    out = render_lonely_explanation(
        "First DD review since account start",
        {"first_dd_review": {"is_first": True, "n_prior_dd_entries": 0}},
        {"prev_amount": 82.75},
    )
    assert out is not None
    assert "first review" in out["explanation"].lower()
    assert "£82.75" in out["explanation"]
    assert "necessary catch-up" not in out["explanation"]
    assert "standard process" not in out["explanation"]


def test_first_dd_review_without_prev_amount_no_invented_clause():
    out = render_lonely_explanation(
        "First DD review since account start",
        {"first_dd_review": {"is_first": True}},
        {"prev_amount": None},
    )
    assert out is not None
    # Must not invent a £ amount when prev_amount is missing
    assert "£" not in out["explanation"]


def test_first_dd_review_returns_none_when_not_first():
    out = render_lonely_explanation(
        "First DD review since account start",
        {},  # no first_dd_review key
        {},
    )
    assert out is None


def test_missed_payments_renders_n_period_amount():
    out = render_lonely_explanation(
        "Missed/bounced DD payments",
        {"missed_payments": {
            "n_missed": 2,
            "most_recent_period": "Jan 2026",
            "most_recent_amount_gbp": 63.13,
        }},
        {},
    )
    assert out is not None
    assert "2 unsuccessful" in out["explanation"]
    assert "£63.13" in out["explanation"]
    assert "Jan 2026" in out["explanation"]
    # singular vs plural correctness
    assert "payments" in out["explanation"]


def test_missed_payments_singular():
    out = render_lonely_explanation(
        "Missed/bounced DD payments",
        {"missed_payments": {
            "n_missed": 1,
            "most_recent_period": "Dec 2025",
            "most_recent_amount_gbp": 80.0,
        }},
        {},
    )
    assert "1 unsuccessful Direct Debit payment " in out["explanation"]
    # singular form
    assert " payment " in out["explanation"]


def test_missed_payments_returns_none_when_clean():
    assert render_lonely_explanation("Missed/bounced DD payments", {}, {}) is None


def test_manual_reduction_previous_period():
    out = render_lonely_explanation(
        "Manual reduction",
        {"manual_reduction": {
            "active_in": "previous_period",
            "manual_dd_amount_gbp": 50.0,
            "recommended_dd_amount_gbp": 80.0,
        }},
        {},
    )
    assert out is not None
    assert "£50.00" in out["explanation"]
    assert "£80.00" in out["explanation"]
    assert "previous" in out["explanation"].lower()
    # Must not say "automatic" or other unsupported words
    assert "automatic" not in out["explanation"].lower()


def test_manual_reduction_current_period():
    out = render_lonely_explanation(
        "Manual reduction",
        {"manual_reduction": {
            "active_in": "current_period",
            "manual_dd_amount_gbp": 60.0,
            "recommended_dd_amount_gbp": 90.0,
        }},
        {},
    )
    assert out is not None
    assert "£60.00" in out["explanation"]
    assert "£90.00" in out["explanation"]


def test_manual_reduction_missing_recommended_no_invented():
    out = render_lonely_explanation(
        "Manual reduction",
        {"manual_reduction": {
            "active_in": "current_period",
            "manual_dd_amount_gbp": 60.0,
            "recommended_dd_amount_gbp": None,
        }},
        {},
    )
    assert out is not None
    assert "£60.00" in out["explanation"]
    # Don't invent a second £ amount
    assert out["explanation"].count("£") == 1


def test_exemption_expiry_strips_time():
    out = render_lonely_explanation(
        "Exemption Expiry",
        {"exemption_expiry": {
            "expired_on": "2025-10-28 00:00:00",
            "previous_dd_amount_gbp": 70.0,
            "previous_recommended_dd_amount_gbp": 100.0,
        }},
        {},
    )
    assert out is not None
    assert "2025-10-28" in out["explanation"]
    assert "00:00:00" not in out["explanation"]
    assert "£70.00" in out["explanation"]
    assert "£100.00" in out["explanation"]


def test_exemption_expiry_handles_missing_fields():
    out = render_lonely_explanation(
        "Exemption Expiry",
        {"exemption_expiry": {}},
        {},
    )
    assert out is not None
    assert "exemption" in out["explanation"].lower()
    # No invented £ amounts
    assert "£" not in out["explanation"]


def test_no_triggers_identified_renders_with_tariff():
    out = render_lonely_explanation(
        NO_TRIGGERS_LABEL,
        grounding={},  # Stage-1 said nothing applies, so no grounding
        valid_facts={"tariffs": ["Simpler Energy"], "prev_amount": 90.0},
    )
    assert out is not None
    # Must mention the tariff via "tariff <Name>" so _TARIFF_RE picks it up
    assert "tariff Simpler Energy" in out["explanation"]
    # Must mention the prev amount
    assert "£90.00" in out["explanation"]


def test_no_triggers_identified_returns_none_without_tariff():
    """When there's no tariff to cite, the template can't escape the inaction
    loophole — better to fall through to LLM than emit prose with no anchors."""
    out = render_lonely_explanation(
        NO_TRIGGERS_LABEL,
        grounding={},
        valid_facts={"tariffs": [], "prev_amount": None},
    )
    assert out is None


def test_overwrite_explanations_replaces_lonely_keeps_slots():
    parsed = {"explanations": [
        {
            "trigger": "First DD review since account start",
            "header": "<llm header>",
            "explanation": "<llm fabricated prose with template phrases>",
            "prev_amount_cited": 82.75,
            "tariff_cited": "Simpler Energy",
            "rate_change_pct_cited": 0.0,
        },
        {
            "trigger": "Change in usage",
            "header": "<llm header for usage>",
            "explanation": "<llm prose for usage — should be untouched>",
            "prev_amount_cited": None,
            "tariff_cited": None,
            "rate_change_pct_cited": None,
        },
    ]}
    grounding = {"first_dd_review": {"is_first": True}}
    valid_facts = {"prev_amount": 82.75}
    out = overwrite_explanations(parsed, grounding, valid_facts)
    # Lonely trigger overwritten
    assert out["explanations"][0]["header"] == "First Direct Debit review since account start"
    assert "first review" in out["explanations"][0]["explanation"].lower()
    # Slot fields PRESERVED
    assert out["explanations"][0]["prev_amount_cited"] == 82.75
    assert out["explanations"][0]["tariff_cited"] == "Simpler Energy"
    assert out["explanations"][0]["rate_change_pct_cited"] == 0.0
    # Non-lonely trigger untouched
    assert out["explanations"][1]["header"] == "<llm header for usage>"
    assert out["explanations"][1]["explanation"] == "<llm prose for usage — should be untouched>"


def test_overwrite_explanations_handles_malformed_gracefully():
    # No explanations key
    assert overwrite_explanations({}, {}, {}) == {}
    # Non-list explanations
    bad = {"explanations": "oops"}
    assert overwrite_explanations(bad, {}, {})["explanations"] == "oops"
    # Non-dict entries skipped
    out = overwrite_explanations({"explanations": [None, "x", {"trigger": "Change in usage"}]}, {}, {})
    assert out["explanations"][:2] == [None, "x"]


# --------------------------------------------------------------------------
# backfill_missing_triggers
# --------------------------------------------------------------------------


def _grounding_with_first_dd_review() -> dict:
    return {"first_dd_review": {"is_first": True, "n_prior_dd_entries": 0}}


def _valid_facts_simpler() -> dict:
    return {
        "tariffs": ["Simpler Energy"],
        "rate_percentages": [],
        "prev_amount": 82.75,
    }


def test_backfill_no_op_when_no_trigger_missing():
    parsed = {"explanations": [
        {"trigger": "Change in usage", "header": "h", "explanation": "e",
         "tariff_cited": "Simpler Energy", "rate_change_pct_cited": 0.0,
         "prev_amount_cited": 82.75},
    ]}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["Change in usage"],
        stage1_probs={"Change in usage": 0.99},
        grounding={},
        valid_facts=_valid_facts_simpler(),
    )
    assert len(out["explanations"]) == 1


def test_backfill_appends_missing_trigger_with_complete_slots():
    parsed = {"explanations": [
        {"trigger": "Change in usage", "header": "h", "explanation": "e",
         "tariff_cited": "Simpler Energy", "rate_change_pct_cited": 0.0,
         "prev_amount_cited": 82.75},
    ]}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["Change in usage", "First DD review since account start"],
        stage1_probs={"Change in usage": 0.99,
                      "First DD review since account start": 0.95},
        grounding=_grounding_with_first_dd_review(),
        valid_facts=_valid_facts_simpler(),
    )
    assert len(out["explanations"]) == 2
    appended = out["explanations"][1]
    assert appended["trigger"] == "First DD review since account start"
    assert appended["header"] == "First Direct Debit review since account start"
    assert "first review" in appended["explanation"].lower()
    # Slot fields populated from valid_facts so the rubric scores them.
    assert appended["tariff_cited"] == "Simpler Energy"
    assert appended["prev_amount_cited"] == 82.75
    assert appended["rate_change_pct_cited"] == 0.0
    # Existing entry untouched.
    assert out["explanations"][0]["trigger"] == "Change in usage"


def test_backfill_gate_blocks_low_confidence_predictions():
    parsed = {"explanations": []}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["First DD review since account start"],
        stage1_probs={"First DD review since account start": 0.5},
        grounding=_grounding_with_first_dd_review(),
        valid_facts=_valid_facts_simpler(),
        confidence_threshold=0.9,
    )
    # Below threshold → no backfill.
    assert out["explanations"] == []


def test_backfill_gate_threshold_boundary_inclusive():
    parsed = {"explanations": []}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["First DD review since account start"],
        stage1_probs={"First DD review since account start": 0.9},
        grounding=_grounding_with_first_dd_review(),
        valid_facts=_valid_facts_simpler(),
        confidence_threshold=0.9,
    )
    # prob == threshold → backfilled (>=).
    assert len(out["explanations"]) == 1


def test_backfill_skips_when_renderer_unavailable():
    # An unknown trigger has no renderer in either registry.
    parsed = {"explanations": []}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["Some Unknown Trigger That Does Not Exist"],
        stage1_probs={"Some Unknown Trigger That Does Not Exist": 0.99},
        grounding={},
        valid_facts=_valid_facts_simpler(),
    )
    assert out["explanations"] == []


def test_backfill_skips_when_grounding_missing():
    # Renderer for First DD review needs grounding.first_dd_review.is_first.
    parsed = {"explanations": []}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["First DD review since account start"],
        stage1_probs={"First DD review since account start": 0.99},
        grounding={},  # no first_dd_review key
        valid_facts=_valid_facts_simpler(),
    )
    assert out["explanations"] == []


def test_backfill_probs_none_bypasses_gate():
    parsed = {"explanations": []}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["First DD review since account start"],
        stage1_probs=None,
        grounding=_grounding_with_first_dd_review(),
        valid_facts=_valid_facts_simpler(),
    )
    assert len(out["explanations"]) == 1


def test_backfill_appends_multiple_missing_triggers():
    parsed = {"explanations": []}
    grounding = {
        "first_dd_review": {"is_first": True, "n_prior_dd_entries": 0},
        "missed_payments": {
            "n_missed": 2,
            "most_recent_period": "2025-09",
            "most_recent_amount_gbp": 60.0,
        },
    }
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=[
            "First DD review since account start",
            "Missed/bounced DD payments",
        ],
        stage1_probs={
            "First DD review since account start": 0.99,
            "Missed/bounced DD payments": 0.99,
        },
        grounding=grounding,
        valid_facts=_valid_facts_simpler(),
    )
    triggers = [e["trigger"] for e in out["explanations"]]
    assert triggers == [
        "First DD review since account start",
        "Missed/bounced DD payments",
    ]


def test_backfill_idempotent():
    parsed = {"explanations": []}
    args = dict(
        stage1_triggers=["First DD review since account start"],
        stage1_probs={"First DD review since account start": 0.99},
        grounding=_grounding_with_first_dd_review(),
        valid_facts=_valid_facts_simpler(),
    )
    backfill_missing_triggers(parsed, **args)
    backfill_missing_triggers(parsed, **args)
    # Second call should not duplicate.
    assert len(parsed["explanations"]) == 1


def test_backfill_handles_malformed_parsed():
    # No explanations key.
    assert backfill_missing_triggers({}, [], None, {}, {}) == {}
    # Non-list explanations.
    bad = {"explanations": "oops"}
    assert backfill_missing_triggers(bad, [], None, {}, {})["explanations"] == "oops"


def test_backfill_no_stage1_triggers_is_no_op():
    parsed = {"explanations": []}
    out = backfill_missing_triggers(parsed, [], {}, {}, _valid_facts_simpler())
    assert out["explanations"] == []


# --------------------------------------------------------------------------
# Change in unit rates / Change in usage renderers (backfill-only)
# --------------------------------------------------------------------------


def test_change_in_unit_rates_with_significant_rate_pct():
    out = render_for_backfill(
        "Change in unit rates",
        grounding={},
        valid_facts={
            "tariffs": ["2-Year Fixed"],
            "rate_percentages": [-3.73, 5.62, 0.4],
            "prev_amount": 132.97,
        },
    )
    assert out is not None
    assert out["header"] == "Tariff rate adjustments applied"
    expl = out["explanation"]
    # Tariff cited via "tariff <Name>" form so _TARIFF_RE accepts it.
    assert "tariff 2-Year Fixed" in expl
    # Largest-abs rate (>= 1.0) cited; 0.4 ignored (rubric only validates abs >= 1.0).
    assert "5.62%" in expl
    assert "increased" in expl
    assert "£132.97" in expl


def test_change_in_unit_rates_negative_rate_says_decreased():
    out = render_for_backfill(
        "Change in unit rates",
        grounding={},
        valid_facts={
            "tariffs": ["Better Energy Fixed"],
            "rate_percentages": [-7.5, 1.2],
            "prev_amount": 60.29,
        },
    )
    assert out is not None
    assert "decreased" in out["explanation"]
    assert "7.50%" in out["explanation"]


def test_change_in_unit_rates_no_significant_rates_omits_clause():
    out = render_for_backfill(
        "Change in unit rates",
        grounding={},
        valid_facts={
            "tariffs": ["Simpler Energy"],
            "rate_percentages": [0.4, -0.8],   # all below abs 1.0
            "prev_amount": 80.0,
        },
    )
    assert out is not None
    # No rate clause because none meets the rubric's abs >= 1.0 threshold.
    assert "%" not in out["explanation"]
    assert "tariff Simpler Energy" in out["explanation"]


def test_change_in_unit_rates_returns_none_without_tariff():
    out = render_for_backfill(
        "Change in unit rates",
        grounding={},
        valid_facts={"tariffs": [], "rate_percentages": [5.0], "prev_amount": 100.0},
    )
    assert out is None


def test_change_in_usage_renders_without_pct():
    out = render_for_backfill(
        "Change in usage",
        grounding={},
        valid_facts={
            "tariffs": ["Simpler Energy"],
            "rate_percentages": [],
            "prev_amount": 112.17,
        },
    )
    assert out is not None
    assert out["header"] == "Direct Debit updated for usage changes"
    expl = out["explanation"]
    # Tariff cited.
    assert "tariff Simpler Energy" in expl
    # No specific usage % invented (the renderer deliberately doesn't cite one).
    assert "%" not in expl
    # prev_amount cited.
    assert "£112.17" in expl


def test_change_in_usage_returns_none_without_tariff():
    out = render_for_backfill(
        "Change in usage",
        grounding={},
        valid_facts={"tariffs": [], "rate_percentages": [], "prev_amount": 80.0},
    )
    assert out is None


def test_render_lonely_does_not_overwrite_change_in_usage():
    """`render_lonely_explanation` must NOT overwrite Change in usage / unit rates
    so passing LLM prose for those triggers stays untouched."""
    out = render_lonely_explanation(
        "Change in usage",
        grounding={},
        valid_facts={
            "tariffs": ["Simpler Energy"],
            "rate_percentages": [],
            "prev_amount": 112.17,
        },
    )
    assert out is None
    out = render_lonely_explanation(
        "Change in unit rates",
        grounding={},
        valid_facts={
            "tariffs": ["2-Year Fixed"],
            "rate_percentages": [5.0],
            "prev_amount": 100.0,
        },
    )
    assert out is None


def test_overwrite_explanations_does_not_touch_change_in_usage():
    """Belt-and-braces: overwrite_explanations leaves Change in usage prose alone."""
    parsed = {"explanations": [{
        "trigger": "Change in usage",
        "header": "<llm header>",
        "explanation": "<llm prose with usage % 23.23%>",
        "tariff_cited": "Simpler Energy",
        "rate_change_pct_cited": 0.0,
        "prev_amount_cited": 112.17,
    }]}
    out = overwrite_explanations(parsed, {}, {
        "tariffs": ["Simpler Energy"], "rate_percentages": [], "prev_amount": 112.17,
    })
    assert out["explanations"][0]["header"] == "<llm header>"
    assert out["explanations"][0]["explanation"] == "<llm prose with usage % 23.23%>"


def test_backfill_now_handles_change_in_unit_rates():
    parsed = {"explanations": [
        {"trigger": "Manual reduction", "header": "h", "explanation": "e",
         "tariff_cited": "2-Year Fixed", "rate_change_pct_cited": 0.0,
         "prev_amount_cited": 132.97},
    ]}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["Manual reduction", "Change in unit rates"],
        stage1_probs={"Manual reduction": 0.99, "Change in unit rates": 0.99},
        grounding={},
        valid_facts={
            "tariffs": ["2-Year Fixed"],
            "rate_percentages": [5.62],
            "prev_amount": 132.97,
        },
    )
    triggers = [e["trigger"] for e in out["explanations"]]
    assert triggers == ["Manual reduction", "Change in unit rates"]
    appended = out["explanations"][1]
    assert "5.62%" in appended["explanation"]
    assert appended["tariff_cited"] == "2-Year Fixed"
    assert appended["prev_amount_cited"] == 132.97


def test_backfill_now_handles_change_in_usage():
    parsed = {"explanations": [
        {"trigger": "Missed/bounced DD payments", "header": "h", "explanation": "e",
         "tariff_cited": "Simpler Energy", "rate_change_pct_cited": 0.0,
         "prev_amount_cited": 112.17},
    ]}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["Missed/bounced DD payments", "Change in usage"],
        stage1_probs={"Missed/bounced DD payments": 0.99, "Change in usage": 0.99},
        grounding={},
        valid_facts={
            "tariffs": ["Simpler Energy"],
            "rate_percentages": [],
            "prev_amount": 112.17,
        },
    )
    triggers = [e["trigger"] for e in out["explanations"]]
    assert triggers == ["Missed/bounced DD payments", "Change in usage"]


def test_backfill_handles_missing_tariff_in_valid_facts():
    parsed = {"explanations": []}
    out = backfill_missing_triggers(
        parsed,
        stage1_triggers=["First DD review since account start"],
        stage1_probs={"First DD review since account start": 0.99},
        grounding=_grounding_with_first_dd_review(),
        valid_facts={"tariffs": [], "rate_percentages": [], "prev_amount": 50.0},
    )
    # Renderer returns prose without tariff_clause; backfill still appends with
    # tariff_cited="" so the entry is well-formed.
    assert len(out["explanations"]) == 1
    assert out["explanations"][0]["tariff_cited"] == ""
    assert out["explanations"][0]["prev_amount_cited"] == 50.0
