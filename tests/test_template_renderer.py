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
    overwrite_explanations,
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
