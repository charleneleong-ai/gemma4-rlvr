"""Smoke tests for slot-enforced decoding (PR-B).

Verifies:
- `build_slot_enforcement_schema` produces a valid JSON schema dict for
  populated and empty allowed-lists.
- `build_slot_prefix_fn` builds a callable that returns sensible allowed
  tokens at known waypoints in a valid JSON output.
- Per-row schema variation (different rows in a batch see different
  enum constraints) is honored via batch_id routing.

These tests don't load Gemma — only the tokenizer and LMFE machinery.
Real end-to-end validation is in scripts/two_stage_eval.py --enforce-slots.
"""

from __future__ import annotations

import torch
import pytest


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("unsloth/gemma-4-E4B-it")


def test_schema_with_facts_force_populates_tariff_and_rate():
    """PR-F: tariff_cited / rate_change_pct_cited are required + value-enum
    (no null branch) when the row has any facts of that type. Same recipe
    PR-E used for prev_amount_cited.
    """
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema(
        {"tariffs": ["Simply Fixed", "Simply Variable"], "rate_percentages": [12.3, -3.0]}
    )
    items = schema["properties"]["explanations"]["items"]
    item_props = items["properties"]
    assert item_props["trigger"]["enum"][:1] == ["Manual reduction"]
    # No anyOf / null branch — value-enum only
    assert item_props["tariff_cited"] == {
        "type": "string", "enum": ["Simply Fixed", "Simply Variable"],
    }
    assert item_props["rate_change_pct_cited"] == {
        "type": "number", "enum": [12.3, -3.0],
    }
    # Both fields force-populated via `required`
    assert "tariff_cited" in items["required"]
    assert "rate_change_pct_cited" in items["required"]


def test_schema_empty_facts_collapses_to_null_only():
    """When a row has no facts of a given type, the slot is null-only and
    NOT in required (model is allowed to skip what doesn't exist)."""
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema({"tariffs": [], "rate_percentages": []})
    items = schema["properties"]["explanations"]["items"]
    item_props = items["properties"]
    assert item_props["tariff_cited"] == {"type": "null"}
    assert item_props["rate_change_pct_cited"] == {"type": "null"}
    assert item_props["prev_amount_cited"] == {"type": "null"}
    # Empty slots are NOT required
    assert "tariff_cited" not in items["required"]
    assert "rate_change_pct_cited" not in items["required"]


def test_schema_partial_facts_only_required_when_present():
    """Mixed row: 1 rate but no tariffs. rate_change_pct_cited is required
    + value-enum; tariff_cited is null-only and NOT required."""
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema(
        {"tariffs": [], "rate_percentages": [5.5], "prev_amount": 90.0}
    )
    items = schema["properties"]["explanations"]["items"]
    item_props = items["properties"]
    assert item_props["tariff_cited"] == {"type": "null"}
    assert item_props["rate_change_pct_cited"] == {"type": "number", "enum": [5.5]}
    assert "tariff_cited" not in items["required"]
    assert "rate_change_pct_cited" in items["required"]
    assert "prev_amount_cited" in items["required"]


def test_schema_with_prev_amount_cited():
    """PR-E Option B v2: schema force-pins prev_amount_cited to the single allowed
    value (no null option when a value is available). The field is also added to
    the per-explanation `required` list so LMFE forces the model to emit it.
    """
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema(
        {"tariffs": [], "rate_percentages": [], "prev_amount": 90.0}
    )
    items = schema["properties"]["explanations"]["items"]
    field = items["properties"]["prev_amount_cited"]
    # Single-value number enum, no null (forces population)
    assert field == {"type": "number", "enum": [90.0]}
    # And the field is required, so LMFE forces emission
    assert "prev_amount_cited" in items["required"]


def test_schema_prev_amount_none_falls_through_to_null():
    """When the row has no prev_amount, the slot is null-only (no enum)."""
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema(
        {"tariffs": [], "rate_percentages": [], "prev_amount": None}
    )
    item_props = schema["properties"]["explanations"]["items"]["properties"]
    assert item_props["prev_amount_cited"] == {"type": "null"}


def test_extract_valid_facts_returns_prev_amount():
    """PR-E: extract_valid_facts pulls dd_amount - dd_amount_change."""
    from dd_explainer_two_stage import extract_valid_facts

    inp = {
        "account_context": {"contract_history": []},
        "latest_dd_change": {"dd_amount": 100.0, "dd_amount_change": 10.0},
    }
    facts = extract_valid_facts(inp)
    assert facts["prev_amount"] == 90.0  # 100 - 10


def test_extract_valid_facts_prev_amount_handles_missing_change():
    """When dd_amount_change is None, prev_amount = dd_amount (treats as 0 delta)."""
    from dd_explainer_two_stage import extract_valid_facts

    inp = {
        "account_context": {"contract_history": []},
        "latest_dd_change": {"dd_amount": 75.5, "dd_amount_change": None},
    }
    facts = extract_valid_facts(inp)
    assert facts["prev_amount"] == 75.5


# PR-H: trigger grounding extraction. Each predicate is 100%-recall / 0%-FPR
# vs the corresponding ground-truth trigger in the n=1000 PR-G eval.

def test_extract_trigger_grounding_first_dd_review():
    from dd_explainer_two_stage import extract_trigger_grounding

    only_one = {
        "account_context": {"dd_change_history": [{"is_currently_active_DD": True}]},
        "latest_dd_change": {},
    }
    g = extract_trigger_grounding(only_one)
    assert g["first_dd_review"] == {"is_first": True, "n_prior_dd_entries": 0}

    has_prior = {
        "account_context": {"dd_change_history": [{}, {}]},
        "latest_dd_change": {},
    }
    assert "first_dd_review" not in extract_trigger_grounding(has_prior)


def test_extract_trigger_grounding_missed_payments():
    from dd_explainer_two_stage import extract_trigger_grounding

    inp = {
        "account_context": {
            "dd_change_history": [],
            "payment_history": [
                {"is_payment_successful": True, "payment_period": "Aug 2025",
                 "transaction_amount_in_pounds": 80.0, "transaction_timestamp": "2025-08-01"},
                {"is_payment_successful": False, "payment_period": "Sep 2025",
                 "transaction_amount_in_pounds": 80.0, "transaction_timestamp": "2025-09-01"},
            ],
        },
        "latest_dd_change": {},
    }
    g = extract_trigger_grounding(inp)
    assert g["missed_payments"]["n_missed"] == 1
    assert g["missed_payments"]["most_recent_period"] == "Sep 2025"
    assert g["missed_payments"]["most_recent_amount_gbp"] == 80.0

    clean = {"account_context": {"payment_history": [{"is_payment_successful": True}]}}
    assert "missed_payments" not in extract_trigger_grounding(clean)


def test_extract_trigger_grounding_manual_reduction():
    from dd_explainer_two_stage import extract_trigger_grounding

    # Manual reduction in current period
    inp_now = {
        "account_context": {"dd_change_history": [{}, {}]},
        "latest_dd_change": {
            "is_amount_manually_reduced_lower_than_recommended_amount": True,
            "dd_amount": 60.0, "recommended_dd_amount": 90.0,
            "datetime_from": "2026-01-01",
        },
    }
    g = extract_trigger_grounding(inp_now)
    assert g["manual_reduction"]["active_in"] == "current_period"
    assert g["manual_reduction"]["manual_dd_amount_gbp"] == 60.0

    # Previous-period only
    inp_prev = {
        "account_context": {"dd_change_history": [
            {"is_amount_manually_reduced_lower_than_recommended_amount": True,
             "dd_amount": 50.0, "recommended_dd_amount": 80.0, "datetime_from": "2025-01-01"},
            {},
        ]},
        "latest_dd_change": {"is_amount_manually_reduced_lower_than_recommended_amount": False},
    }
    g = extract_trigger_grounding(inp_prev)
    assert g["manual_reduction"]["active_in"] == "previous_period"
    assert g["manual_reduction"]["manual_dd_amount_gbp"] == 50.0

    none = {"account_context": {"dd_change_history": []}, "latest_dd_change": {}}
    assert "manual_reduction" not in extract_trigger_grounding(none)


def test_extract_trigger_grounding_exemption_expiry():
    from dd_explainer_two_stage import extract_trigger_grounding

    inp = {
        "account_context": {"dd_change_history": [
            {"is_exemption": True, "exemption_expiry_date": "2025-10-28",
             "dd_amount": 70.0, "recommended_dd_amount": 100.0},
            {"is_exemption": False},
        ]},
        "latest_dd_change": {"is_exemption": False},
    }
    g = extract_trigger_grounding(inp)
    assert g["exemption_expiry"]["expired_on"] == "2025-10-28"
    assert g["exemption_expiry"]["previous_dd_amount_gbp"] == 70.0

    # Still on exemption — not expired
    still_active = {
        "account_context": {"dd_change_history": [
            {"is_exemption": True, "exemption_expiry_date": "2025-10-28"},
            {"is_exemption": True},
        ]},
        "latest_dd_change": {"is_exemption": True},
    }
    assert "exemption_expiry" not in extract_trigger_grounding(still_active)


def test_build_two_stage_prompt_renders_grounding_block_only_when_present():
    """Grounding block appears in the prompt suffix only when extract_trigger_grounding
    returned at least one trigger context."""
    from dd_explainer_two_stage import build_two_stage_prompt

    base = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    msg = build_two_stage_prompt(
        base, ["First DD review since account start"],
        trigger_grounding={"first_dd_review": {"is_first": True, "n_prior_dd_entries": 0}},
    )
    text = msg[0]["content"][0]["text"]
    assert "TRIGGER GROUNDING CONTEXT" in text
    assert "First DD review since account start" in text

    msg_empty = build_two_stage_prompt(base, ["x"], trigger_grounding={})
    assert "TRIGGER GROUNDING CONTEXT" not in msg_empty[0]["content"][0]["text"]

    msg_none = build_two_stage_prompt(base, ["x"], trigger_grounding=None)
    assert "TRIGGER GROUNDING CONTEXT" not in msg_none[0]["content"][0]["text"]


def test_prefix_fn_routes_per_row(tokenizer):
    from dd_explainer_slot_decoder import (
        build_slot_enforcement_schema,
        build_slot_prefix_fn,
        build_tokenizer_data,
    )

    schemas = [
        build_slot_enforcement_schema({"tariffs": ["Simply Fixed"], "rate_percentages": [12.3]}),
        build_slot_enforcement_schema({"tariffs": ["Other Plan"], "rate_percentages": [-3.0]}),
    ]
    fn = build_slot_prefix_fn(build_tokenizer_data(tokenizer), schemas)

    empty = torch.tensor([], dtype=torch.long)
    allowed_0 = fn(0, empty)
    allowed_1 = fn(1, empty)
    # Both rows are at step 0 — the JSON object hasn't started yet, so the
    # allowed-set should be identical (whitespace + opening brace).
    assert set(allowed_0) == set(allowed_1)
    assert len(allowed_0) > 0


def test_prefix_fn_after_open_brace(tokenizer):
    """Simulates HF's `prefix_allowed_tokens_fn` call pattern: incremental
    growth one-token-at-a-time. LMFE's `TokenEnforcer` builds parser state
    by walking the prefix step by step — calling cold with a non-empty
    prefix doesn't work, so production usage always goes through the
    incremental path."""
    from dd_explainer_slot_decoder import (
        build_slot_enforcement_schema,
        build_slot_prefix_fn,
        build_tokenizer_data,
    )

    schemas = [build_slot_enforcement_schema(
        {"tariffs": ["Simply Fixed"], "rate_percentages": [12.3]}
    )]
    fn = build_slot_prefix_fn(build_tokenizer_data(tokenizer), schemas)

    # Walk the parser through `{"` token by token.
    target_ids = tokenizer.encode('{"', add_special_tokens=False)
    so_far: list[int] = []
    for tid in [None] + target_ids:  # None = the empty-prefix call HF makes first
        if tid is not None:
            so_far.append(tid)
        fn(0, torch.tensor(so_far, dtype=torch.long))

    allowed = fn(0, torch.tensor(so_far, dtype=torch.long))
    decoded = {t: tokenizer.decode([t]) for t in allowed}
    has_e_prefix = any(d.lstrip().lower().startswith("e") for d in decoded.values())
    assert has_e_prefix, (
        f"no allowed token starts with 'e' for 'explanations' field name; "
        f"sample={list(decoded.values())[:10]}"
    )
    # A token spelling a different field name should NOT be allowed —
    # `triggers` is not a property in our schema.
    triggers_tokens = tokenizer.encode("triggers", add_special_tokens=False)
    assert triggers_tokens[0] not in allowed, (
        f"token for 'triggers' (id={triggers_tokens[0]}) leaked into allowed set"
    )
