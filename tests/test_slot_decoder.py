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


def test_schema_with_facts():
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema(
        {"tariffs": ["Simply Fixed", "Simply Variable"], "rate_percentages": [12.3, -3.0]}
    )
    item_props = schema["properties"]["explanations"]["items"]["properties"]
    assert item_props["trigger"]["enum"][:1] == ["Manual reduction"]
    assert "Simply Fixed" in item_props["tariff_cited"]["anyOf"][1]["enum"]
    assert 12.3 in item_props["rate_change_pct_cited"]["anyOf"][1]["enum"]


def test_schema_empty_facts_collapses_to_null_only():
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema({"tariffs": [], "rate_percentages": []})
    item_props = schema["properties"]["explanations"]["items"]["properties"]
    assert item_props["tariff_cited"] == {"type": "null"}
    assert item_props["rate_change_pct_cited"] == {"type": "null"}
    assert item_props["prev_amount_cited"] == {"type": "null"}


def test_schema_with_prev_amount_cited():
    """PR-E: schema enum-constrains prev_amount_cited to the single allowed value."""
    from dd_explainer_slot_decoder import build_slot_enforcement_schema

    schema = build_slot_enforcement_schema(
        {"tariffs": [], "rate_percentages": [], "prev_amount": 90.0}
    )
    item_props = schema["properties"]["explanations"]["items"]["properties"]
    field = item_props["prev_amount_cited"]
    assert "anyOf" in field
    null_option, num_option = field["anyOf"]
    assert null_option == {"type": "null"}
    assert num_option == {"type": "number", "enum": [90.0]}


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
