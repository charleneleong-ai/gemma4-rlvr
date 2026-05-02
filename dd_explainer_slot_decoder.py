"""Slot-enforced JSON decoding for the dd_explainer task — PR-B foundation.

Plugs `lm-format-enforcer` into Gemma's `model.generate()` via a per-row
LogitsProcessor that constrains:

  - `tariff_cited` ∈ valid_facts["tariffs"]
  - `rate_change_pct_cited` ∈ valid_facts["rate_percentages"]
  - `trigger` ∈ TRIGGER_LABELS

while leaving `header` and `explanation` as free strings. This is the
"logit masking" referenced in v1_constrained.md's next-move list — the
real one, applied to structured slot fields rather than free prose
(which is the textbook reason structured-output frameworks ship with
JSON-schema-only enforcement).

Per-row schema variation: each row's `extract_valid_facts(input_json)`
returns a different allowed-list, so we build N parsers (one per batch
row) and route by `batch_id` in the prefix-allowed-tokens callback.
This is why `lm-format-enforcer` (lazy, char-level FSA) wins over
Outlines (eager, precompiled DFA) for this task shape — recompiling a
DFA per row at n=1000 would dwarf generation cost.

Usage:

    schemas = [build_slot_enforcement_schema(valid_facts) for valid_facts in per_row]
    fn = build_slot_prefix_fn(tokenizer, schemas)
    out = model.generate(..., prefix_allowed_tokens_fn=fn)

This module is a no-op until LMFE is installed. The compat shim at the
top patches `transformers.tokenization_utils.PreTrainedTokenizerBase`
back into existence — LMFE 0.11.3's transformers integration imports
from the pre-5.x location.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

# Compat shim — LMFE 0.11.3 imports `PreTrainedTokenizerBase` from
# `transformers.tokenization_utils` which transformers 5.x moved to
# `tokenization_utils_base`. Alias before importing LMFE's integration.
import transformers.tokenization_utils as _tu  # noqa: E402
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB  # noqa: E402

if not hasattr(_tu, "PreTrainedTokenizerBase"):
    _tu.PreTrainedTokenizerBase = _PTB

import torch  # noqa: E402
from lmformatenforcer import JsonSchemaParser, TokenEnforcerTokenizerData  # noqa: E402
from lmformatenforcer.integrations.transformers import (  # noqa: E402
    build_token_enforcer_tokenizer_data,
)
from lmformatenforcer.tokenenforcer import TokenEnforcer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from train_trigger_classifier import TRIGGER_LABELS  # noqa: E402


def build_slot_enforcement_schema(valid_facts: dict[str, list]) -> dict[str, Any]:
    """Build the JSON schema for slot-enforced decoding on one row.

    Schema constrains:
      - `trigger` ∈ TRIGGER_LABELS (always — eliminates schema-violation rows)
      - `tariff_cited` is null OR ∈ valid_facts["tariffs"]
      - `rate_change_pct_cited` is null OR ∈ valid_facts["rate_percentages"]

    `header` and `explanation` are free strings (no length cap — let
    well_formed score it). The schema returns a `DirectDebitExplainerResponse`
    shape compatible with the existing `parse_response` Pydantic validator.

    Empty allowed-lists collapse to a single-value enum {""} for tariff and
    {0.0} for percentage (the model can still pick null) — JSON schema
    requires at least one enum member, so we fall through to optional null
    when the row genuinely has no facts.
    """
    triggers = list(TRIGGER_LABELS) + ["No triggers identified"]
    tariff_options: list = ["null"] + list(valid_facts.get("tariffs") or [])
    pct_options: list = list(valid_facts.get("rate_percentages") or [])

    tariff_field: dict[str, Any]
    if len(tariff_options) > 1:
        tariff_field = {"anyOf": [{"type": "null"}, {"type": "string", "enum": tariff_options[1:]}]}
    else:
        tariff_field = {"type": "null"}

    pct_field: dict[str, Any]
    if pct_options:
        pct_field = {"anyOf": [{"type": "null"}, {"type": "number", "enum": pct_options}]}
    else:
        pct_field = {"type": "null"}

    return {
        "type": "object",
        "properties": {
            "explanations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "trigger": {"type": "string", "enum": triggers},
                        "header": {"type": "string"},
                        "tariff_cited": tariff_field,
                        "rate_change_pct_cited": pct_field,
                        "explanation": {"type": "string"},
                    },
                    "required": ["trigger", "header", "explanation"],
                },
            },
        },
        "required": ["explanations"],
    }


def _unwrap_tokenizer(tokenizer):
    """Gemma 4 + Unsloth returns a `Gemma4Processor` (multimodal wrapper) where
    the text tokenizer is at `.tokenizer`. LMFE's `build_token_enforcer_tokenizer_data`
    calls `len(tokenizer)` which fails on the processor. Extract the inner
    tokenizer if present; otherwise pass through.
    """
    return getattr(tokenizer, "tokenizer", tokenizer)


def build_tokenizer_data(tokenizer) -> TokenEnforcerTokenizerData:
    """Build the LMFE tokenizer data table — expensive (~10-30s on 262k-vocab
    Gemma 4) but vocab-independent of per-row schemas. Cache this OUTSIDE
    the per-batch loop and pass into `build_slot_prefix_fn`.

    Calling this once per eval saves ~16 × 20s = 5min of setup time
    on a 1000-row n=1000 sweep with batch=64.
    """
    return build_token_enforcer_tokenizer_data(_unwrap_tokenizer(tokenizer))


def build_slot_prefix_fn(
    tokenizer_data: TokenEnforcerTokenizerData,
    per_row_schemas: list[dict[str, Any]],
) -> Callable[[int, torch.Tensor], list[int]]:
    """Build a `prefix_allowed_tokens_fn` callback for batched generation
    with a different schema per row.

    `model.generate(..., prefix_allowed_tokens_fn=fn)` calls `fn(batch_id, input_ids)`
    at every decode step and expects a list of allowed token IDs. We route
    by batch_id to a row-specific TokenEnforcer.

    Implementation notes:
    - `tokenizer_data` is computed once per eval via `build_tokenizer_data`
      and reused across all batches — doing this per-batch is what made
      the first-pass smoke slow.
    - Each row gets its own TokenEnforcer wrapping a row-specific
      JsonSchemaParser. State is per-row, advancing as the row generates.
    - The callback decodes input_ids[batch_id] -> list[int] and looks up
      allowed-next-tokens. LMFE handles the FSA bookkeeping internally.
    """
    enforcers: list[TokenEnforcer] = [
        TokenEnforcer(tokenizer_data, JsonSchemaParser(schema))
        for schema in per_row_schemas
    ]

    def prefix_fn(batch_id: int, sent: torch.Tensor) -> list[int]:
        # LMFE returns a `TokenList` wrapper; HF's `prefix_allowed_tokens_fn`
        # contract expects `list[int]`, so we unwrap. Mirrors LMFE's own
        # `TransformersPrefixAllowedTokensFn.__call__`.
        return enforcers[batch_id].get_allowed_tokens(sent.tolist()).allowed_tokens

    return prefix_fn


__all__ = [
    "build_slot_enforcement_schema",
    "build_slot_prefix_fn",
    "build_tokenizer_data",
]
