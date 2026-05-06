"""Two-stage pipeline — Stage 1 classifier feeds Stage 2 LLM explainer.

Stage 1: `TwoStageClassifier` predicts the trigger set from the raw input
JSON, mirroring `dd_explainer_gate.GateModel` but with a 6-d sigmoid head.
Returns a list of trigger strings, applying the default-rule (none of the
6 → ["No triggers identified"]).

Stage 2: `build_two_stage_prompt` modifies the chat prompt to include the
predicted triggers as an explicit response template — turning the LLM's
job from "pick triggers AND explain" into "explain the given triggers".

Usage in an A/B harness:

    from dd_explainer_two_stage import TwoStageClassifier, build_two_stage_prompt

    cls = TwoStageClassifier.load("data/trigger_classifier_v3_extra_features.pt")
    triggers = cls.predict_triggers(row["input_json"])      # list[str]
    msgs = build_two_stage_prompt(pin, triggers)            # list[chat msg]
    completion = generate(msgs, model, tokenizer)
    score = score_completion(completion, gt_triggers, input_json, ...)

The build_two_stage_prompt function preserves the system + base user prompt
and appends a templated "RESPONSE TEMPLATE" section that pre-fills the
trigger choices, leaving header + explanation to the LLM. This mirrors the
"infill" pattern used by structured-output systems.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Reuse the gate's helpers.
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from train_outlier_encoder import (  # noqa: E402
    _embed,
    _extract_numeric_features,
)
from train_trigger_classifier import (  # noqa: E402
    CLASSIFIER_EXTRA_FEATURE_NAMES,
    DEFAULT_FALLBACK_LABEL,
    TRIGGER_LABELS,
    _extract_classifier_extra_features,
    _serialize_input,
)


class TwoStageClassifier:
    """Inference wrapper around the saved trigger classifier head.

    Same architecture pattern as `GateModel` but with 6-d output. The
    `predict_triggers` method returns a list of trigger strings ready to
    drop into a templated prompt.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        encoder: AutoModel,
        head: nn.Linear,
        device: torch.device,
        numeric_mean: torch.Tensor,
        numeric_std: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.head = head
        self.device = device
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std
        self.threshold = threshold
        # Convenience: head_in_dim from the first Linear layer regardless of
        # whether `head` is bare Linear or wrapped in Sequential.
        first_linear = head if isinstance(head, nn.Linear) else next(
            m for m in head.modules() if isinstance(m, nn.Linear)
        )
        self.head_in_dim = first_linear.in_features

    @classmethod
    def load(cls, head_path: Path | str) -> TwoStageClassifier:
        """Load a classifier head saved by `scripts/train_trigger_classifier.py`.

        Supports both 'linear' (v0-v3) and '2-layer-mlp' (v4+) head types.
        Falls back to 'linear' when `head_type` is absent for backwards-compat.
        """
        ckpt = torch.load(Path(head_path), map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder"])
        encoder = AutoModel.from_pretrained(ckpt["encoder"]).to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        head_type = ckpt.get("head_type", "linear")
        if head_type == "linear":
            head: nn.Module = nn.Linear(ckpt["head_in_dim"], len(TRIGGER_LABELS)).to(device)
            head.weight.data = ckpt["weight"].to(device)
            head.bias.data = ckpt["bias"].to(device)
        elif head_type == "2-layer-mlp":
            head = nn.Sequential(
                nn.Linear(ckpt["head_in_dim"], ckpt["mlp_hidden"]),
                nn.GELU(),
                nn.Dropout(ckpt["mlp_dropout"]),
                nn.Linear(ckpt["mlp_hidden"], len(TRIGGER_LABELS)),
            ).to(device)
            head.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
        else:
            raise ValueError(f"Unknown head_type {head_type!r}")
        head.eval()

        return cls(
            tokenizer=tokenizer,
            encoder=encoder,
            head=head,
            device=device,
            numeric_mean=ckpt["numeric_mean"].to(device),
            numeric_std=ckpt["numeric_std"].to(device),
            threshold=ckpt.get("threshold", 0.5),
        )

    @torch.no_grad()
    def predict_probabilities(self, input_json: dict[str, Any]) -> dict[str, float]:
        """Returns sigmoid probabilities per trigger (6 keys)."""
        text = _serialize_input(input_json)
        emb = _embed([text], self.tokenizer, self.encoder, self.device)
        num = torch.tensor(
            [_extract_numeric_features(input_json)], dtype=torch.float32, device=self.device
        )
        num = (num - self.numeric_mean) / self.numeric_std
        extra = torch.tensor(
            [_extract_classifier_extra_features(input_json)],
            dtype=torch.float32,
            device=self.device,
        )
        x = torch.cat([emb, num, extra], dim=-1)
        logits = self.head(x).squeeze(0)
        probs = torch.sigmoid(logits)
        return {name: float(probs[i].item()) for i, name in enumerate(TRIGGER_LABELS)}

    def predict_triggers(self, input_json: dict[str, Any]) -> list[str]:
        """Apply threshold + fallback rule. Returns a list of trigger names
        in canonical order; never empty (default rule fills if no class fires)."""
        probs = self.predict_probabilities(input_json)
        triggers = [name for name in TRIGGER_LABELS if probs[name] >= self.threshold]
        if not triggers:
            return [DEFAULT_FALLBACK_LABEL]
        return triggers


def extract_valid_facts(input_json: dict[str, Any]) -> dict[str, Any]:
    """Pull citation-eligible facts from input_json — matches the contract
    that `reward_no_hallucinated_facts` and `reward_previous_dd_amount_correct`
    validate against.

    Returns:
        {
            "tariffs": [str, ...],            # case-preserved tariff names
            "rate_percentages": [float, ...], # rate change %s; reward only
                                              # validates citations with abs() >= 1.0
            "prev_amount": float | None,      # PR-E: latest.dd_amount - dd_amount_change,
                                              # the single allowed value for prev_amount_cited
        }

    Used to build a prompt-time constraint that surfaces the verbatim
    allowed-list to the LLM. The LLM is then instructed to only cite
    facts from this list, eliminating the no_halluc plateau by removing
    the LLM's freedom to invent facts.
    """
    ac = input_json.get("account_context", {}) or {}
    contracts = ac.get("contract_history") or []

    tariffs: list[str] = []
    seen_t: set[str] = set()
    for ch in contracts:
        name = (ch.get("tariff_name") or "").strip()
        if name and name.lower() not in seen_t:
            seen_t.add(name.lower())
            tariffs.append(name)

    pcts: list[float] = []
    seen_p: set[float] = set()
    for ch in contracts:
        for rh in ch.get("contract_rates_history") or []:
            for rate in rh.get("rates") or []:
                v = rate.get("change_since_previous_rate_percent")
                if v is None:
                    continue
                fv = round(float(v), 2)
                if fv not in seen_p:
                    seen_p.add(fv)
                    pcts.append(fv)
    pcts.sort()

    # PR-E: the single allowed prev_amount value
    # (matches `reward_previous_dd_amount_correct` expected = dd_amount - dd_amount_change).
    prev_amount: float | None = None
    latest = input_json.get("latest_dd_change") or {}
    dd_amount = latest.get("dd_amount")
    dd_amount_change = latest.get("dd_amount_change")
    if dd_amount is not None:
        prev_amount = round(float(dd_amount) - float(dd_amount_change or 0.0), 2)

    return {
        "tariffs": tariffs,
        "rate_percentages": pcts,
        "prev_amount": prev_amount,
    }


def extract_trigger_grounding(input_json: dict[str, Any]) -> dict[str, Any]:
    """Pull deterministic per-trigger grounding signals from input_json.

    The PR-G eval showed that when triggers like 'First DD review since
    account start', 'Missed/bounced DD payments', 'Manual reduction', or
    'Exemption Expiry' fire alone, row-level no_halluc collapses to ~0%
    because the model has no grounded source to anchor explanatory prose
    to and falls back to fabricated template language. These signals
    are 100%-recall / 0%-FPR derivable from the structured input — we
    just weren't surfacing them. Returns one block per trigger with the
    raw values needed to write a grounded explanation.
    """
    ac = input_json.get("account_context", {}) or {}
    hist = ac.get("dd_change_history") or []
    pay = ac.get("payment_history") or []
    latest = input_json.get("latest_dd_change") or {}

    grounding: dict[str, Any] = {}

    # First DD review since account start
    n_prior_dd = max(0, len(hist) - 1)
    if n_prior_dd == 0:
        grounding["first_dd_review"] = {
            "is_first": True,
            "n_prior_dd_entries": 0,
        }

    # Missed/bounced DD payments
    failed = [p for p in pay if not p.get("is_payment_successful", True)]
    if failed:
        most_recent = failed[-1]
        grounding["missed_payments"] = {
            "n_missed": len(failed),
            "most_recent_period": most_recent.get("payment_period"),
            "most_recent_amount_gbp": most_recent.get("transaction_amount_in_pounds"),
            "most_recent_timestamp": most_recent.get("transaction_timestamp"),
        }

    # Manual reduction (current or immediately-prior period)
    is_manual_now = bool(latest.get("is_amount_manually_reduced_lower_than_recommended_amount"))
    prev_dd = hist[-2] if len(hist) >= 2 else None
    prev_is_manual = bool(prev_dd and prev_dd.get("is_amount_manually_reduced_lower_than_recommended_amount"))
    if is_manual_now or prev_is_manual:
        src = latest if is_manual_now else (prev_dd or {})
        grounding["manual_reduction"] = {
            "active_in": "current_period" if is_manual_now else "previous_period",
            "manual_dd_amount_gbp": src.get("dd_amount"),
            "recommended_dd_amount_gbp": src.get("recommended_dd_amount"),
            "period_start": src.get("datetime_from"),
        }

    # Exemption Expiry — exemption was active in prior DD entry, not in current
    is_exemption_now = bool(latest.get("is_exemption"))
    prev_is_exemption = bool(prev_dd and prev_dd.get("is_exemption"))
    if prev_is_exemption and not is_exemption_now:
        grounding["exemption_expiry"] = {
            "expired_on": (prev_dd or {}).get("exemption_expiry_date"),
            "previous_dd_amount_gbp": (prev_dd or {}).get("dd_amount"),
            "previous_recommended_dd_amount_gbp": (prev_dd or {}).get("recommended_dd_amount"),
        }

    return grounding


def build_two_stage_prompt(
    base_messages: list[dict[str, Any]],
    triggers: list[str],
    *,
    valid_facts: dict[str, list] | None = None,
    trigger_grounding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Modify a `build_chat_messages` output to include predicted triggers.

    Appends a "RESPONSE TEMPLATE" suffix to the user-message content that
    pre-fills the trigger choices, leaving the LLM to infill `header` and
    `explanation`. Preserves the system message + the existing user prompt
    structure so the LLM still sees the full account context.

    If `valid_facts` is given (output of `extract_valid_facts`), also appends
    a "VALID FACTS" allowed-list block. The LLM is instructed to only cite
    these tariffs / rate-percentages and not invent others — this is the
    prompt-time constrained-decoding lever that targets the no_halluc plateau.

    `base_messages` is the output of `dd_explainer_data_generator.build_chat_messages(pin)`.
    """
    new_messages: list[dict[str, Any]] = []
    for msg in base_messages:
        if msg["role"] != "user":
            new_messages.append(msg)
            continue

        # Build the response-template suffix — pre-fills trigger field, leaves
        # header + explanation as templates the LLM should fill.
        template_explanations = [
            {
                "trigger": t,
                "header": "<fill in: short summary header for this trigger>",
                "explanation": "<fill in: 1-3 sentence explanation grounded in the account context>",
            }
            for t in triggers
        ]
        suffix = (
            "\n\n"
            "STAGE 1 PRE-DETERMINED TRIGGERS:\n"
            f"The following triggers have been identified for this DD change. "
            f"Use exactly these triggers in your response — do not add or remove any.\n"
            f"  {triggers}\n\n"
            "RESPONSE TEMPLATE (fill in the header + explanation fields, "
            "keep the trigger field as given):\n"
            f"{json.dumps({'explanations': template_explanations}, indent=2)}\n"
        )

        if valid_facts is not None:
            tariffs = valid_facts.get("tariffs") or []
            rate_pcts = valid_facts.get("rate_percentages") or []
            prev_amount = valid_facts.get("prev_amount")
            tariff_block = (
                "\n".join(f"  - \"{t}\"" for t in tariffs)
                if tariffs else "  (no tariff names available — do not cite any)"
            )
            # Reward only checks citations with abs() >= 1.0; surface those plus
            # 0%-style "no change" so the LLM has accurate context for prose.
            pct_block = (
                "\n".join(f"  - {p:+.2f}%" for p in rate_pcts)
                if rate_pcts else "  (no rate changes available — do not cite any)"
            )
            prev_amount_line = (
                f"  - £{prev_amount:.2f}" if prev_amount is not None
                else "  (no previous DD amount available)"
            )
            suffix += (
                "\n"
                "GROUNDING CONSTRAINT — VALID FACTS YOU MAY CITE:\n"
                "The ONLY tariff names, rate-change percentages, and previous DD "
                "amount allowed in your explanation are the ones listed below. They "
                "come verbatim from the account_context / latest_dd_change above. "
                "Do NOT invent or paraphrase any other tariff name, percentage, or "
                "amount; doing so will fail the no_hallucinated_facts / "
                "prev_amount_correct rubrics.\n\n"
                "Allowed tariff names:\n"
                f"{tariff_block}\n\n"
                "Allowed rate change percentages (cite verbatim, magnitudes >=1.0% are checked):\n"
                f"{pct_block}\n\n"
                "Allowed previous DD amount (cite verbatim if referenced):\n"
                f"{prev_amount_line}\n\n"
                "Use the structured slots `tariff_cited` / `rate_change_pct_cited` / "
                "`prev_amount_cited` (with the {tariff_cited} / {rate_change_pct_cited} / "
                "{prev_amount_cited} placeholders inside the explanation prose) so the "
                "rendered output is grounded in the slot values exactly. If a trigger "
                "does not require citing a tariff, rate, or amount, leave the "
                "corresponding slot null and describe qualitatively.\n"
            )

        if trigger_grounding:
            grounding_lines: list[str] = []
            label_map = {
                "first_dd_review": "First DD review since account start",
                "missed_payments": "Missed/bounced DD payments",
                "manual_reduction": "Manual reduction",
                "exemption_expiry": "Exemption Expiry",
            }
            for key, label in label_map.items():
                ctx = trigger_grounding.get(key)
                if not ctx:
                    continue
                ctx_json = json.dumps(ctx, indent=2, default=str)
                grounding_lines.append(f"- Trigger {label!r}:\n{ctx_json}")
            if grounding_lines:
                suffix += (
                    "\n"
                    "TRIGGER GROUNDING CONTEXT (use these structured anchors when "
                    "writing the explanation prose for the matching trigger; do not "
                    "invent template phrases like 'standard process', 'to cover the "
                    "difference', 'necessary catch-up' that are not anchored in these "
                    "values or the account context above):\n"
                    + "\n\n".join(grounding_lines)
                    + "\n"
                )
        # Content is a list of typed blocks (per `build_chat_messages` shape)
        new_content = []
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                new_content.append({"type": "text", "text": block["text"] + suffix})
            else:
                new_content.append(block)
        new_messages.append({"role": "user", "content": new_content})

    return new_messages


__all__ = [
    "TwoStageClassifier",
    "build_two_stage_prompt",
    "extract_valid_facts",
    "extract_trigger_grounding",
]
