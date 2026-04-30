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


def build_two_stage_prompt(
    base_messages: list[dict[str, Any]],
    triggers: list[str],
) -> list[dict[str, Any]]:
    """Modify a `build_chat_messages` output to include predicted triggers.

    Appends a "RESPONSE TEMPLATE" suffix to the user-message content that
    pre-fills the trigger choices, leaving the LLM to infill `header` and
    `explanation`. Preserves the system message + the existing user prompt
    structure so the LLM still sees the full account context.

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
        # Content is a list of typed blocks (per `build_chat_messages` shape)
        new_content = []
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                new_content.append({"type": "text", "text": block["text"] + suffix})
            else:
                new_content.append(block)
        new_messages.append({"role": "user", "content": new_content})

    return new_messages


__all__ = ["TwoStageClassifier", "build_two_stage_prompt"]
