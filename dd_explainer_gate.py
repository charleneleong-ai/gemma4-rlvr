"""Inference-time encoder-outlier gate for the dd_explainer task.

Loads a trained encoder + linear head (saved by `scripts/train_outlier_encoder.py`)
and exposes:

  - `predict_outlier_score(input_json) -> float`       # P(OOD)
  - `should_gate(input_json, threshold=0.5) -> bool`
  - `fallback_response() -> dict`                      # canned "no triggers" output

Usage in an eval harness:

    from dd_explainer_gate import GateModel, fallback_response

    gate = GateModel.load("data/outlier_head_v1.pt")
    for row in heldout_rows:
        if gate.should_gate(row["input_json"]):
            response = fallback_response()        # short-circuit, no Gemma call
        else:
            response = run_gemma(model, row)
        score = score_completion(response, row, ...)

The fallback returns a `DirectDebitExplainerResponse` schema-shaped dict
with a single `TriggerExplanation` using the existing `No triggers identified`
enum value — no schema changes, no new training data, no rubric changes
required to wire it in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from scripts.train_outlier_encoder import (
    NUMERIC_FEATURE_NAMES,
    _embed,
    _extract_numeric_features,
)


def fallback_response() -> dict[str, Any]:
    """Canned response for OOD inputs — emits a `No triggers identified`
    explanation so the eval rubric sees a valid `DirectDebitExplainerResponse`
    without invoking the LLM. Schema matches `DirectDebitExplainerResponse`
    from `dd_explainer_data_generator.py`.

    Used by the encoder gate when the input fails the OOD check. By using
    the existing `No_triggers_identified` enum value we avoid having to
    extend the schema, retrain Gemma on a new label, or change the rubric.
    """
    return {
        "explanations": [
            {
                "trigger": "No triggers identified",
                "header": "Unable to explain change",
                "explanation": (
                    "The provided account context does not contain enough "
                    "information to identify a specific reason for this Direct "
                    "Debit change. Please review the account manually."
                ),
            }
        ]
    }


class GateModel:
    """Loads a saved encoder + linear head and scores raw input_json dicts.

    The encoder is frozen (no grad), runs on whatever device is available.
    Numeric features (when present in the saved checkpoint) are z-scored
    using the in-distribution mean/std saved at training time.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        encoder: AutoModel,
        head: nn.Linear,
        features: str,
        device: torch.device,
        numeric_mean: torch.Tensor | None = None,
        numeric_std: torch.Tensor | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.head = head
        self.features = features
        self.device = device
        self.numeric_mean = numeric_mean
        self.numeric_std = numeric_std

    @classmethod
    def load(cls, head_path: Path | str) -> GateModel:
        """Load a head saved by `train_outlier_encoder.py`. Re-loads the
        encoder by repo-id so the embedding side matches what the head was
        trained against."""
        ckpt = torch.load(Path(head_path), map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder"])
        encoder = AutoModel.from_pretrained(ckpt["encoder"]).to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False

        head = nn.Linear(ckpt["head_in_dim"], 1).to(device)
        head.weight.data = ckpt["weight"].to(device)
        head.bias.data = ckpt["bias"].to(device)
        head.eval()

        return cls(
            tokenizer=tokenizer,
            encoder=encoder,
            head=head,
            features=ckpt["features"],
            device=device,
            numeric_mean=ckpt.get("numeric_mean").to(device) if ckpt.get("numeric_mean") is not None else None,
            numeric_std=ckpt.get("numeric_std").to(device) if ckpt.get("numeric_std") is not None else None,
        )

    @torch.no_grad()
    def predict_outlier_score(self, input_json: dict[str, Any]) -> float:
        """Returns P(OOD) in [0, 1] — sigmoid of the head logit.

        `default=str` handles datetime objects that arrive from HuggingFace
        Datasets (the heldout rows can carry parsed datetimes rather than the
        ISO strings the synthetic generator wrote out).
        """
        text = json.dumps(input_json.get("account_context", {}), sort_keys=True, default=str)
        emb = _embed([text], self.tokenizer, self.encoder, self.device)
        if self.features == "text+numeric":
            assert self.numeric_mean is not None and self.numeric_std is not None
            num = torch.tensor(
                [_extract_numeric_features(input_json)],
                dtype=torch.float32,
                device=self.device,
            )
            num = (num - self.numeric_mean) / self.numeric_std
            emb = torch.cat([emb, num], dim=-1)
        logit = self.head(emb).squeeze().item()
        return float(torch.sigmoid(torch.tensor(logit)).item())

    def should_gate(self, input_json: dict[str, Any], threshold: float = 0.5) -> bool:
        """True if the gate should short-circuit to `fallback_response()`."""
        return self.predict_outlier_score(input_json) >= threshold


__all__ = ["GateModel", "fallback_response", "NUMERIC_FEATURE_NAMES"]
