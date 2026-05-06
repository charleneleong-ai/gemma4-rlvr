"""Multi-label trigger classifier — Stage 1 of the two-stage pipeline.

Promotes the encoder-outlier gate architecture from binary OOD detection
to 7-way multi-label trigger classification:

    account_context + latest_dd_change (JSON-stringified, truncated)
      -> bge-small-en-v1.5 (frozen, 33M params, 384-d)
      -> mean-pool over tokens (attention-mask weighted)
      -> concat 9 manual numeric features (z-scored vs in-dist train)
      -> nn.Linear(393, 7)
      -> sigmoid -> per-trigger probability
      -> threshold 0.5 -> binary labels per trigger

Same architecture as `dd_explainer_gate.GateModel` but the head is now
7-d output instead of 1-d. Trained with `BCEWithLogitsLoss` against the
`ground_truth_triggers` enum-list label.

Why this beats the in-LLM trigger picker:

  - Stage 1 sees only the structured input. No prompt overhead, no rollouts.
  - 33M frozen + ~2.7K trainable params (vs E18's 280M LoRA params at r=128).
    Trains in ~5 min on A100 vs E18's ~70 min.
  - Macro-f1 at the classifier feeds into the GRPO rubric's f1_triggers
    reward at the explanation stage. If classifier macro-f1 = 0.9, Stage 2
    f1_triggers ~= 9-10 by construction (vs E18's 7.745).

Usage:

    uv run python scripts/train_trigger_classifier.py
    uv run python scripts/train_trigger_classifier.py --epochs 200 --lr 5e-3
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import typer
from transformers import AutoModel, AutoTokenizer

# Reuse the gate's helpers — same architectural primitives. scripts/ isn't
# a package, so add it to sys.path explicitly (matches the pattern in
# dd_explainer_gate.py).
sys.path.insert(0, str(Path(__file__).parent))
from train_outlier_encoder import (  # noqa: E402
    NUMERIC_FEATURE_NAMES,
    _embed,
    _extract_numeric_features,
    _zscore_fit,
)

# Trigger-discriminator features — boolean / numeric markers that the data
# generator uses to define the per-trigger ground truth. Three encoders
# (bge-small, bge-base, qwen3-0.6B) all plateau at F1=0.55-0.70 on Manual
# reduction / Exemption Expiry / Change in usage because those triggers are
# defined by isolated booleans the linear head can't pull out of the JSON
# embedding. Surfacing them explicitly turns the multi-label task into a
# rule-decoding one for these 3 classes.
CLASSIFIER_EXTRA_FEATURE_NAMES = (
    "older_change_is_manually_reduced",       # → Manual reduction
    "older_change_reason_is_customer_request",# → Manual reduction
    "older_change_is_exemption",              # → Exemption Expiry
    "older_change_exemption_expired",         # → Exemption Expiry
    "abs_electricity_change_percent",         # → Change in usage
    "abs_gas_change_percent",                 # → Change in usage
    # v5 (2026-05-06): add features for the 3 triggers that previously had
    # NO discriminator — Missed/bounced, First DD review, Change in unit rates.
    # The E28 + templates retrospective showed these dragged f1 down (745/1000
    # perfect, with 3-trigger combos missing 35-65%). All four predicates are
    # 100% recall / 0% FPR derivable from input fields.
    "n_failed_payments",                      # → Missed/bounced DD payments
    "is_first_dd_review",                     # → First DD review since account start
    "max_abs_rate_change_percent",            # → Change in unit rates (>= 1% magnitude)
)


def _extract_classifier_extra_features(input_json: dict[str, Any]) -> list[float]:
    """Trigger-discriminator features. Returned in the order of
    CLASSIFIER_EXTRA_FEATURE_NAMES. Boolean values become 0.0/1.0."""
    ac = input_json.get("account_context", {}) or {}
    ldd = input_json.get("latest_dd_change", {}) or {}
    dd_history = ac.get("dd_change_history") or []
    older = dd_history[0] if dd_history else {}

    # Bool markers — generator sets these on `dd_change_history[0]`
    is_manual = bool(older.get("is_amount_manually_reduced_lower_than_recommended_amount"))
    is_customer_request = (older.get("reason_for_DD_change") == "customer request")
    is_exemption = bool(older.get("is_exemption"))

    # Exemption expired = expiry_date < latest_dd_change.datetime_from
    exemption_expired = False
    expiry = older.get("exemption_expiry_date")
    latest_from = ldd.get("datetime_from")
    if expiry and latest_from:
        # Both should be ISO-format strings; lexicographic compare works
        # for valid ISO dates even with mixed datetime / date precision.
        exemption_expired = str(expiry) < str(latest_from)

    # Consumption change % per fuel
    pch = ac.get("projected_consumption_history") or {}
    elec = (pch.get("electricity") or {}).get("change_percent") or 0.0
    gas = (pch.get("gas") or {}).get("change_percent") or 0.0

    # v5 features
    pay = ac.get("payment_history") or []
    n_failed = sum(1 for p in pay if not p.get("is_payment_successful", True))
    # First DD review = only one entry in dd_change_history (no priors)
    is_first_dd_review = len(dd_history) <= 1
    # Max abs rate change % across contract_history.contract_rates_history.rates
    max_abs_rate = 0.0
    for ch in (ac.get("contract_history") or []):
        for rh in (ch.get("contract_rates_history") or []):
            for rate in (rh.get("rates") or []):
                v = rate.get("change_since_previous_rate_percent")
                if v is not None:
                    a = abs(float(v))
                    if a > max_abs_rate:
                        max_abs_rate = a

    return [
        1.0 if is_manual else 0.0,
        1.0 if is_customer_request else 0.0,
        1.0 if is_exemption else 0.0,
        1.0 if exemption_expired else 0.0,
        abs(float(elec)),
        abs(float(gas)),
        float(n_failed),
        1.0 if is_first_dd_review else 0.0,
        float(max_abs_rate),
    ]

app = typer.Typer(add_completion=False, no_args_is_help=False)

# Six explicit triggers — no "No triggers identified" in the classifier output.
# That's a default rule: if Stage 1 predicts none of the 6, Stage 2 emits
# {"trigger": "No triggers identified"} as the fallback. Avoids the
# class-imbalance drag from a 2.5% prevalence class.
TRIGGER_LABELS = (
    "Manual reduction",
    "Exemption Expiry",
    "Change in usage",
    "Change in unit rates",
    "Missed/bounced DD payments",
    "First DD review since account start",
)
DEFAULT_FALLBACK_LABEL = "No triggers identified"
N_TRIGGERS = len(TRIGGER_LABELS)
TRIGGER_TO_IDX = {t: i for i, t in enumerate(TRIGGER_LABELS)}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DATA = Path("data/dd_dataset_20260427T201521Z_5500rows.jsonl")


def _load_dd_dataset(path: Path) -> list[dict[str, Any]]:
    """Load the synthetic dataset, skipping the first metadata row."""
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_label(triggers: list[str]) -> torch.Tensor:
    """Convert a list of trigger names to a 7-d binary label tensor."""
    label = torch.zeros(N_TRIGGERS, dtype=torch.float32)
    for t in triggers:
        if t in TRIGGER_TO_IDX:
            label[TRIGGER_TO_IDX[t]] = 1.0
    return label


def _serialize_input(input_json: dict[str, Any]) -> str:
    """JSON-stringify account_context + latest_dd_change as the encoder text.
    `default=str` handles datetime objects from HF Datasets."""
    return json.dumps(
        {
            "account_context": input_json.get("account_context", {}),
            "latest_dd_change": input_json.get("latest_dd_change", {}),
        },
        sort_keys=True,
        default=str,
    )


def _per_trigger_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """Compute precision / recall / f1 per trigger + macro-f1.

    Returns a dict with one key per trigger plus '_macro_f1' / '_micro_f1'.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    out: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for i, name in enumerate(TRIGGER_LABELS):
        p = preds[:, i]
        l = labels[:, i]
        tp = ((p == 1) & (l == 1)).sum().item()
        fp = ((p == 1) & (l == 0)).sum().item()
        fn = ((p == 0) & (l == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1_scores.append(f1)
    out["_macro_f1"] = sum(f1_scores) / len(f1_scores)
    # Micro-f1: aggregated TP/FP/FN across triggers
    total_tp = sum(out[name]["tp"] for name in TRIGGER_LABELS)
    total_fp = sum(out[name]["fp"] for name in TRIGGER_LABELS)
    total_fn = sum(out[name]["fn"] for name in TRIGGER_LABELS)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    out["_micro_f1"] = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    return out


def _exact_match(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of rows where the predicted trigger set exactly matches GT."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return ((preds == labels).all(dim=1)).float().mean().item()


def _rubric_f1_reward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fallback_labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Per-row F1, then mapped to the rubric reward `f1 * 12 - 2` (range [-2, +10])
    used by `reward_triggers_match_ground_truth`.

    Applies the default-rule for `No triggers identified`: rows where the
    classifier predicts ALL ZEROS (none of the 6 explicit triggers fire)
    are scored against `fallback_labels` (1.0 if ground truth was empty,
    0.0 otherwise). This matches what Stage 2 will do at inference: emit
    a `No triggers identified` explanation when Stage 1 predicts nothing.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    f1_per_row = []
    for i, (p, l) in enumerate(zip(preds, labels)):
        any_predicted = p.sum().item() > 0
        any_gt = l.sum().item() > 0
        # Fallback case: neither classifier nor GT picked any of the 6 triggers
        # AND ground truth was originally `No triggers identified` only.
        if not any_predicted and not any_gt and fallback_labels[i].item() > 0:
            f1_per_row.append(1.0)
            continue
        # Fallback emitted but GT had real triggers → f1 = 0
        if not any_predicted and any_gt:
            f1_per_row.append(0.0)
            continue
        # Classifier predicted real triggers but GT was empty (No_triggers_identified) → f1 = 0
        if any_predicted and not any_gt:
            f1_per_row.append(0.0)
            continue
        tp = ((p == 1) & (l == 1)).sum().item()
        fp = ((p == 1) & (l == 0)).sum().item()
        fn = ((p == 0) & (l == 1)).sum().item()
        if tp + fp + fn == 0:
            f1_per_row.append(1.0)
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_row.append(f1)
    mean_f1 = sum(f1_per_row) / len(f1_per_row)
    rubric_reward = mean_f1 * 12 - 2  # match dd_explainer_rewards.reward_triggers_match_ground_truth
    return {"per_row_f1_mean": mean_f1, "rubric_reward": rubric_reward}


def _sweep_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fallback_labels: torch.Tensor,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> None:
    """Print rubric reward at multiple thresholds — picks the operating point
    that maximises Stage 2's f1_triggers reward."""
    typer.echo("\n=== threshold sweep (per-row rubric reward) ===")
    typer.echo(f"  {'threshold':<10} {'per_row_f1':<12} {'rubric_reward':<14} {'macro_f1':<10} {'exact_match':<12}")
    for t in thresholds:
        rubric = _rubric_f1_reward(logits, labels, fallback_labels, t)
        macro = _per_trigger_metrics(logits, labels, t)["_macro_f1"]
        em = _exact_match(logits, labels, t)
        typer.echo(
            f"  {t:<10.2f} {rubric['per_row_f1_mean']:<12.3f} {rubric['rubric_reward']:<14.3f} {macro:<10.3f} {em:<12.3f}"
        )


@app.command()
def main(
    data: Path = typer.Option(DEFAULT_DATA, help="Source dataset JSONL"),
    model_name: str = typer.Option(DEFAULT_MODEL, help="HF encoder repo id"),
    n_train: int = typer.Option(5000, help="Train set size sampled from dataset"),
    n_heldout: int = typer.Option(500, help="Heldout set size sampled separately"),
    epochs: int = typer.Option(200, help="Linear-head training epochs"),
    lr: float = typer.Option(5e-3, help="AdamW learning rate"),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay"),
    threshold: float = typer.Option(0.5, help="Sigmoid threshold for binary prediction"),
    seed: int = typer.Option(42, help="RNG seed for split + init"),
    head_type: str = typer.Option(
        "linear",
        "--head-type",
        help="'linear' = single-layer head (v0-v3); '2-layer-mlp' = "
             "Linear(in→hidden) → GELU → Dropout → Linear(hidden→out). "
             "More capacity for non-linear class boundaries.",
    ),
    mlp_hidden: int = typer.Option(128, help="Hidden dim when --head-type=2-layer-mlp"),
    mlp_dropout: float = typer.Option(0.1, help="Dropout between MLP layers"),
    save_head: Path | None = typer.Option(
        Path("data/trigger_classifier_v0.pt"),
        help="Path to save trained linear head (None to skip).",
    ),
) -> None:
    """Train + evaluate a multi-label trigger classifier on frozen bge embeddings."""
    if not data.exists():
        raise typer.BadParameter(f"{data} not found")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"device: {device}  features: text+numeric")

    # Encoder
    typer.echo(f"loading {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = encoder.config.hidden_size
    typer.echo(f"encoder loaded: {embed_dim}-d hidden, frozen")

    # Dataset
    rows = _load_dd_dataset(data)
    typer.echo(f"loaded {len(rows)} rows from {data.name}")
    rng.shuffle(rows)
    train_rows = rows[:n_train]
    heldout_rows = rows[n_train : n_train + n_heldout]
    typer.echo(f"split: {len(train_rows)} train  /  {len(heldout_rows)} heldout")

    # Trigger frequency in train (informs class-imbalance loss weighting)
    train_label_counts = Counter()
    for r in train_rows:
        for t in r["ground_truth_triggers"]:
            train_label_counts[t] += 1
    typer.echo("trigger frequency in train:")
    for name in TRIGGER_LABELS:
        c = train_label_counts.get(name, 0)
        typer.echo(f"  {name:42s} {c:>5d}  ({c/len(train_rows):>5.1%})")

    # Texts + features + labels
    train_texts = [_serialize_input(r["input_json"]) for r in train_rows]
    heldout_texts = [_serialize_input(r["input_json"]) for r in heldout_rows]
    train_labels = torch.stack([_row_to_label(r["ground_truth_triggers"]) for r in train_rows]).to(device)
    heldout_labels = torch.stack([_row_to_label(r["ground_truth_triggers"]) for r in heldout_rows]).to(device)
    # Fallback flag: 1.0 iff GT was exactly [No_triggers_identified] alone
    heldout_fallback = torch.tensor(
        [
            1.0 if r["ground_truth_triggers"] == [DEFAULT_FALLBACK_LABEL] else 0.0
            for r in heldout_rows
        ],
        device=device,
    )

    typer.echo("embedding train + heldout (frozen forward, no grad)…")
    train_emb = _embed(train_texts, tokenizer, encoder, device)
    heldout_emb = _embed(heldout_texts, tokenizer, encoder, device)

    # Numeric features
    train_num = torch.tensor(
        [_extract_numeric_features(r["input_json"]) for r in train_rows],
        dtype=torch.float32,
        device=device,
    )
    heldout_num = torch.tensor(
        [_extract_numeric_features(r["input_json"]) for r in heldout_rows],
        dtype=torch.float32,
        device=device,
    )
    indist_mask = torch.ones(len(train_rows), dtype=torch.bool, device=device)
    feature_mean, feature_std = _zscore_fit(train_num, indist_mask)
    train_num = (train_num - feature_mean) / feature_std
    heldout_num = (heldout_num - feature_mean) / feature_std

    # Classifier-extra features: trigger-discriminator booleans/numerics that
    # the bge encoder was failing to surface from the JSON. NOT z-scored —
    # they're already in [0,1] for booleans and 0-25 for change_percent which
    # the head can scale via its weights.
    train_extra = torch.tensor(
        [_extract_classifier_extra_features(r["input_json"]) for r in train_rows],
        dtype=torch.float32,
        device=device,
    )
    heldout_extra = torch.tensor(
        [_extract_classifier_extra_features(r["input_json"]) for r in heldout_rows],
        dtype=torch.float32,
        device=device,
    )

    train_X = torch.cat([train_emb, train_num, train_extra], dim=-1)
    heldout_X = torch.cat([heldout_emb, heldout_num, heldout_extra], dim=-1)
    head_in_dim = train_X.shape[-1]
    typer.echo(
        f"input dim: {head_in_dim} = {embed_dim} embed + {train_num.shape[-1]} numeric + "
        f"{train_extra.shape[-1]} classifier-extra"
    )
    typer.echo(f"  numeric: {', '.join(NUMERIC_FEATURE_NAMES)}")
    typer.echo(f"  classifier-extra: {', '.join(CLASSIFIER_EXTRA_FEATURE_NAMES)}")

    # Class imbalance: pos_weight per trigger = (n_neg / n_pos)
    pos_weight = torch.tensor(
        [
            (len(train_rows) - train_label_counts.get(name, 0))
            / max(train_label_counts.get(name, 0), 1)
            for name in TRIGGER_LABELS
        ],
        dtype=torch.float32,
        device=device,
    )

    if head_type == "linear":
        head: nn.Module = nn.Linear(head_in_dim, N_TRIGGERS).to(device)
    elif head_type == "2-layer-mlp":
        head = nn.Sequential(
            nn.Linear(head_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, N_TRIGGERS),
        ).to(device)
    else:
        raise typer.BadParameter(
            f"--head-type must be 'linear' or '2-layer-mlp', got {head_type!r}"
        )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    typer.echo(
        f"head: type={head_type}, "
        f"{sum(p.numel() for p in head.parameters())} trainable params, "
        f"{epochs} epochs"
    )

    # Train
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(train_X)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            head.eval()
            with torch.no_grad():
                train_logits = head(train_X)
                heldout_logits = head(heldout_X)
                train_macro_f1 = _per_trigger_metrics(train_logits, train_labels, threshold)["_macro_f1"]
                heldout_macro_f1 = _per_trigger_metrics(heldout_logits, heldout_labels, threshold)["_macro_f1"]
                heldout_em = _exact_match(heldout_logits, heldout_labels, threshold)
            typer.echo(
                f"  epoch {epoch:>3}  loss={loss.item():.4f}  "
                f"train_macro_f1={train_macro_f1:.3f}  heldout_macro_f1={heldout_macro_f1:.3f}  heldout_exact_match={heldout_em:.3f}"
            )

    # Final eval
    head.eval()
    with torch.no_grad():
        heldout_logits = head(heldout_X)
    metrics = _per_trigger_metrics(heldout_logits, heldout_labels, threshold)
    em = _exact_match(heldout_logits, heldout_labels, threshold)

    typer.echo(f"\n=== heldout per-trigger metrics (threshold={threshold}) ===")
    for name in TRIGGER_LABELS:
        m = metrics[name]
        typer.echo(
            f"  {name:42s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"(tp={m['tp']:>3} fp={m['fp']:>3} fn={m['fn']:>3})"
        )
    typer.echo(f"\n  macro_f1                                   {metrics['_macro_f1']:.3f}")
    typer.echo(f"  micro_f1                                   {metrics['_micro_f1']:.3f}")
    typer.echo(f"  exact_match (all 7 dims correct)          {em:.3f}")

    # Per-row rubric reward — the metric that actually matters for Stage 2.
    rubric_default = _rubric_f1_reward(heldout_logits, heldout_labels, heldout_fallback, threshold)
    typer.echo(
        f"  per-row F1 mean (Stage 2 input)            {rubric_default['per_row_f1_mean']:.3f}"
    )
    typer.echo(
        f"  rubric reward (f1*12-2, range [-2,+10])    {rubric_default['rubric_reward']:.3f}"
    )
    typer.echo(f"  vs E18's f1_triggers reward                7.745")
    _sweep_thresholds(heldout_logits, heldout_labels, heldout_fallback)

    if save_head is not None:
        save_head.parent.mkdir(parents=True, exist_ok=True)
        ckpt: dict[str, Any] = {
            "encoder": model_name,
            "embed_dim": embed_dim,
            "head_in_dim": head_in_dim,
            "head_type": head_type,
            "trigger_labels": list(TRIGGER_LABELS),
            "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
            "numeric_mean": feature_mean.detach().cpu(),
            "numeric_std": feature_std.detach().cpu(),
            "threshold": threshold,
            "heldout_macro_f1": metrics["_macro_f1"],
            "heldout_micro_f1": metrics["_micro_f1"],
            "heldout_exact_match": em,
        }
        if head_type == "linear":
            ckpt["weight"] = head.weight.detach().cpu()
            ckpt["bias"] = head.bias.detach().cpu()
        else:  # 2-layer-mlp — save the full state_dict
            ckpt["state_dict"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            ckpt["mlp_hidden"] = mlp_hidden
            ckpt["mlp_dropout"] = mlp_dropout
        torch.save(ckpt, save_head)
        typer.echo(f"\nsaved classifier to {save_head}")


if __name__ == "__main__":
    app()
