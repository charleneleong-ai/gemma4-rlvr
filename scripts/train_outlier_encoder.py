"""Train a linear OOD-detection head on top of a frozen sentence encoder.

Pipeline (deliberately simple — see `docs/experiments/dd_explainer/encoder_outlier/v0_gate.md`):

    account_context (dict)
      -> json.dumps + truncate to 512 tokens
      -> frozen BAAI/bge-small-en-v1.5 forward (33M params, 384-d)
      -> mean-pool over tokens (attention-mask weighted)
      -> nn.Linear(384, 1)
      -> sigmoid -> P(OOD)

Frozen encoder + linear head means the *only* trainable parameters are
the head (~385 weights). The pretrained representation does the heavy
lifting — if it already encodes "this looks like a real account context
vs. an obviously-broken one", the half-plane decision boundary is enough.

Train/heldout split is 50/50 (100 rows each), stratified by `is_outlier`
so both halves see all 6 mutations. Reports overall AUROC + per-mutation
AUROC for diagnosis (which failure modes the encoder catches vs. misses).

Usage:

    uv run python scripts/train_outlier_encoder.py
    uv run python scripts/train_outlier_encoder.py --epochs 100 --lr 1e-2
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import typer
from transformers import AutoModel, AutoTokenizer

app = typer.Typer(add_completion=False, no_args_is_help=False)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DATA = Path("data/outlier_set_v0.jsonl")


def _load_outlier_set(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_frac: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split with equal OOD/in-dist counts per side. Mutations within OOD are
    NOT explicitly stratified — random shuffle within each label block keeps
    the test simple and the rounded-down counts give roughly even per-mutation
    coverage at n=200."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["is_outlier"] == 1]
    neg = [r for r in rows if r["is_outlier"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_n = int(len(pos) * train_frac)
    neg_n = int(len(neg) * train_frac)
    train = pos[:pos_n] + neg[:neg_n]
    heldout = pos[pos_n:] + neg[neg_n:]
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def _serialize(row: dict[str, Any]) -> str:
    """Compact JSON of account_context — what the encoder sees. Keep it
    deterministic (sort_keys) so re-runs hash to the same input. `default=str`
    handles datetime objects that may arrive from HF Datasets after parsing."""
    return json.dumps(row["input_json"]["account_context"], sort_keys=True, default=str)


# Domain features that bypass the encoder's blind spot for sign / magnitude /
# array-length OOD. Computed from raw input_json — no language model needed,
# just JSON traversal. Concatenated to the encoder embedding before the head.
NUMERIC_FEATURE_NAMES = (
    "dd_amount",
    "dd_amount_change",
    "recommended_dd_amount",
    "dd_to_recommended_ratio",
    "n_payment_records",
    "n_contracts",
    "n_dd_changes",
    "n_missed_payments",
    "frac_missed_payments",
)


def _extract_numeric_features(input_json: dict[str, Any]) -> list[float]:
    """Extract domain features in the order of NUMERIC_FEATURE_NAMES.

    These cover the mutations bge can't see:
      - dd_amount sign + magnitude (negative_numerics, large_debt)
      - dd_amount_change magnitude (significant_increase)
      - n_payment_records (truncate_payment_history)
      - n_contracts, n_dd_changes (drop_*, empty_*)
      - n_missed_payments + frac_missed_payments (many_missed_payments)

    Z-scoring happens at the call site once train statistics are known.
    """
    ldd = input_json.get("latest_dd_change", {}) or {}
    ac = input_json.get("account_context", {}) or {}
    dd = float(ldd.get("dd_amount") or 0.0)
    rec = float(ldd.get("recommended_dd_amount") or 0.0)
    ph = ac.get("payment_history") or []
    n_missed = sum(1 for p in ph if p.get("is_payment_successful") is False)
    return [
        dd,
        float(ldd.get("dd_amount_change") or 0.0),
        rec,
        dd / rec if rec else 0.0,
        float(len(ph)),
        float(len(ac.get("contract_history") or [])),
        float(len(ac.get("dd_change_history") or [])),
        float(n_missed),
        n_missed / len(ph) if ph else 0.0,
    ]


def _zscore_fit(features: torch.Tensor, indist_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std on the *in-distribution* train rows only — so OOD
    rows with extreme values don't contaminate the normalisation. Returns
    (mean, std) tensors with std clipped at 1e-6 to avoid div-by-zero on
    constant features."""
    indist = features[indist_mask]
    mean = indist.mean(dim=0)
    std = indist.std(dim=0).clamp(min=1e-6)
    return mean, std


@torch.no_grad()
def _embed(
    texts: list[str],
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Frozen forward pass + mean-pool. Returns [N, 384] tensor on device."""
    out: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        last = encoder(**enc).last_hidden_state  # [B, L, D]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # bge guidance: L2-normalise before downstream use.
        pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        out.append(pooled)
    return torch.cat(out, dim=0)


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U formulation. O(n²) — fine for n=100."""
    pos = [s for s, lab in zip(scores, labels) if lab == 1]
    neg = [s for s, lab in zip(scores, labels) if lab == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(p > n for p in pos for n in neg)
    ties = sum(p == n for p in pos for n in neg)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


@app.command()
def main(
    data: Path = typer.Option(DEFAULT_DATA, help="Outlier set JSONL"),
    model_name: str = typer.Option(DEFAULT_MODEL, help="HF encoder repo id"),
    features: str = typer.Option(
        "text",
        help="Input features: 'text' (bge embedding only) or 'text+numeric' "
        "(bge embedding concatenated with z-scored manual numeric features). "
        "v0 = text; v1 = text+numeric.",
    ),
    epochs: int = typer.Option(100, help="Linear-head training epochs"),
    lr: float = typer.Option(1e-2, help="AdamW learning rate"),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay"),
    seed: int = typer.Option(42, help="RNG seed for split + init"),
    save_head: Path | None = typer.Option(
        Path("data/outlier_head_v0.pt"),
        help="Path to save trained linear head (None to skip).",
    ),
) -> None:
    """Train + evaluate a linear OOD head on frozen bge embeddings."""
    if features not in ("text", "text+numeric"):
        raise typer.BadParameter(f"--features must be 'text' or 'text+numeric'. Got {features!r}.")
    if not data.exists():
        raise typer.BadParameter(
            f"{data} not found — run scripts/build_outlier_set.py first."
        )

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"device: {device}  features: {features}")

    typer.echo(f"loading {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = encoder.config.hidden_size
    typer.echo(f"encoder loaded: {embed_dim}-d hidden, {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params (frozen)")

    rows = _load_outlier_set(data)
    train_rows, heldout_rows = _stratified_split(rows, seed=seed)
    typer.echo(f"split: {len(train_rows)} train  /  {len(heldout_rows)} heldout")

    train_texts = [_serialize(r) for r in train_rows]
    heldout_texts = [_serialize(r) for r in heldout_rows]
    train_labels = torch.tensor([r["is_outlier"] for r in train_rows], dtype=torch.float32, device=device)
    heldout_labels = torch.tensor([r["is_outlier"] for r in heldout_rows], dtype=torch.float32, device=device)

    typer.echo("embedding train + heldout (frozen forward, no grad)…")
    train_emb = _embed(train_texts, tokenizer, encoder, device)
    heldout_emb = _embed(heldout_texts, tokenizer, encoder, device)

    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None
    if features == "text+numeric":
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
        # z-score on in-distribution train rows so OOD extremes don't poison
        # the normalisation statistics
        indist_mask = train_labels == 0
        feature_mean, feature_std = _zscore_fit(train_num, indist_mask)
        train_num = (train_num - feature_mean) / feature_std
        heldout_num = (heldout_num - feature_mean) / feature_std
        train_emb = torch.cat([train_emb, train_num], dim=-1)
        heldout_emb = torch.cat([heldout_emb, heldout_num], dim=-1)
        typer.echo(
            f"concat'd numeric features: +{train_num.shape[-1]} dims "
            f"({', '.join(NUMERIC_FEATURE_NAMES)})"
        )

    head_in_dim = train_emb.shape[-1]
    head = nn.Linear(head_in_dim, 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    typer.echo(f"training linear head: {sum(p.numel() for p in head.parameters())} params, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(train_emb).squeeze(-1)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            with torch.no_grad():
                head.eval()
                train_logits = head(train_emb).squeeze(-1)
                heldout_logits = head(heldout_emb).squeeze(-1)
                train_auroc = _auroc(train_logits.tolist(), train_labels.int().tolist())
                heldout_auroc = _auroc(heldout_logits.tolist(), heldout_labels.int().tolist())
            typer.echo(
                f"  epoch {epoch:>3}  loss={loss.item():.4f}  "
                f"train_auroc={train_auroc:.3f}  heldout_auroc={heldout_auroc:.3f}"
            )

    head.eval()
    with torch.no_grad():
        heldout_logits = head(heldout_emb).squeeze(-1)
    heldout_scores = heldout_logits.tolist()
    heldout_labels_list = heldout_labels.int().tolist()
    overall = _auroc(heldout_scores, heldout_labels_list)

    by_mutation: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(heldout_rows, heldout_scores):
        if row["is_outlier"] == 1:
            by_mutation[row["mutation"]].append((score, 1))
    indist_scores = [s for s, lab in zip(heldout_scores, heldout_labels_list) if lab == 0]
    for mut, ood_scored in by_mutation.items():
        scores = [s for s, _ in ood_scored] + indist_scores
        labels = [1] * len(ood_scored) + [0] * len(indist_scores)
        by_mutation[mut] = _auroc(scores, labels)

    typer.echo("\n=== heldout AUROC ===")
    typer.echo(f"  overall                          {overall:.3f}")
    for mut in sorted(by_mutation.keys()):
        typer.echo(f"  {mut:30s}   {by_mutation[mut]:.3f}")

    # Tail-tag false-positive rate on in-dist rows: how often does the gate
    # flag a legitimate-but-extreme customer (high_debt / high_increase /
    # many_missed)? Tags are independent — a row can carry multiple, so we
    # report per-tag rate plus the "no tag" baseline.
    threshold = 0.0  # logit > 0 → P(OOD) > 0.5
    indist_rows = [r for r in heldout_rows if r["is_outlier"] == 0]
    indist_scores_ordered = [s for r, s in zip(heldout_rows, heldout_scores) if r["is_outlier"] == 0]
    typer.echo("\n=== in-dist tail false-positive rate (logit > 0 means flagged) ===")
    all_tags = ("high_debt", "high_change", "many_missed", "in_credit")
    for tag in all_tags:
        sliced = [s for r, s in zip(indist_rows, indist_scores_ordered) if tag in (r.get("tail_flags") or [])]
        if not sliced:
            continue
        flagged = sum(1 for s in sliced if s > threshold)
        typer.echo(f"  tail.{tag:25s}   {flagged}/{len(sliced)}  ({flagged/len(sliced):.0%})")
    untagged = [s for r, s in zip(indist_rows, indist_scores_ordered) if not (r.get("tail_flags") or [])]
    if untagged:
        flagged = sum(1 for s in untagged if s > threshold)
        typer.echo(f"  tail.none{'':22s}   {flagged}/{len(untagged)}  ({flagged/len(untagged):.0%})")

    if save_head is not None:
        save_head.parent.mkdir(parents=True, exist_ok=True)
        ckpt: dict[str, Any] = {
            "weight": head.weight.detach().cpu(),
            "bias": head.bias.detach().cpu(),
            "encoder": model_name,
            "features": features,
            "embed_dim": embed_dim,
            "head_in_dim": head_in_dim,
            "heldout_auroc": overall,
        }
        if features == "text+numeric":
            ckpt["numeric_feature_names"] = list(NUMERIC_FEATURE_NAMES)
            ckpt["numeric_mean"] = feature_mean.detach().cpu()
            ckpt["numeric_std"] = feature_std.detach().cpu()
        torch.save(ckpt, save_head)
        typer.echo(f"\nsaved head to {save_head}")


if __name__ == "__main__":
    app()
