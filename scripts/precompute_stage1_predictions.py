"""Pre-compute Stage 1 predictions for the full dd_explainer dataset.

Stage 2 GRPO training feeds the LLM a prompt that contains Stage 1's
predicted triggers as a templated response skeleton. Running Stage 1 at
each training-rollout step would be wasteful (predictions are a function
of input only — deterministic per row). This script does it once, keyed
by `row_index`, and saves `data/stage1_predictions_v3.json`:

    { row_index: [trigger_str, ...], ... }

train.py's `_load_dataset` then loads this JSON and uses
`build_two_stage_prompt` to inject the triggers into each row's prompt
during training data prep.

Usage:

    uv run python scripts/precompute_stage1_predictions.py \\
      --classifier-path data/trigger_classifier_v3_extra_features.pt \\
      --out data/stage1_predictions_v3.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))
from dd_explainer_two_stage import TwoStageClassifier  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_dataset_rows(data_dir: Path) -> list[dict[str, Any]]:
    """Load the newest dd_dataset_*_*rows.jsonl with row_index intact."""
    jsonl_files = sorted(data_dir.glob("dd_dataset_*_*rows.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No dd_dataset_*.jsonl in {data_dir}")
    path = jsonl_files[-1]
    typer.echo(f"loading {path.name}…")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("__meta__") is True:
                continue
            rows.append(r)
    typer.echo(f"  loaded {len(rows)} rows")
    return rows


@app.command()
def main(
    classifier_path: Path = typer.Option(
        Path("data/trigger_classifier_v3_extra_features.pt"),
        help="Saved Stage 1 classifier head.",
    ),
    data_dir: Path = typer.Option(Path("data"), help="Where dd_dataset_*.jsonl lives."),
    out: Path = typer.Option(
        Path("data/stage1_predictions_v3.json"),
        help="Output JSON: {row_index: [trigger_str, ...], ...}.",
    ),
) -> None:
    """Run Stage 1 over every dataset row and cache the predicted triggers."""
    typer.echo(f"loading Stage 1 classifier from {classifier_path}…")
    classifier = TwoStageClassifier.load(classifier_path)
    typer.echo(f"  ready (head_in_dim={classifier.head.in_features})")

    rows = _load_dataset_rows(data_dir)

    typer.echo("running Stage 1 on every row…")
    t0 = time.monotonic()
    predictions: dict[str, list[str]] = {}
    for i, row in enumerate(rows):
        triggers = classifier.predict_triggers(row["input_json"])
        # Use string keys so JSON round-trip is lossless even if row_index is None
        key = str(row.get("row_index", i))
        predictions[key] = triggers
        if (i + 1) % 500 == 0 or i == len(rows) - 1:
            typer.echo(f"  {i + 1}/{len(rows)}  ({(i + 1) / (time.monotonic() - t0):.0f} rows/s)")

    # Distribution sanity check
    from collections import Counter
    trigger_counter = Counter()
    multi_count = 0
    for triggers in predictions.values():
        for t in triggers:
            trigger_counter[t] += 1
        if len(triggers) > 1:
            multi_count += 1
    typer.echo("\ntrigger prediction distribution:")
    for trigger, count in sorted(trigger_counter.items(), key=lambda x: -x[1]):
        typer.echo(f"  {trigger:42s} {count:>5d} ({count / len(rows):.1%})")
    typer.echo(f"  multi-trigger rows: {multi_count}/{len(rows)} ({multi_count / len(rows):.1%})")

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier_path": str(classifier_path),
        "n_rows": len(rows),
        "predictions": predictions,
    }
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"\nwrote {out}  ({len(predictions)} predictions)")


if __name__ == "__main__":
    app()
