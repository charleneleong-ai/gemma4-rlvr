"""Compute Stage-1 classifier sigmoid probabilities for an eval per_row dump.

Produces a JSON the rescore path can use to gate `--backfill-triggers`:

    {
        "classifier_path": "...",
        "n_rows": 1000,
        "probabilities": {
            "0": {"Change in usage": 0.99, ...},
            "1": {...},
            ...
        }
    }

Usage:

    uv run python scripts/compute_stage1_probs.py \\
      --classifier-path data/trigger_classifier_v5_bool_features.pt \\
      --per-row data/eval_e28_v5_templates_n1000.per_row.jsonl \\
      --out data/stage1_probs_v5_eval_n1000.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from dd_explainer_two_stage import TwoStageClassifier  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    classifier_path: Path = typer.Option(
        Path("data/trigger_classifier_v5_bool_features.pt"),
        help="Stage-1 classifier head.",
    ),
    per_row: Path = typer.Option(
        ...,
        "--per-row",
        help="Eval per_row.jsonl produced by scripts/two_stage_eval.py "
             "(needs `i` and `input_json` keys per row).",
    ),
    out: Path = typer.Option(..., "--out", help="Where to write the probs JSON."),
) -> None:
    typer.echo(f"loading classifier {classifier_path.name}…")
    classifier = TwoStageClassifier.load(classifier_path)

    rows = [json.loads(l) for l in per_row.read_text().splitlines() if l.strip()]
    typer.echo(f"loaded {len(rows)} rows from {per_row.name}")

    typer.echo("running Stage-1 inference…")
    t0 = time.monotonic()
    probabilities: dict[str, dict[str, float]] = {}
    for i, r in enumerate(rows):
        probs = classifier.predict_probabilities(r["input_json"])
        probabilities[str(r["i"])] = probs
        if (i + 1) % 200 == 0 or i == len(rows) - 1:
            typer.echo(
                f"  {i + 1}/{len(rows)}  ({(i + 1) / (time.monotonic() - t0):.1f} rows/s)"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "classifier_path": str(classifier_path),
        "n_rows": len(rows),
        "probabilities": probabilities,
    }, indent=2))
    typer.echo(f"\nwrote {out}")


if __name__ == "__main__":
    app()
