"""Gated A/B evaluation — does the encoder gate lift E18's mean_total past 9.5?

Generates completions ONCE with E18's adapter on the heldout, then scores each
row two ways:

  ungated  — Gemma's actual completion, scored as today.
  gated    — if the gate flags the row OOD, substitute the canned fallback
             response text (no Gemma call needed for the substitution; but we
             do score it through the same rubric so the comparison is honest).

Reports per-rubric mean / pass-all for both passes plus the count of rows
the gate flipped, so you can see whether the gain (if any) comes from
shedding bad-LLM-output rows or paying a cost to do so.

Usage:

    uv run python scripts/gated_eval.py \
      --lora-path gemma_4_lora/train_v2_80gb/exp_18 \
      --gate-path data/outlier_head_v1.pt \
      --eval-heldout-n 1000 \
      --threshold 0.5

Heldout sampling is seed-stable (same `--seed` as the original training run
gives the same 1000-row sample, so this is comparable to E18's logged
heldout numbers).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import typer

# Module-level helpers that stay outside train.py so we don't have to load
# its full hydra setup.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dd_explainer_gate import GateModel, fallback_response  # noqa: E402
from dd_explainer_rewards import score_completion  # noqa: E402
from train import (  # noqa: E402
    _aggregate_scores,
    _generate_batch,
    _load_dataset,
    _load_model,
    _setup_workspace_env,
)

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _format_agg(agg: dict[str, Any]) -> str:
    return (
        f"mean_total={agg.get('mean_total'):.3f}  "
        f"f1={agg.get('f1_triggers_mean'):.3f}  "
        f"no_halluc={agg.get('no_hallucinated_facts_mean'):.3f}  "
        f"pass_all={agg.get('pass_all_pct')}%"
    )


@app.command()
def main(
    lora_path: Path = typer.Option(
        Path("gemma_4_lora/train_v2_80gb/exp_18"),
        help="Path to the saved LoRA adapter (default: E18 — current branch champion).",
    ),
    gate_path: Path = typer.Option(
        Path("data/outlier_head_v1.pt"),
        help="Path to the trained gate head (built by scripts/train_outlier_encoder.py).",
    ),
    model_name: str = typer.Option("unsloth/gemma-4-E4B-it"),
    data_dir: Path = typer.Option(Path("data"), help="Where the dd_dataset_*.jsonl lives"),
    eval_heldout_n: int = typer.Option(1000, help="Number of heldout rows to score."),
    threshold: float = typer.Option(0.5, help="Gate threshold P(OOD) — rows above are flagged."),
    seed: int = typer.Option(42, help="Heldout-sampling seed (must match training)."),
    max_seq_length: int = typer.Option(8192),
    max_completion_length: int = typer.Option(1024),
    batch_size: int = typer.Option(32),
    lora_rank: int = typer.Option(128),
    out: Path = typer.Option(
        Path("data/gated_eval_v1.json"),
        help="Where to dump the comparison JSON.",
    ),
) -> None:
    """Run the gated A/B and dump a comparison JSON."""
    _setup_workspace_env()

    typer.echo(f"loading gate head from {gate_path}…")
    gate = GateModel.load(gate_path)
    typer.echo(f"  gate ready: features={gate.features}, head_in_dim={gate.head.in_features}")

    typer.echo(f"loading {model_name} + LoRA from {lora_path}…")
    model, tokenizer = _load_model(model_name, max_seq_length, lora_rank, lora_path=lora_path)

    _, heldout_ds = _load_dataset(data_dir, heldout_n=eval_heldout_n, seed=seed)
    n = len(heldout_ds)
    typer.echo(f"heldout ready: {n} rows")

    typer.echo("scoring gate on every heldout row…")
    t0 = time.monotonic()
    gate_scores = [gate.predict_outlier_score(heldout_ds[i]["input_json"]) for i in range(n)]
    n_flagged = sum(1 for s in gate_scores if s >= threshold)
    typer.echo(
        f"  gate done in {time.monotonic() - t0:.1f}s  "
        f"flagged {n_flagged}/{n} ({n_flagged / n:.1%}) at threshold={threshold}"
    )

    typer.echo("running Gemma on the full heldout (temp=0, batched)…")
    t0 = time.monotonic()
    completions: list[str] = []
    for start in range(0, n, batch_size):
        chunk = [heldout_ds[i]["prompt"] for i in range(start, min(start + batch_size, n))]
        completions.extend(_generate_batch(model, tokenizer, chunk, max_completion_length))
        done = min(start + batch_size, n)
        if done % (batch_size * 4) == 0 or done == n:
            typer.echo(f"  generation: {done}/{n}")
    typer.echo(f"  generation done in {(time.monotonic() - t0) / 60:.1f}min")

    fallback_text = json.dumps(fallback_response())

    ungated_rows: list[dict[str, float]] = []
    gated_rows: list[dict[str, float]] = []
    for i in range(n):
        gt = sorted(heldout_ds[i]["ground_truth_triggers"])
        inp = heldout_ds[i]["input_json"]
        ungated_score = score_completion(completions[i], gt, inp)
        ungated_rows.append(ungated_score)
        if gate_scores[i] >= threshold:
            gated_rows.append(score_completion(fallback_text, gt, inp))
        else:
            gated_rows.append(ungated_score)

    agg_ungated = _aggregate_scores(ungated_rows)
    agg_gated = _aggregate_scores(gated_rows)

    typer.echo("\n=== heldout (n={n}) — E18 adapter ===".format(n=n))
    typer.echo(f"  ungated:  {_format_agg(agg_ungated)}")
    typer.echo(f"  gated:    {_format_agg(agg_gated)}")

    delta_keys = (
        ("mean_total", "mean_total"),
        ("f1", "f1_triggers_mean"),
        ("no_halluc", "no_hallucinated_facts_mean"),
        ("well_formed", "well_formed_mean"),
        ("prev_amount", "prev_amount_correct_mean"),
        ("pass_all_pct", "pass_all_pct"),
    )
    typer.echo("\n=== delta (gated - ungated) ===")
    for label, key in delta_keys:
        u = agg_ungated.get(key, 0.0)
        g = agg_gated.get(key, 0.0)
        typer.echo(f"  {label:18s} {u:>8.3f} -> {g:>8.3f}   delta={g - u:+.3f}")

    out_payload = {
        "lora_path": str(lora_path),
        "gate_path": str(gate_path),
        "threshold": threshold,
        "n_heldout": n,
        "n_flagged": n_flagged,
        "gate_flag_rate": n_flagged / n if n else 0.0,
        "ungated": agg_ungated,
        "gated": agg_gated,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, default=str))
    typer.echo(f"\nwrote {out}")


if __name__ == "__main__":
    app()
