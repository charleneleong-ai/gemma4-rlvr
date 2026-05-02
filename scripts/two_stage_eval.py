"""Two-stage A/B evaluation — E18 vanilla vs E18 + Stage-1-injected prompt.

Generates completions twice on the same heldout (same seed, same Gemma adapter):

  vanilla: build_chat_messages(pin)  -> Gemma -> completion
  twostage: build_two_stage_prompt(build_chat_messages(pin), Stage1.predict_triggers)
            -> Gemma -> completion

Scores both with the existing rubric. The two-stage variant should score
higher on `f1_triggers` (Stage 1 hit rubric=8.77 vs E18's 7.745) and at
least as high on `no_halluc` / `well_formed` since the LLM now has a
simpler job (fill in prose only).

Usage:

    uv run python scripts/two_stage_eval.py \\
      --classifier-path data/trigger_classifier_v3_extra_features.pt \\
      --lora-path gemma_4_lora/train_v2_80gb/exp_18 \\
      --eval-heldout-n 1000 \\
      --out data/two_stage_eval_v0.json

Heldout sampling is seed-stable (matches the original training split).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import typer

# Reuse the existing eval helpers.
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch import GPUMonitor  # noqa: E402

from dd_explainer_rewards import score_completion  # noqa: E402
from dd_explainer_two_stage import (  # noqa: E402
    TwoStageClassifier,
    build_two_stage_prompt,
    extract_valid_facts,
)
from dd_explainer_slot_decoder import (  # noqa: E402
    build_slot_enforcement_schema,
    build_slot_prefix_fn,
    build_tokenizer_data,
)
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
    classifier_path: Path = typer.Option(
        Path("data/trigger_classifier_v3_extra_features.pt"),
        help="Saved trigger classifier head (built by scripts/train_trigger_classifier.py).",
    ),
    lora_path: Path = typer.Option(
        Path("gemma_4_lora/train_v2_80gb/exp_18"),
        help="Path to E18's saved LoRA adapter.",
    ),
    model_name: str = typer.Option("unsloth/gemma-4-E4B-it"),
    data_dir: Path = typer.Option(Path("data")),
    eval_heldout_n: int = typer.Option(1000, help="Number of heldout rows to score."),
    seed: int = typer.Option(42, help="Heldout-sampling seed (must match training)."),
    max_seq_length: int = typer.Option(8192),
    max_completion_length: int = typer.Option(1024),
    batch_size: int = typer.Option(32),
    lora_rank: int = typer.Option(128),
    constrain_facts: bool = typer.Option(
        False,
        "--constrain-facts/--no-constrain-facts",
        help="If set, the two-stage pass injects an explicit allowed-list of "
             "valid tariff names + rate percentages into the prompt. The LLM is "
             "instructed to cite only from this list — directly attacks the "
             "no_halluc plateau.",
    ),
    enforce_slots: bool = typer.Option(
        False,
        "--enforce-slots/--no-enforce-slots",
        help="PR-B: token-level mask via lm-format-enforcer constraining "
             "`tariff_cited` ∈ valid_tariffs and `rate_change_pct_cited` ∈ "
             "valid_pcts. Forces JSON output to populate slots with valid "
             "values; rubric substitutes them into prose. Combine with "
             "--constrain-facts so the model also gets a textual hint of the "
             "allowed-list.",
    ),
    out: Path = typer.Option(
        Path("data/two_stage_eval_v0.json"),
        help="Where to dump the comparison JSON.",
    ),
) -> None:
    """Run the two-stage A/B and dump a comparison JSON."""
    _setup_workspace_env()

    typer.echo(f"loading Stage 1 classifier from {classifier_path}…")
    classifier = TwoStageClassifier.load(classifier_path)
    typer.echo(f"  classifier ready: head_in_dim={classifier.head_in_dim}")

    typer.echo(f"loading {model_name} + LoRA from {lora_path}…")
    model, tokenizer = _load_model(model_name, max_seq_length, lora_rank, lora_path=lora_path)

    _, heldout_ds = _load_dataset(data_dir, heldout_n=eval_heldout_n, seed=seed)
    n = len(heldout_ds)
    typer.echo(f"heldout ready: {n} rows")

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    # Stage 1 predictions on every heldout row
    typer.echo("running Stage 1 classifier on every heldout row…")
    t0 = time.monotonic()
    predicted_triggers: list[list[str]] = [
        classifier.predict_triggers(heldout_ds[i]["input_json"]) for i in range(n)
    ]
    typer.echo(f"  Stage 1 done in {time.monotonic() - t0:.1f}s")

    # Trigger-distribution sanity check
    from collections import Counter
    trigger_counter = Counter()
    for triggers in predicted_triggers:
        for t in triggers:
            trigger_counter[t] += 1
    typer.echo("  Stage 1 prediction distribution:")
    for trigger, count in sorted(trigger_counter.items(), key=lambda x: -x[1]):
        typer.echo(f"    {trigger:42s} {count:>4d} ({count / n:.1%})")
    multi_count = sum(1 for triggers in predicted_triggers if len(triggers) > 1)
    typer.echo(f"  multi-trigger rows: {multi_count}/{n} ({multi_count / n:.1%})")

    # Vanilla E18 generation
    typer.echo("\nrunning Gemma E18 vanilla on the full heldout…")
    t0 = time.monotonic()
    vanilla_completions: list[str] = []
    for start in range(0, n, batch_size):
        chunk = [heldout_ds[i]["prompt"] for i in range(start, min(start + batch_size, n))]
        vanilla_completions.extend(_generate_batch(model, tokenizer, chunk, max_completion_length))
        done = min(start + batch_size, n)
        if done % (batch_size * 4) == 0 or done == n:
            typer.echo(f"  vanilla generation: {done}/{n}")
    typer.echo(f"  vanilla done in {(time.monotonic() - t0) / 60:.1f}min")

    # Two-stage generation: same model, modified prompts
    desc = "Stage-1-injected"
    if constrain_facts:
        desc += " + VALID FACTS prompt-time list"
    if enforce_slots:
        desc += " + LMFE slot mask"
    typer.echo(f"\nrunning Gemma E18 with {desc} prompts on the full heldout…")

    # Build the LMFE tokenizer-data table ONCE — it's vocab-independent of
    # per-row schemas, expensive on Gemma 4's 262k vocab (~20s), and would
    # otherwise rebuild every batch.
    tokenizer_data = None
    if enforce_slots:
        typer.echo("  building LMFE tokenizer-data table (one-time, ~20s)…")
        t_td = time.monotonic()
        tokenizer_data = build_tokenizer_data(tokenizer)
        typer.echo(f"  tokenizer-data ready in {time.monotonic() - t_td:.1f}s")

    t0 = time.monotonic()
    two_stage_completions: list[str] = []
    for start in range(0, n, batch_size):
        chunk_msgs = []
        chunk_facts: list[dict] = []
        for i in range(start, min(start + batch_size, n)):
            base = heldout_ds[i]["prompt"]
            facts = extract_valid_facts(heldout_ds[i]["input_json"])
            chunk_facts.append(facts)
            constrain_arg = facts if constrain_facts else None
            modified = build_two_stage_prompt(base, predicted_triggers[i], valid_facts=constrain_arg)
            chunk_msgs.append(modified)
        prefix_fn = None
        if enforce_slots:
            schemas = [build_slot_enforcement_schema(f) for f in chunk_facts]
            prefix_fn = build_slot_prefix_fn(tokenizer_data, schemas)
        t_batch = time.monotonic()
        two_stage_completions.extend(
            _generate_batch(
                model, tokenizer, chunk_msgs, max_completion_length,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        )
        done = min(start + batch_size, n)
        # Per-batch print so long LMFE-enforced runs are observable
        # mid-flight; lets us catch hangs before n=1000 finishes.
        elapsed = time.monotonic() - t0
        rate = done / elapsed if elapsed else 0
        eta_s = (n - done) / rate if rate else 0
        typer.echo(
            f"  two-stage generation: {done}/{n}  "
            f"(batch took {time.monotonic() - t_batch:.0f}s, "
            f"rate={rate:.2f} rows/s, ETA={eta_s/60:.1f}min)"
        )
    typer.echo(f"  two-stage done in {(time.monotonic() - t0) / 60:.1f}min")

    # Score both
    vanilla_rows: list[dict[str, float]] = []
    two_stage_rows: list[dict[str, float]] = []
    for i in range(n):
        gt = sorted(heldout_ds[i]["ground_truth_triggers"])
        inp = heldout_ds[i]["input_json"]
        vanilla_rows.append(score_completion(vanilla_completions[i], gt, inp))
        two_stage_rows.append(score_completion(two_stage_completions[i], gt, inp))

    agg_vanilla = _aggregate_scores(vanilla_rows)
    agg_two_stage = _aggregate_scores(two_stage_rows)

    typer.echo(f"\n=== heldout (n={n}) — E18 adapter ===")
    typer.echo(f"  vanilla:    {_format_agg(agg_vanilla)}")
    typer.echo(f"  two-stage:  {_format_agg(agg_two_stage)}")

    delta_keys = (
        ("mean_total", "mean_total"),
        ("f1", "f1_triggers_mean"),
        ("no_halluc(prose)", "no_hallucinated_facts_mean"),
        ("no_halluc(slots)", "no_hallucinated_facts_slots_mean"),
        ("well_formed", "well_formed_mean"),
        ("prev_amount", "prev_amount_correct_mean"),
        ("pass_all_pct", "pass_all_pct"),
    )
    typer.echo("\n=== delta (two-stage - vanilla) ===")
    for label, key in delta_keys:
        v = agg_vanilla.get(key, 0.0)
        t = agg_two_stage.get(key, 0.0)
        typer.echo(f"  {label:18s} {v:>8.3f} -> {t:>8.3f}   delta={t - v:+.3f}")

    gpu_monitor.stop()
    gpu_summary = gpu_monitor.summary()
    typer.echo(f"\n=== GPU ===\n  {gpu_monitor.format_summary()}")

    out_payload = {
        "classifier_path": str(classifier_path),
        "lora_path": str(lora_path),
        "n_heldout": n,
        "constrain_facts": constrain_facts,
        "enforce_slots": enforce_slots,
        "stage1_trigger_distribution": dict(trigger_counter),
        "stage1_multi_trigger_rate": multi_count / n if n else 0.0,
        "vanilla": agg_vanilla,
        "two_stage": agg_two_stage,
        "gpu": {
            "mean_util_pct": gpu_summary.mean_util_pct,
            "peak_util_pct": gpu_summary.peak_util_pct,
            "peak_mem_gb": gpu_summary.peak_mem_gb,
            "mem_total_gb": gpu_summary.mem_total_gb,
            "runtime_s": gpu_summary.runtime_s,
            "hints": gpu_summary.hints,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, default=str))
    typer.echo(f"\nwrote {out}")


if __name__ == "__main__":
    app()
