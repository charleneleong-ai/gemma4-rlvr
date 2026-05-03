"""Eval harness for the dd_explainer task — A/B (or N-way) on a held-out split.

Each `EvalArm` is one decoding configuration:

  use_stage1       inject Stage-1 classifier triggers into the prompt
  constrain_facts  inject the prompt-time allowed-facts list (PR #12)
  use_lmfe         wrap generation with the LMFE token-mask (PR-B)

Default arms (today's behaviour preserved):

  vanilla   = EvalArm("vanilla",   use_stage1=False, constrain_facts=False, use_lmfe=False)
  two_stage = EvalArm("two_stage", use_stage1=True,  constrain_facts=cf,    use_lmfe=es)

Two modes:

  GENERATE (default): load the model, run each arm, score, write an aggregate
                      JSON. With --dump-per-row also writes <out>.per_row.jsonl
                      with completions + scores so a future rubric change can
                      re-score without regenerating.

  RESCORE  (--rescore-from <per_row.jsonl>): skip the model entirely, read the
                                              cached completions, re-apply the
                                              current rubric, write a fresh
                                              aggregate JSON in ~10s.

Heldout sampling is seed-stable (matches the original training split).

Usage:

    # Generate + score (~30min for n=200 with LMFE, ~2.5h for n=1000)
    uv run python scripts/two_stage_eval.py \\
      --lora-path gemma_4_lora/train_v2_80gb/exp_27 \\
      --eval-heldout-n 200 \\
      --constrain-facts --enforce-slots --dump-per-row \\
      --out data/eval_e27_n200.json

    # Re-score that run under a new rubric (~10s, no GPU)
    uv run python scripts/two_stage_eval.py \\
      --rescore-from data/eval_e27_n200.per_row.jsonl \\
      --out data/eval_e27_n200_v2rubric.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch import GPUMonitor  # noqa: E402

from dd_explainer_rewards import RUBRIC_VERSION, score_completion  # noqa: E402
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


@dataclass(frozen=True)
class EvalArm:
    name: str
    use_stage1: bool
    constrain_facts: bool
    use_lmfe: bool

    def description(self) -> str:
        if not self.use_stage1:
            return "vanilla (no Stage-1 injection, no constraints)"
        bits = ["Stage-1-injected"]
        if self.constrain_facts:
            bits.append("VALID FACTS prompt list")
        if self.use_lmfe:
            bits.append("LMFE slot mask")
        return " + ".join(bits)


def _format_agg(agg: dict[str, Any]) -> str:
    return (
        f"mean_total={agg.get('mean_total'):.3f}  "
        f"f1={agg.get('f1_triggers_mean'):.3f}  "
        f"no_halluc={agg.get('no_hallucinated_facts_mean'):.3f}  "
        f"pass_all={agg.get('pass_all_pct')}%"
    )


def _run_arm(
    arm: EvalArm,
    *,
    model,
    tokenizer,
    heldout_ds,
    predicted_triggers: list[list[str]],
    tokenizer_data,
    max_completion_length: int,
    batch_size: int,
) -> list[str]:
    """Generate completions for one arm. Returns list of completion strings (length n)."""
    n = len(heldout_ds)
    typer.echo(f"\nrunning arm '{arm.name}': {arm.description()}")
    t0 = time.monotonic()
    completions: list[str] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_msgs = []
        chunk_facts: list[dict] = []
        for i in range(start, end):
            base = heldout_ds[i]["prompt"]
            inp = heldout_ds[i]["input_json"]
            if arm.use_stage1:
                facts = extract_valid_facts(inp)
                chunk_facts.append(facts)
                constrain_arg = facts if arm.constrain_facts else None
                msg = build_two_stage_prompt(
                    base, predicted_triggers[i], valid_facts=constrain_arg,
                )
            else:
                msg = base
                chunk_facts.append({})
            chunk_msgs.append(msg)
        prefix_fn = None
        if arm.use_lmfe:
            schemas = [build_slot_enforcement_schema(f) for f in chunk_facts]
            prefix_fn = build_slot_prefix_fn(tokenizer_data, schemas)
        t_batch = time.monotonic()
        completions.extend(
            _generate_batch(
                model, tokenizer, chunk_msgs, max_completion_length,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        )
        elapsed = time.monotonic() - t0
        rate = end / elapsed if elapsed else 0
        eta_s = (n - end) / rate if rate else 0
        typer.echo(
            f"  {arm.name}: {end}/{n}  "
            f"(batch took {time.monotonic() - t_batch:.0f}s, "
            f"rate={rate:.2f} rows/s, ETA={eta_s/60:.1f}min)"
        )
    typer.echo(f"  arm '{arm.name}' done in {(time.monotonic() - t0) / 60:.1f}min")
    return completions


def _score_completions(
    completions: list[str],
    heldout_ds,
) -> list[dict[str, float]]:
    """Apply score_completion across all rows."""
    rows: list[dict[str, float]] = []
    for i, c in enumerate(completions):
        gt = sorted(heldout_ds[i]["ground_truth_triggers"])
        inp = heldout_ds[i]["input_json"]
        rows.append(score_completion(c, gt, inp))
    return rows


def _build_payload(
    *,
    arm_aggregates: dict[str, dict],
    n: int,
    heldout_meta: dict[str, Any],
    gpu_summary: Optional[dict] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rubric_version": RUBRIC_VERSION,
        "n_heldout": n,
        **heldout_meta,
        # Back-compat top-level keys ("vanilla", "two_stage" expected by callers)
        **arm_aggregates,
    }
    if gpu_summary is not None:
        payload["gpu"] = gpu_summary
    return payload


def _print_deltas(
    arm_aggregates: dict[str, dict],
    n: int,
    *,
    baseline_arm: str,
    target_arm: str,
) -> None:
    typer.echo(f"\n=== heldout (n={n}) ===")
    for name, agg in arm_aggregates.items():
        typer.echo(f"  {name:12s}: {_format_agg(agg)}")
    if baseline_arm in arm_aggregates and target_arm in arm_aggregates:
        delta_keys = (
            ("mean_total", "mean_total"),
            ("f1", "f1_triggers_mean"),
            ("no_halluc(prose)", "no_hallucinated_facts_mean"),
            ("no_halluc(slots)", "no_hallucinated_facts_slots_mean"),
            ("well_formed", "well_formed_mean"),
            ("prev_amount", "prev_amount_correct_mean"),
            ("pass_all_pct", "pass_all_pct"),
        )
        v = arm_aggregates[baseline_arm]
        t = arm_aggregates[target_arm]
        typer.echo(f"\n=== delta ({target_arm} - {baseline_arm}) ===")
        for label, key in delta_keys:
            vv = v.get(key, 0.0)
            tt = t.get(key, 0.0)
            typer.echo(f"  {label:18s} {vv:>8.3f} -> {tt:>8.3f}   delta={tt - vv:+.3f}")


def _rescore_from_per_row(
    per_row_path: Path,
    out: Path,
) -> None:
    """Re-score cached completions from a previous run under the current rubric."""
    rows = [json.loads(l) for l in per_row_path.read_text().splitlines() if l.strip()]
    if not rows:
        raise typer.BadParameter(f"{per_row_path} is empty")
    if "completions" not in rows[0]:
        raise typer.BadParameter(
            f"{per_row_path} predates --dump-per-row v2; no `completions` field. "
            "Re-generate with the current script to enable rescore."
        )
    arm_names = sorted(rows[0]["completions"].keys())
    typer.echo(
        f"rescore mode: {len(rows)} rows × {len(arm_names)} arms ({arm_names}) "
        f"under rubric {RUBRIC_VERSION}"
    )
    arm_aggregates: dict[str, dict] = {}
    rescored_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        new_scores: dict[str, dict[str, float]] = {}
        gt = sorted(r["ground_truth_triggers"])
        inp = r["input_json"]
        for arm in arm_names:
            new_scores[arm] = score_completion(r["completions"][arm], gt, inp)
        rescored_rows.append({**r, "scores": new_scores})
    for arm in arm_names:
        arm_aggregates[arm] = _aggregate_scores([r["scores"][arm] for r in rescored_rows])

    n = len(rows)
    payload = _build_payload(
        arm_aggregates=arm_aggregates,
        n=n,
        heldout_meta={"rescored_from": str(per_row_path)},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    typer.echo(f"\nwrote {out}")
    _print_deltas(
        arm_aggregates, n,
        baseline_arm=arm_names[0],
        target_arm=arm_names[-1],
    )


@app.command()
def main(
    classifier_path: Path = typer.Option(
        Path("data/trigger_classifier_v3_extra_features.pt"),
        help="Saved trigger classifier head (built by scripts/train_trigger_classifier.py).",
    ),
    lora_path: Path = typer.Option(
        Path("gemma_4_lora/train_v2_80gb/exp_18"),
        help="Path to a saved Gemma LoRA adapter.",
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
        help="Two-stage arm: inject the allowed-facts list into the prompt (PR #12).",
    ),
    enforce_slots: bool = typer.Option(
        False,
        "--enforce-slots/--no-enforce-slots",
        help="Two-stage arm: token-mask via lm-format-enforcer on `tariff_cited` / "
             "`rate_change_pct_cited` slots (PR-B).",
    ),
    out: Path = typer.Option(
        Path("data/two_stage_eval_v0.json"),
        help="Where to dump the aggregate JSON.",
    ),
    dump_per_row: bool = typer.Option(
        False,
        "--dump-per-row/--no-dump-per-row",
        help="Also dump per-row completions + scores to <out>.per_row.jsonl. "
             "Required to use --rescore-from later.",
    ),
    rescore_from: Optional[Path] = typer.Option(
        None,
        "--rescore-from",
        help="If set, skip the model and re-score completions from a previous "
             "<out>.per_row.jsonl under the current rubric. Cheap (~10s).",
    ),
) -> None:
    """Generate + score (default), or re-score cached completions (--rescore-from)."""
    if rescore_from is not None:
        _rescore_from_per_row(rescore_from, out)
        return

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

    typer.echo("running Stage 1 classifier on every heldout row…")
    t0 = time.monotonic()
    predicted_triggers: list[list[str]] = [
        classifier.predict_triggers(heldout_ds[i]["input_json"]) for i in range(n)
    ]
    typer.echo(f"  Stage 1 done in {time.monotonic() - t0:.1f}s")

    trigger_counter: Counter[str] = Counter()
    for triggers in predicted_triggers:
        for t in triggers:
            trigger_counter[t] += 1
    typer.echo("  Stage 1 prediction distribution:")
    for trigger, count in sorted(trigger_counter.items(), key=lambda x: -x[1]):
        typer.echo(f"    {trigger:42s} {count:>4d} ({count / n:.1%})")
    multi_count = sum(1 for triggers in predicted_triggers if len(triggers) > 1)
    typer.echo(f"  multi-trigger rows: {multi_count}/{n} ({multi_count / n:.1%})")

    tokenizer_data = None
    if enforce_slots:
        typer.echo("  building LMFE tokenizer-data table (one-time, ~20s)…")
        t_td = time.monotonic()
        tokenizer_data = build_tokenizer_data(tokenizer)
        typer.echo(f"  tokenizer-data ready in {time.monotonic() - t_td:.1f}s")

    arms = [
        EvalArm("vanilla", use_stage1=False, constrain_facts=False, use_lmfe=False),
        EvalArm(
            "two_stage",
            use_stage1=True,
            constrain_facts=constrain_facts,
            use_lmfe=enforce_slots,
        ),
    ]

    arm_completions: dict[str, list[str]] = {}
    arm_rows: dict[str, list[dict[str, float]]] = {}
    arm_aggregates: dict[str, dict] = {}
    for arm in arms:
        completions = _run_arm(
            arm,
            model=model,
            tokenizer=tokenizer,
            heldout_ds=heldout_ds,
            predicted_triggers=predicted_triggers,
            tokenizer_data=tokenizer_data,
            max_completion_length=max_completion_length,
            batch_size=batch_size,
        )
        arm_completions[arm.name] = completions
        arm_rows[arm.name] = _score_completions(completions, heldout_ds)
        arm_aggregates[arm.name] = _aggregate_scores(arm_rows[arm.name])

    gpu_monitor.stop()
    gpu_summary = gpu_monitor.summary()
    typer.echo(f"\n=== GPU ===\n  {gpu_monitor.format_summary()}")

    payload = _build_payload(
        arm_aggregates=arm_aggregates,
        n=n,
        heldout_meta={
            "classifier_path": str(classifier_path),
            "lora_path": str(lora_path),
            "constrain_facts": constrain_facts,
            "enforce_slots": enforce_slots,
            "stage1_trigger_distribution": dict(trigger_counter),
            "stage1_multi_trigger_rate": multi_count / n if n else 0.0,
        },
        gpu_summary={
            "mean_util_pct": gpu_summary.mean_util_pct,
            "peak_util_pct": gpu_summary.peak_util_pct,
            "peak_mem_gb": gpu_summary.peak_mem_gb,
            "mem_total_gb": gpu_summary.mem_total_gb,
            "runtime_s": gpu_summary.runtime_s,
            "hints": gpu_summary.hints,
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    typer.echo(f"\nwrote {out}")
    _print_deltas(arm_aggregates, n, baseline_arm="vanilla", target_arm="two_stage")

    if dump_per_row:
        per_row_path = out.with_suffix(".per_row.jsonl")
        with per_row_path.open("w") as fh:
            for i in range(n):
                fh.write(json.dumps({
                    "i": i,
                    "ground_truth_triggers": sorted(heldout_ds[i]["ground_truth_triggers"]),
                    "stage1_triggers": sorted(predicted_triggers[i]),
                    "input_json": heldout_ds[i]["input_json"],
                    "completions": {arm: arm_completions[arm][i] for arm in arm_completions},
                    "scores": {arm: arm_rows[arm][i] for arm in arm_rows},
                    # Back-compat keys for older readers (per_trigger_leak.py etc).
                    "vanilla": arm_rows.get("vanilla", [{}])[i] if "vanilla" in arm_rows else {},
                    "two_stage": arm_rows.get("two_stage", [{}])[i] if "two_stage" in arm_rows else {},
                }, default=str) + "\n")
        typer.echo(f"wrote per-row {per_row_path}")


if __name__ == "__main__":
    app()
