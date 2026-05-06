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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch import GPUMonitor  # noqa: E402

from dd_explainer_rewards import RUBRIC_VERSION, score_completion  # noqa: E402
from dd_explainer_template_renderer import (  # noqa: E402
    backfill_missing_triggers,
    overwrite_explanations,
)
from dd_explainer_two_stage import (  # noqa: E402
    TwoStageClassifier,
    build_two_stage_prompt,
    extract_trigger_grounding,
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
    use_templates: bool = False

    def description(self) -> str:
        if not self.use_stage1:
            return "vanilla (no Stage-1 injection, no constraints)"
        bits = ["Stage-1-injected"]
        if self.constrain_facts:
            bits.append("VALID FACTS prompt list")
        if self.use_lmfe:
            bits.append("LMFE slot mask")
        if self.use_templates:
            bits.append("lonely-trigger templates")
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
    logprobs_top_k: int | None = None,
) -> tuple[list[str], list[list[list[tuple[int, str, float]]]] | None]:
    """Generate completions for one arm. Returns (completions, logprobs|None)."""
    n = len(heldout_ds)
    typer.echo(f"\nrunning arm '{arm.name}': {arm.description()}")
    t0 = time.monotonic()
    completions: list[str] = []
    all_logprobs: list[list[list[tuple[int, str, float]]]] = []
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
                grounding_arg = extract_trigger_grounding(inp) if arm.constrain_facts else None
                msg = build_two_stage_prompt(
                    base, predicted_triggers[i],
                    valid_facts=constrain_arg,
                    trigger_grounding=grounding_arg,
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
        result = _generate_batch(
            model, tokenizer, chunk_msgs, max_completion_length,
            prefix_allowed_tokens_fn=prefix_fn,
            logprobs_top_k=logprobs_top_k,
        )
        if logprobs_top_k is not None:
            chunk_completions, chunk_logprobs = result
            completions.extend(chunk_completions)
            all_logprobs.extend(chunk_logprobs)
        else:
            completions.extend(result)
        elapsed = time.monotonic() - t0
        rate = end / elapsed if elapsed else 0
        eta_s = (n - end) / rate if rate else 0
        typer.echo(
            f"  {arm.name}: {end}/{n}  "
            f"(batch took {time.monotonic() - t_batch:.0f}s, "
            f"rate={rate:.2f} rows/s, ETA={eta_s/60:.1f}min)"
        )
    typer.echo(f"  arm '{arm.name}' done in {(time.monotonic() - t0) / 60:.1f}min")
    return completions, (all_logprobs if logprobs_top_k is not None else None)


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


def _per_trigger_leak_summary(
    per_row_dicts: list[dict[str, Any]],
    arm_name: str,
    *,
    bucket_key: str = "ground_truth_triggers",
) -> list[dict[str, Any]]:
    """Bucket per-row scores by trigger, return a list of {trigger, n, fail_pct, no_halluc_mean}.

    `arm_name` selects the arm's scores from each row (looks under r["scores"][arm]
    for new-format rows or directly under r[arm] for legacy).
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in per_row_dicts:
        scores = r.get("scores", {}).get(arm_name) or r.get(arm_name) or {}
        if not scores:
            continue
        for t in r.get(bucket_key) or []:
            buckets[t].append(scores)
    summary: list[dict[str, Any]] = []
    for t, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if not rows:
            continue
        nh_vals = [s.get("no_hallucinated_facts", 0.0) for s in rows]
        n_fail = sum(1 for v in nh_vals if v <= -2.5)
        summary.append({
            "trigger": t,
            "n": len(rows),
            "fail_pct": 100.0 * n_fail / len(rows),
            "no_halluc_mean": mean(nh_vals) if nh_vals else 0.0,
        })
    return summary


def _log_to_wandb(
    *,
    wandb_project: str,
    wandb_run_name: Optional[str],
    wandb_notes: Optional[str],
    wandb_tags: list[str],
    wandb_config: dict[str, Any],
    arm_aggregates: dict[str, dict],
    per_row_dicts: Optional[list[dict[str, Any]]] = None,
    arm_names: Optional[list[str]] = None,
) -> None:
    """Push aggregate + per-row + per-trigger leak summary to a W&B run.

    Lazy import keeps wandb optional — the eval script runs fine without it.
    """
    import wandb  # noqa: PLC0415

    tags = sorted({RUBRIC_VERSION, "eval", *wandb_tags})
    run = wandb.init(
        project=wandb_project,
        entity="chaleong",
        name=wandb_run_name,
        notes=wandb_notes or "",
        tags=tags,
        config=wandb_config,
        reinit=True,
    )
    try:
        for arm_name, agg in arm_aggregates.items():
            for k, v in agg.items():
                if isinstance(v, (int, float)):
                    wandb.run.summary[f"eval/{arm_name}/{k}"] = v

        if per_row_dicts is None or not per_row_dicts:
            return
        if arm_names is None:
            arm_names = sorted(arm_aggregates.keys())

        # 1. Per-row Table (filterable / sortable in the W&B UI)
        cols = ["i", "gt_triggers", "stage1_triggers"]
        for a in arm_names:
            cols += [f"{a}_no_halluc", f"{a}_f1", f"{a}_well_formed", f"{a}_total"]
        rows: list[list] = []
        for r in per_row_dicts:
            row: list = [
                r.get("i"),
                ", ".join(r.get("ground_truth_triggers") or []),
                ", ".join(r.get("stage1_triggers") or []),
            ]
            for a in arm_names:
                s = r.get("scores", {}).get(a) or r.get(a) or {}
                row += [
                    s.get("no_hallucinated_facts"),
                    s.get("f1_triggers"),
                    s.get("well_formed"),
                    sum(v for v in s.values() if isinstance(v, (int, float))),
                ]
            rows.append(row)
        wandb.log({"eval/per_row": wandb.Table(columns=cols, data=rows)})

        # 2. Per-trigger leak summary tables (one per arm, both bucket keys)
        for a in arm_names:
            for bucket_key, label in (
                ("ground_truth_triggers", "by_gt"),
                ("stage1_triggers", "by_stage1"),
            ):
                summary = _per_trigger_leak_summary(
                    per_row_dicts, a, bucket_key=bucket_key,
                )
                if not summary:
                    continue
                cols = ["trigger", "n", "fail_pct", "no_halluc_mean"]
                data = [[s["trigger"], s["n"], s["fail_pct"], s["no_halluc_mean"]] for s in summary]
                wandb.log({
                    f"eval/per_trigger/{a}/{label}": wandb.Table(columns=cols, data=data),
                })
    finally:
        run.finish()


def _rescore_from_per_row(
    per_row_path: Path,
    out: Path,
    *,
    use_templates: bool = False,
    backfill_triggers: bool = False,
    stage1_probs_path: Optional[Path] = None,
    backfill_threshold: float = 0.9,
    wandb_log: bool = False,
    wandb_project: str = "gemma4-rlvr-eval",
    wandb_run_name: Optional[str] = None,
    wandb_notes: Optional[str] = None,
) -> None:
    """Re-score cached completions from a previous run under the current rubric.

    If `use_templates` is set, post-processes each completion through
    `overwrite_explanations` (lonely-trigger templates) before scoring.

    If `backfill_triggers` is set, additionally appends a templated entry
    for every Stage-1-predicted trigger missing from the LLM's output.
    Gated on classifier confidence when `stage1_probs_path` is provided
    (recommended for production); ungated when omitted (rescore-only mode
    that trusts a 100%-exact-match Stage-1).
    """
    rows = [json.loads(l) for l in per_row_path.read_text().splitlines() if l.strip()]
    if not rows:
        raise typer.BadParameter(f"{per_row_path} is empty")
    if "completions" not in rows[0]:
        raise typer.BadParameter(
            f"{per_row_path} predates --dump-per-row v2; no `completions` field. "
            "Re-generate with the current script to enable rescore."
        )
    arm_names = sorted(rows[0]["completions"].keys())

    stage1_probs_by_row: dict[str, dict[str, float]] | None = None
    if backfill_triggers and stage1_probs_path is not None:
        probs_payload = json.loads(stage1_probs_path.read_text())
        stage1_probs_by_row = probs_payload.get("probabilities", probs_payload)
        typer.echo(
            f"loaded Stage-1 probs for {len(stage1_probs_by_row)} rows "
            f"from {stage1_probs_path.name}"
        )

    typer.echo(
        f"rescore mode: {len(rows)} rows × {len(arm_names)} arms ({arm_names}) "
        f"under rubric {RUBRIC_VERSION}"
        + (" + lonely-trigger templates" if use_templates else "")
        + (
            f" + backfill (gate>={backfill_threshold})"
            if backfill_triggers and stage1_probs_by_row is not None
            else " + backfill (UNGATED)" if backfill_triggers else ""
        )
    )
    arm_aggregates: dict[str, dict] = {}
    rescored_rows: list[dict[str, Any]] = []
    n_overwritten = 0
    n_backfilled = 0
    n_parse_failed = 0
    for i, r in enumerate(rows):
        new_scores: dict[str, dict[str, float]] = {}
        gt = sorted(r["ground_truth_triggers"])
        inp = r["input_json"]
        stage1_triggers = r.get("stage1_triggers") or []
        row_probs: dict[str, float] | None = None
        if stage1_probs_by_row is not None:
            row_probs = stage1_probs_by_row.get(str(r["i"]))
        for arm in arm_names:
            comp = r["completions"][arm]
            if (use_templates or backfill_triggers) and arm != "vanilla":
                grounding = extract_trigger_grounding(inp)
                facts = extract_valid_facts(inp)
                try:
                    parsed = json.loads(comp)
                    before = json.dumps(parsed, sort_keys=True)
                    if use_templates:
                        parsed = overwrite_explanations(parsed, grounding, facts)
                    after_overwrite = json.dumps(parsed, sort_keys=True)
                    if before != after_overwrite:
                        n_overwritten += 1
                    if backfill_triggers:
                        n_before = len(parsed.get("explanations") or [])
                        parsed = backfill_missing_triggers(
                            parsed,
                            stage1_triggers=stage1_triggers,
                            stage1_probs=row_probs,
                            grounding=grounding,
                            valid_facts=facts,
                            confidence_threshold=backfill_threshold,
                        )
                        if len(parsed.get("explanations") or []) > n_before:
                            n_backfilled += 1
                    comp = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError):
                    n_parse_failed += 1
            new_scores[arm] = score_completion(comp, gt, inp)
        rescored_rows.append({**r, "scores": new_scores})
    if use_templates:
        typer.echo(
            f"  templates: overwrote {n_overwritten} arm-rows; "
            f"{n_parse_failed} json-parse failures"
        )
    if backfill_triggers:
        typer.echo(f"  backfill: appended trigger entries on {n_backfilled} arm-rows")
    for arm in arm_names:
        arm_aggregates[arm] = _aggregate_scores([r["scores"][arm] for r in rescored_rows])

    n = len(rows)
    payload = _build_payload(
        arm_aggregates=arm_aggregates,
        n=n,
        heldout_meta={
            "rescored_from": str(per_row_path),
            "use_templates": use_templates,
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    typer.echo(f"\nwrote {out}")
    _print_deltas(
        arm_aggregates, n,
        baseline_arm=arm_names[0],
        target_arm=arm_names[-1],
    )

    if wandb_log:
        _log_to_wandb(
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_notes=wandb_notes,
            wandb_tags=["rescore"],
            wandb_config={
                "rescored_from": str(per_row_path),
                "n_heldout": n,
                "rubric_version": RUBRIC_VERSION,
                "arm_names": arm_names,
            },
            arm_aggregates=arm_aggregates,
            per_row_dicts=rescored_rows,
            arm_names=arm_names,
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
    use_templates: bool = typer.Option(
        False,
        "--use-templates/--no-use-templates",
        help="Two-stage arm: post-generation, overwrite header+explanation for the "
             "4 lonely triggers (First DD review / Missed-bounced / Manual reduction "
             "/ Exemption Expiry) with deterministic templates rendered from "
             "extract_trigger_grounding values. Slot fields untouched.",
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
    backfill_triggers: bool = typer.Option(
        False,
        "--backfill-triggers/--no-backfill-triggers",
        help="Append a templated explanation entry for every Stage-1-predicted "
             "trigger missing from the LLM's output. Closes the residual "
             "f1<10 failures where Stage-1 was right but the LLM dropped a "
             "trigger. Gated on classifier confidence when --stage1-probs-path "
             "is provided.",
    ),
    stage1_probs_path: Optional[Path] = typer.Option(
        None,
        "--stage1-probs-path",
        help="Path to a JSON file mapping row-index -> {trigger: prob} from "
             "the Stage-1 classifier. When set with --backfill-triggers, only "
             "backfills triggers whose Stage-1 prob >= --backfill-threshold. "
             "Strongly recommended for production to guard against Stage-1 "
             "false-positives being forced into the output.",
    ),
    backfill_threshold: float = typer.Option(
        0.9,
        "--backfill-threshold",
        help="Stage-1 classifier prob threshold for backfilling a trigger. "
             "Only used when --stage1-probs-path is provided.",
    ),
    save_logprobs: int = typer.Option(
        0,
        "--save-logprobs",
        help="If >0, capture top-K logprobs per generated token under the actual "
             "(LMFE-constrained) sampling distribution and save them in "
             "<out>.per_row.jsonl alongside completions. ~5-10%% slower; modest "
             "extra disk (~3MB/1000 rows at K=5). Required for confidence-based "
             "diagnostics like scripts/token_confidence_summary.py.",
    ),
    wandb_log: bool = typer.Option(
        False,
        "--wandb-log/--no-wandb-log",
        help="Push aggregate metrics + per-row + per-trigger leak Tables to W&B "
             "for filtering/scrubbing. Lazy import — wandb is optional.",
    ),
    wandb_project: str = typer.Option(
        "gemma4-rlvr-eval",
        help="W&B project for eval runs (separate from the training project).",
    ),
    wandb_run_name: Optional[str] = typer.Option(
        None,
        help="W&B run name. Default: auto-generated from lora_path basename + timestamp.",
    ),
    wandb_notes: Optional[str] = typer.Option(
        None,
        help="Free-text notes attached to the W&B run.",
    ),
    arms: str = typer.Option(
        "all",
        help="Comma-separated arm names to run (e.g. 'vanilla', 'two_stage', "
             "or 'all' for both). Allows splitting arms across parallel jobs.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume from per-arm checkpoint files (<out>.ckpt.<arm>.jsonl). "
             "Skips generation for arms whose checkpoint exists; still scores "
             "under the current rubric.",
    ),
) -> None:
    """Generate + score (default), or re-score cached completions (--rescore-from)."""
    if rescore_from is not None:
        _rescore_from_per_row(
            rescore_from, out,
            use_templates=use_templates,
            backfill_triggers=backfill_triggers,
            stage1_probs_path=stage1_probs_path,
            backfill_threshold=backfill_threshold,
            wandb_log=wandb_log,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_notes=wandb_notes,
        )
        return

    _setup_workspace_env()

    typer.echo(f"loading Stage 1 classifier from {classifier_path}…")
    classifier = TwoStageClassifier.load(classifier_path)
    typer.echo(f"  classifier ready: head_in_dim={classifier.head_in_dim}")

    # Model + tokenizer loaded lazily -- only when an arm needs generation
    model = tokenizer = None
    _model_loaded = False

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
    all_arms = [
        EvalArm("vanilla", use_stage1=False, constrain_facts=False, use_lmfe=False),
        EvalArm(
            "two_stage",
            use_stage1=True,
            constrain_facts=constrain_facts,
            use_lmfe=enforce_slots,
            use_templates=use_templates,
        ),
    ]

    # --arms filtering
    if arms.lower() == "all":
        selected_arms = all_arms
    else:
        requested = {a.strip() for a in arms.split(",")}
        valid_names = {a.name for a in all_arms}
        unknown = requested - valid_names
        if unknown:
            raise typer.BadParameter(
                f"Unknown arm(s): {unknown}. Valid: {sorted(valid_names)}"
            )
        selected_arms = [a for a in all_arms if a.name in requested]

    arm_completions: dict[str, list[str]] = {}
    arm_rows: dict[str, list[dict[str, float]]] = {}
    arm_logprobs_by_arm: dict[str, list[list[list[tuple[int, str, float]]]] | None] = {}
    arm_aggregates: dict[str, dict] = {}
    for arm in selected_arms:
        ckpt_path = out.with_suffix(f".ckpt.{arm.name}.jsonl")

        # --resume: load cached completions if checkpoint exists
        arm_logprobs = None
        if resume and ckpt_path.exists():
            typer.echo(f"\nresuming arm '{arm.name}' from {ckpt_path}")
            ckpt_records = [
                json.loads(line)
                for line in ckpt_path.read_text().splitlines()
                if line.strip()
            ]
            completions = [r["completion"] for r in ckpt_records]
            if any("logprobs" in r for r in ckpt_records):
                arm_logprobs = [r.get("logprobs") for r in ckpt_records]
            if len(completions) != n:
                typer.echo(
                    f"  WARNING: checkpoint has {len(completions)} rows, "
                    f"expected {n} -- regenerating"
                )
                completions = None
                arm_logprobs = None
            else:
                typer.echo(f"  loaded {len(completions)} completions (skipping generation)")
        else:
            completions = None

        if completions is None:
            # Lazy model load on first arm that needs generation
            if not _model_loaded:
                typer.echo(f"loading {model_name} + LoRA from {lora_path}…")
                model, tokenizer = _load_model(model_name, max_seq_length, lora_rank, lora_path=lora_path)
                if enforce_slots:
                    typer.echo("  building LMFE tokenizer-data table (one-time, ~20s)…")
                    t_td = time.monotonic()
                    tokenizer_data = build_tokenizer_data(tokenizer)
                    typer.echo(f"  tokenizer-data ready in {time.monotonic() - t_td:.1f}s")
                _model_loaded = True
            completions, arm_logprobs = _run_arm(
                arm,
                model=model,
                tokenizer=tokenizer,
                heldout_ds=heldout_ds,
                predicted_triggers=predicted_triggers,
                tokenizer_data=tokenizer_data,
                max_completion_length=max_completion_length,
                batch_size=batch_size,
                logprobs_top_k=save_logprobs if save_logprobs > 0 else None,
            )
            # Checkpoint: save completions immediately after generation
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with ckpt_path.open("w") as fh:
                for i, comp in enumerate(completions):
                    rec = {"i": i, "completion": comp}
                    if arm_logprobs is not None:
                        rec["logprobs"] = arm_logprobs[i]
                    fh.write(json.dumps(rec) + "\n")
            typer.echo(f"  checkpointed {len(completions)} completions to {ckpt_path}")

        # Lonely-trigger templates: post-generation, parse the LLM JSON,
        # overwrite header+explanation for the 4 lonely triggers with
        # deterministic templates rendered from extract_trigger_grounding
        # values. Slot fields untouched. Raw completions stay in the
        # checkpoint above so --rescore-from can A/B with/without templates.
        scoring_completions = completions
        if arm.use_templates:
            n_overwritten = 0
            n_parse_failed = 0
            templated: list[str] = []
            for i, comp in enumerate(completions):
                inp = heldout_ds[i]["input_json"]
                grounding = extract_trigger_grounding(inp)
                facts = extract_valid_facts(inp)
                try:
                    parsed = json.loads(comp)
                except (json.JSONDecodeError, TypeError):
                    n_parse_failed += 1
                    templated.append(comp)
                    continue
                before = json.dumps(parsed, sort_keys=True)
                parsed = overwrite_explanations(parsed, grounding, facts)
                after = json.dumps(parsed, sort_keys=True)
                if before != after:
                    n_overwritten += 1
                templated.append(json.dumps(parsed, indent=2))
            scoring_completions = templated
            typer.echo(
                f"  templates: overwrote {n_overwritten}/{len(completions)} rows; "
                f"{n_parse_failed} json-parse failures (left as-is)"
            )

        arm_completions[arm.name] = scoring_completions
        arm_logprobs_by_arm[arm.name] = arm_logprobs
        arm_rows[arm.name] = _score_completions(scoring_completions, heldout_ds)
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
            "use_templates": use_templates,
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
    if "vanilla" in arm_aggregates and "two_stage" in arm_aggregates:
        _print_deltas(arm_aggregates, n, baseline_arm="vanilla", target_arm="two_stage")
    else:
        for name, agg in arm_aggregates.items():
            typer.echo(f"\n=== {name} (n={n}) ===\n  {_format_agg(agg)}")

    per_row_dicts: list[dict[str, Any]] = []
    if dump_per_row or wandb_log:
        for i in range(n):
            row = {
                "i": i,
                "ground_truth_triggers": sorted(heldout_ds[i]["ground_truth_triggers"]),
                "stage1_triggers": sorted(predicted_triggers[i]),
                "input_json": heldout_ds[i]["input_json"],
                "completions": {arm: arm_completions[arm][i] for arm in arm_completions},
                "scores": {arm: arm_rows[arm][i] for arm in arm_rows},
                # Back-compat keys for older readers (per_trigger_leak.py etc).
                "vanilla": arm_rows.get("vanilla", [{}])[i] if "vanilla" in arm_rows else {},
                "two_stage": arm_rows.get("two_stage", [{}])[i] if "two_stage" in arm_rows else {},
            }
            arm_logprobs_present = {
                a: lp for a, lp in arm_logprobs_by_arm.items() if lp is not None
            }
            if arm_logprobs_present:
                row["logprobs"] = {a: lp[i] for a, lp in arm_logprobs_present.items()}
            per_row_dicts.append(row)
    if dump_per_row:
        per_row_path = out.with_suffix(".per_row.jsonl")
        with per_row_path.open("w") as fh:
            for r in per_row_dicts:
                fh.write(json.dumps(r, default=str) + "\n")
        typer.echo(f"wrote per-row {per_row_path}")

    if wandb_log:
        _log_to_wandb(
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name or f"eval_{lora_path.name}_{int(time.time())}",
            wandb_notes=wandb_notes,
            wandb_tags=["generate", lora_path.name],
            wandb_config={
                "lora_path": str(lora_path),
                "classifier_path": str(classifier_path),
                "n_heldout": n,
                "constrain_facts": constrain_facts,
                "enforce_slots": enforce_slots,
                "rubric_version": RUBRIC_VERSION,
                "arm_names": [arm.name for arm in arms],
            },
            arm_aggregates=arm_aggregates,
            per_row_dicts=per_row_dicts,
            arm_names=[arm.name for arm in selected_arms],
        )


if __name__ == "__main__":
    app()
