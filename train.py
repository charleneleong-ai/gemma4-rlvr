"""CLI for the direct_debit_explainer RLVR mock.

Subcommands:
  train    — GRPO-tune Gemma 4 against the 7 verifiable rewards.
  infer    — generate a synthetic DDExplainerPromptInput and roll out the (optionally tuned) model on it.
  regress  — re-score the model on real LangSmith failed traces.
"""

from __future__ import annotations

import json
import os
import random
import signal
from collections import deque
from pathlib import Path
from typing import Optional


# =============================================================================
# Env setup — must run BEFORE unsloth / transformers import.
# =============================================================================


def _setup_workspace_env(workspace: Path = Path("/workspace/gemma4_rl")) -> None:
    """Redirect heavy caches to the NVMe volume so model downloads/compile caches
    don't fill the overlay root. Mirrors cell 5 of the notebook.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    for sub in ("hf_cache", "torch_cache", "outputs"):
        (workspace / sub).mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(workspace / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(workspace / "hf_cache/datasets"))
    os.environ.setdefault("TORCH_HOME", str(workspace / "torch_cache"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(workspace / "torch_cache/inductor"))
    # Reduce CUDA allocator fragmentation during long-context GRPO rollouts.
    # `expandable_segments` lets the caching allocator grow/shrink mappings
    # in-place, which is crucial when completion length spikes blow up the
    # peak working set on a 40 GB A100.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Avoid HF tokenizers spawning extra threads (which double host-RAM
    # footprint when the dataloader forks).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Apply env + import unsloth at module load so its patches land BEFORE any
# transformers / trl / peft import anywhere in the process. `load_dotenv` runs
# first so WANDB_API_KEY / HF_TOKEN are in place before anything imports them.
_setup_workspace_env()
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# Heavy imports: unsloth first (patching), then torch / transformers / trl /
# datasets, then our own modules. Moving these to module top raises `--help`
# latency to ~15 s but matches the "all imports at top" convention and keeps
# every subcommand import-free at runtime.
import unsloth  # noqa: E402, F401
import typer  # noqa: E402
import torch  # noqa: E402, F401
import pandas as pd  # noqa: E402
import wandb  # noqa: E402
from datasets import Dataset  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import TextStreamer, TrainerCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from unsloth import FastModel  # noqa: E402

from config import Settings, load_hydra_settings  # noqa: E402
from dd_explainer_data_generator import (  # noqa: E402
    DDExplainerPromptInput,
    Trigger,
    build_chat_messages,
    detect_triggers,
    generate_dd_example,
)
from dd_explainer_rewards import (  # noqa: E402
    REWARD_FUNCS,
    parse_response,
    score_completion,
)
import time  # noqa: E402
from experiments.experiment_progress import log_experiment, plot_progress  # noqa: E402

app = typer.Typer(
    help="RLVR / GRPO training for the direct_debit_explainer mock.",
    add_completion=False,
)


# =============================================================================
# Early-stopping callback (GRPO has no eval set — monitor training reward MA)
# =============================================================================


class RewardPlateauCallback(TrainerCallback):
    """Stop training if the moving-average training `reward` has not improved by
    `min_delta` for `patience` consecutive logging events.

    GRPO is on-policy, so there is no eval set to monitor — the natural plateau
    signal is the training reward scalar. Per-step reward is noisy, so we smooth
    over a window of `window` logs before comparing.
    """

    def __init__(self, patience: int, window: int = 10, min_delta: float = 0.05):
        self.patience = patience
        self.window = window
        self.min_delta = min_delta
        self.recent: "deque[float]" = deque(maxlen=window)
        self.best_ma = float("-inf")
        self.stale_logs = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "reward" not in logs:
            return
        try:
            r = float(logs["reward"])
        except (TypeError, ValueError):
            return
        self.recent.append(r)
        if len(self.recent) < self.window:
            return
        ma = sum(self.recent) / len(self.recent)
        if ma > self.best_ma + self.min_delta:
            self.best_ma = ma
            self.stale_logs = 0
        else:
            self.stale_logs += 1
            if self.stale_logs >= self.patience:
                typer.echo(
                    f"\n[EarlyStopping] reward MA={ma:.3f} best_MA={self.best_ma:.3f} "
                    f"stale for {self.stale_logs} logs (patience={self.patience}). Stopping."
                )
                control.should_training_stop = True


class WandbMetricDefsCallback(TrainerCallback):
    """Register wandb metric groupings + emit a rolling `train/summary/*` panel.

    Two effects:

    1. `on_train_begin` calls `wandb.define_metric(...)` so:
         - Every `train/*` metric plots against `train/global_step`.
         - The run card / runs table show best reward (max), final KL (last),
           min completion length (min), etc.

    2. `on_log` computes running aggregates (best-so-far reward, latest KL, ...)
       and logs them under `train/summary/*`. This creates a dedicated "summary"
       panel group in the Workspace tab — monotonic climb for "best", live
       values for "current" — that you can pin to the top of the dashboard
       instead of hunting across `train/rewards/`, `train/completions/`, etc.
    """

    def __init__(self) -> None:
        self._best_reward: float = float("-inf")
        self._best_per_reward: dict[str, float] = {}
        self._min_comp_length: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if wandb.run is None:
            return
        # `define_metric` only accepts *suffix* wildcards, not interior ones.
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("train/summary/*", step_metric="train/global_step")
        # Climb-metrics: promote the max reached.
        wandb.define_metric("train/reward", summary="max")
        for fn in REWARD_FUNCS:
            wandb.define_metric(f"train/rewards/{fn.__name__}/mean", summary="max")
        # Last-value metrics: show the final state.
        wandb.define_metric("train/kl", summary="last")
        wandb.define_metric("train/loss", summary="last")
        wandb.define_metric("train/grad_norm", summary="last")
        wandb.define_metric("train/completions/clipped_ratio", summary="last")
        wandb.define_metric("train/completions/min_length", summary="min")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if wandb.run is None:
            return

        out: dict[str, float] = {}

        # Running-best reward (monotonic up).
        r = logs.get("reward")
        if r is not None:
            self._best_reward = max(self._best_reward, float(r))
            out["train/summary/best_reward"] = self._best_reward

        # Per-reward running max — one curve per reward function.
        for fn in REWARD_FUNCS:
            v = logs.get(f"rewards/{fn.__name__}/mean")
            if v is None:
                continue
            prev = self._best_per_reward.get(fn.__name__, float("-inf"))
            best = max(prev, float(v))
            self._best_per_reward[fn.__name__] = best
            out[f"train/summary/best_{fn.__name__}"] = best

        # Running-min completion length (catches reward-hacking — monotonic down).
        ml = logs.get("completions/min_length")
        if ml is not None:
            self._min_comp_length = int(ml) if self._min_comp_length is None \
                else min(self._min_comp_length, int(ml))
            out["train/summary/min_completion_length"] = self._min_comp_length

        # Current-state safety signals (not aggregated — the live value matters).
        for src, dst in (
            ("kl", "train/summary/current_kl"),
            ("grad_norm", "train/summary/current_grad_norm"),
            ("loss", "train/summary/current_loss"),
        ):
            v = logs.get(src)
            if v is not None:
                out[dst] = float(v)

        if out:
            wandb.log(out)


# =============================================================================
# Model / dataset loading
# =============================================================================


def _load_model(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_path: Optional[Path] = None,
    load_in_4bit: bool = True,
    use_gradient_checkpointing: "bool | str" = "unsloth",
):
    """Load Gemma 4 via Unsloth.

    - If `lora_path` is an existing directory, load the saved base + adapter from there.
    - Otherwise load `model_name` and attach a fresh LoRA (for training).
    """

    # Coerce string forms ("true"/"false"/"unsloth") into the values Unsloth expects.
    gc = use_gradient_checkpointing
    if isinstance(gc, str):
        gc_norm = gc.strip().lower()
        if gc_norm in ("true", "1", "yes"):
            gc = True
        elif gc_norm in ("false", "0", "no", ""):
            gc = False
        else:
            gc = gc_norm  # e.g. "unsloth" — Unsloth's offloaded variant.

    typer.echo(
        f"Memory-efficient load: load_in_4bit={load_in_4bit}, "
        f"use_gradient_checkpointing={gc!r}"
    )

    if lora_path is not None and Path(lora_path).exists():
        typer.echo(f"Loading saved model + LoRA adapter from {lora_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(lora_path),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            full_finetuning=False,
        )
        return model, tokenizer

    typer.echo(f"Loading base model {model_name} and attaching fresh LoRA (r={lora_rank})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # text-only task
        finetune_language_layers=True,
        finetune_attention_modules=True,  # attention LoRA is the main GRPO lever
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        # "unsloth" = Unsloth's offloaded gradient checkpointing — saves ~30%
        # activation VRAM at near-zero speed cost. Critical for 6k-seq GRPO
        # rollouts on a 40 GB A100.
        use_gradient_checkpointing=gc,
        random_state=3407,
    )
    return model, tokenizer


def _load_dataset(data_dir: Path, heldout_n: int = 0, seed: int = 42):
    """Load the newest `dd_dataset_*_*rows.jsonl`. If `heldout_n > 0`, return
    `(train_ds, heldout_ds)` where heldout is `heldout_n` rows sampled with a
    fixed seed and excluded from the train split. Otherwise return just `train_ds`.
    """
    jsonl_files = sorted(Path(data_dir).glob("dd_dataset_*_*rows.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No dd_dataset_*.jsonl in {data_dir}. "
            f"Run `python dd_explainer_data_generator.py -n 5500` first."
        )
    path = jsonl_files[-1]
    dataset = Dataset.from_json(str(path))
    if "__meta__" in dataset.column_names:
        dataset = dataset.filter(lambda r: r.get("__meta__") is not True).remove_columns("__meta__")
    if "row_index" in dataset.column_names:
        dataset = dataset.remove_columns("row_index")
    typer.echo(f"Loaded {len(dataset)} rows from {path.name}")

    if heldout_n <= 0:
        return dataset
    if heldout_n >= len(dataset):
        raise ValueError(f"heldout_n ({heldout_n}) >= dataset size ({len(dataset)})")
    split = dataset.train_test_split(test_size=heldout_n, seed=seed)
    typer.echo(f"Held-out split: {len(split['train'])} train + {len(split['test'])} eval (seed={seed})")
    return split["train"], split["test"]


# =============================================================================
# LangSmith trace reconstruction (used by `regress` + imported by the notebook)
# =============================================================================


_LATEST_FIELDS = [
    "dd_amount", "dd_amount_change", "recommended_dd_amount",
    "yearly_predicted_energy_cost_gbp", "reason_for_DD_change", "description",
    "collectionDay", "datetime_from", "datetime_to", "is_currently_active_DD",
    "is_exemption", "exemption_expiry_date",
    "is_amount_manually_reduced_lower_than_recommended_amount",
]
_PCH_FUEL_FIELDS = [
    "change_kwh", "change_percent",
    "latest_projected_annual_consumption_kwh", "latest_projection_date",
    "previous_projected_annual_consumption_kwh", "previous_projection_date",
]


def _scalar(v):
    """Unwrap pandas NaN / None to Python None; pass anything else through."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _json_list(v):
    """Parquet stores history arrays as JSON strings. Return a Python list or []."""
    v = _scalar(v)
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def reconstruct_pin_from_trace(row):
    """Rebuild a DDExplainerPromptInput from one parquet row.

    The traces store the transformed input (already runs through the upstream
    graph) under `outputs.dd_account_context.*`, with the history arrays as
    JSON-encoded strings. Returns None if the row is missing required fields.
    """

    try:
        latest = {f: _scalar(row.get(f"outputs.dd_account_context.latest_dd_change.{f}"))
                  for f in _LATEST_FIELDS}
        if latest.get("dd_amount") is None:
            return None

        pch_any = False
        pch: dict = {}
        for fuel in ("electricity", "gas"):
            fuel_dict: dict = {}
            for f in _PCH_FUEL_FIELDS:
                val = _scalar(row.get(
                    f"outputs.dd_account_context.account_context.projected_consumption_history.{fuel}.{f}"
                ))
                if val is not None:
                    fuel_dict[f] = val
            if fuel_dict:
                pch[fuel] = fuel_dict
                pch_any = True

        ctx = {
            "dd_change_history": _json_list(row.get("outputs.dd_account_context.account_context.dd_change_history")),
            "payment_history":   _json_list(row.get("outputs.dd_account_context.account_context.payment_history")),
            "contract_history":  _json_list(row.get("outputs.dd_account_context.account_context.contract_history")),
            "projected_consumption_history": pch if pch_any else None,
        }
        return DDExplainerPromptInput.model_validate(
            {"account_context": ctx, "latest_dd_change": latest}
        )
    except Exception:
        return None


def prompt_token_length(tokenizer, pin) -> int:
    """How many tokens the rendered DD explainer prompt uses. Needed so the
    regression cell can filter out traces whose context overflows."""

    text = tokenizer.apply_chat_template(
        build_chat_messages(pin), tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(text=text, add_special_tokens=False)["input_ids"])


# =============================================================================
# Subcommand: train
# =============================================================================


@app.command()
def train(
    config_name: str = typer.Option(
        "train", "--config-name", "-c",
        help="Hydra config name in configs/ (e.g. 'train', 'smoke', 'train_long').",
    ),
    notes: Optional[str] = typer.Option(
        None, "--notes", "-d",
        help="Free-text note for the run, stored as wandb run notes.",
    ),
    # Tunables default to None so the YAML value is kept unless the user
    # explicitly passes a CLI flag (CLI > YAML > Pydantic defaults).
    model_name: Optional[str] = typer.Option(None),
    data_dir: Optional[Path] = typer.Option(None),
    save_path: Optional[Path] = typer.Option(None),
    output_dir: Optional[Path] = typer.Option(None),
    max_seq_length: Optional[int] = typer.Option(None),
    max_completion_length: Optional[int] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    grad_accum: Optional[int] = typer.Option(None),
    num_generations: Optional[int] = typer.Option(None),
    learning_rate: Optional[float] = typer.Option(None),
    beta: Optional[float] = typer.Option(None),
    lora_rank: Optional[int] = typer.Option(None),
    load_in_4bit: Optional[bool] = typer.Option(
        None,
        "--load-in-4bit/--no-load-in-4bit",
        help="Quantize base model to 4-bit on load (saves ~12 GB of host/GPU RAM).",
    ),
    use_gradient_checkpointing: Optional[str] = typer.Option(
        None,
        help="Gradient checkpointing mode: \"unsloth\" (offloaded, recommended), "
             "\"true\", or \"false\".",
    ),
    max_steps: Optional[int] = typer.Option(None),
    warmup_steps: Optional[int] = typer.Option(None),
    save_steps: Optional[int] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    patience: Optional[int] = typer.Option(
        None,
        help="If > 0, stop early after this many consecutive logs with no moving-avg "
             "reward improvement.",
    ),
    plateau_window: Optional[int] = typer.Option(None),
    plateau_delta: Optional[float] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(
        None, help="Weights & Biases run name. Auto-generated if omitted.",
    ),
) -> None:
    """Run GRPO training with the 7 verifiable rewards from dd_explainer_rewards."""
    # 1. Load Hydra YAML → typed Settings.
    settings = load_hydra_settings(config_name)

    # 2. Overlay non-None CLI flags on top of the YAML.
    cli_overrides = {
        "model_name": model_name, "data_dir": data_dir, "save_path": save_path,
        "output_dir": output_dir, "max_seq_length": max_seq_length,
        "max_completion_length": max_completion_length, "batch_size": batch_size,
        "grad_accum": grad_accum, "num_generations": num_generations,
        "learning_rate": learning_rate, "beta": beta, "lora_rank": lora_rank,
        "load_in_4bit": load_in_4bit,
        "use_gradient_checkpointing": use_gradient_checkpointing,
        "max_steps": max_steps, "warmup_steps": warmup_steps, "save_steps": save_steps,
        "seed": seed, "patience": patience, "plateau_window": plateau_window,
        "plateau_delta": plateau_delta,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            setattr(settings.train, k, v)

    if wandb_run_name is not None:
        settings.wandb.run_name = wandb_run_name
    if notes is not None:
        settings.wandb.notes = notes

    t = settings.train
    if (t.batch_size * t.grad_accum) % t.num_generations != 0:
        raise typer.BadParameter(
            f"(batch_size * grad_accum) must be divisible by num_generations: "
            f"{t.batch_size} * {t.grad_accum} = {t.batch_size * t.grad_accum}, "
            f"num_generations = {t.num_generations}"
        )

    typer.echo(f"=== Resolved Settings (config_name={config_name!r}) ===")
    typer.echo(settings.model_dump_json(indent=2))
    typer.echo("===")

    # 3. Init wandb from Hydra settings (notes/tags/run_id/run_name have no
    # TrainingArguments field, so we init the run ourselves and let HF's
    # WandbCallback attach to it). report_to drives whether the callback runs.
    if settings.wandb.enabled:
        wandb.init(**settings.wandb.init_kwargs(), config=settings.model_dump(mode="json"))
    report_to = "wandb" if settings.wandb.enabled else "none"

    model, tokenizer = _load_model(
        t.model_name,
        t.max_seq_length,
        t.lora_rank,
        load_in_4bit=t.load_in_4bit,
        use_gradient_checkpointing=t.use_gradient_checkpointing,
    )
    if t.eval_heldout_n > 0:
        dataset, heldout_ds = _load_dataset(t.data_dir, heldout_n=t.eval_heldout_n, seed=t.seed)
    else:
        dataset = _load_dataset(t.data_dir)
        heldout_ds = None

    callbacks = []
    if t.patience > 0:
        callbacks.append(RewardPlateauCallback(
            patience=t.patience, window=t.plateau_window, min_delta=t.plateau_delta,
        ))
        typer.echo(
            f"Early stopping armed: patience={t.patience}, "
            f"window={t.plateau_window}, min_delta={t.plateau_delta}"
        )
    if settings.wandb.enabled:
        callbacks.append(WandbMetricDefsCallback())

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=t.learning_rate,
        beta=t.beta,
        weight_decay=t.weight_decay,
        warmup_steps=t.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum,
        num_generations=t.num_generations,
        max_completion_length=t.max_completion_length,
        max_steps=t.max_steps,
        save_steps=t.save_steps,
        report_to=report_to,
        output_dir=str(t.output_dir),
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        # Drop overlong rollouts from the loss so one 1024-token blowup
        # can't dominate the gradient.
        mask_truncated_completions=True,
        # --- Memory-efficient training -------------------------------
        # Even with Unsloth's offloaded checkpointing on the LoRA wrapper,
        # forcing TRL to use checkpointing for the policy forward pass is
        # the belt-and-braces way to cap activation memory.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # The dataset is small + tokenization is cheap; extra workers just
        # multiply host-RAM usage and increase OOM risk in containers with
        # tight cgroup limits.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=t.seed,
        run_name=settings.wandb.run_name,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks,
    )
    # Trap SIGTERM/SIGINT so an external `kill <pid>` still runs the finally
    # block and the run gets logged to experiments/ instead of vanishing.
    def _signal_to_exit(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _signal_to_exit)

    started = time.monotonic()
    train_status = "RUNNING"
    crash_reason: str | None = None
    eval_metrics: dict = {}
    try:
        trainer.train()
        completed = int(getattr(trainer.state, "global_step", 0) or 0)
        train_status = "EARLY_STOPPED" if completed < t.max_steps else "COMPLETED"
        # Post-train evals (only on successful completion — skip on CRASH).
        # Run before autolog so the metrics land in results.jsonl in one row.
        if t.eval_heldout_n > 0 and heldout_ds is not None:
            eval_metrics["heldout"] = _run_heldout(
                model, tokenizer, heldout_ds, t.max_completion_length, t.eval_batch_size,
            )
        if t.eval_regression_n > 0:
            eval_metrics["regression"] = _run_regression(
                model, tokenizer, t.eval_trace_dir, t.eval_regression_n,
                t.max_completion_length, t.max_seq_length, t.eval_batch_size,
            )
    except BaseException as e:
        train_status = "CRASH"
        # KeyboardInterrupt → external SIGINT/SIGTERM; anything else is a
        # real crash (OOM, assertion, etc).
        first_line = str(e).strip().splitlines()[0] if str(e).strip() else ""
        if isinstance(e, KeyboardInterrupt):
            crash_reason = f"interrupted: {first_line or 'SIGINT/SIGTERM'}"
        else:
            crash_reason = f"{type(e).__name__}: {first_line[:120]}" if first_line else type(e).__name__
        raise
    finally:
        runtime_min = (time.monotonic() - started) / 60.0
        if settings.wandb.enabled and wandb.run is not None:
            _autolog_experiment(settings, config_name, trainer, runtime_min, train_status,
                                eval_metrics=eval_metrics, crash_reason=crash_reason)

    t.save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(t.save_path))
    tokenizer.save_pretrained(str(t.save_path))
    _verify_adapter_nonzero(t.save_path)
    typer.echo(f"Saved LoRA adapter to {t.save_path}")


def _autolog_experiment(settings, config_name: str, trainer, runtime_min: float,
                        train_status: str, eval_metrics: dict | None = None,
                        crash_reason: str | None = None) -> None:
    """Append a row to experiments/<task>/results.jsonl + refresh the plot.

    Pulls best-reward / final-kl from `wandb.run.summary` (populated by the
    WandbMetricDefsCallback). On CRASH, status is logged as CRASH; otherwise
    BASELINE (first run) / KEEP (best so far) / DISCARD is decided by
    `experiment_progress.log_experiment`.
    """
    summary = dict(wandb.run.summary) if wandb.run.summary else {}
    best_reward = summary.get("train/summary/best_reward") or summary.get("train/reward") or 0.0
    metrics = {
        "best_reward": float(best_reward),
        "final_kl": summary.get("train/summary/current_kl"),
        "final_loss": summary.get("train/summary/current_loss"),
        "min_completion_length": summary.get("train/summary/min_completion_length"),
        "train_status": train_status,
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}
    if eval_metrics:
        for k, v in eval_metrics.items():
            if v and v.get("n", 0) > 0:
                metrics[k] = v
    if crash_reason:
        metrics["crash_reason"] = crash_reason

    # CRASH gets a hard status; everything else (including EARLY_STOPPED) goes
    # through the auto BASELINE/KEEP/DISCARD scoring so partial runs still get
    # judged on whatever best_reward they achieved.
    forced_status = "CRASH" if train_status == "CRASH" else None
    desc_tag = f"[{train_status.lower()}] " if train_status in ("CRASH", "EARLY_STOPPED") else ""

    log_experiment(
        score=float(best_reward),
        description=f"{desc_tag}config={config_name}; {settings.wandb.notes or '(no notes)'}",
        config_name=config_name,
        steps=int(getattr(trainer.state, "global_step", 0) or 0),
        runtime_min=runtime_min,
        status=forced_status,
        metrics=metrics,
        notes=settings.wandb.notes or "",
        wandb_url=wandb.run.get_url() or "",
        wandb_run_id=wandb.run.id,
        wandb_run_name=wandb.run.name or "",
    )
    plot_progress()


# =============================================================================
# Post-train evaluation helpers (held-out + regression)
# =============================================================================
#
# Both eval modes generate at temperature=0 (deterministic) and score with the
# same 7 reward functions used during training. Outputs aggregate dicts that
# get nested under `metrics["heldout"]` / `metrics["regression"]` in
# results.jsonl so the chart can surface them alongside the train reward.

# Per-component max scores — used to compute "pass" (= component at its max)
_REWARD_MAX = {
    "schema_valid":          1.0,
    "in_enum":               1.0,
    "f1_triggers":          10.0,
    "prev_amount_correct":   2.0,
    "no_hallucinated_facts": 1.0,
    "underpayment_ok":       0.5,
    "well_formed":           0.5,
}
_REWARD_TOTAL_MAX = sum(_REWARD_MAX.values())  # 16.0


@torch.no_grad()
def _generate_batch(model, tokenizer, messages_list, max_completion_length: int) -> list[str]:
    """Batched temp=0 generation. ~5-8x throughput vs single-row on A100."""
    if not messages_list:
        return []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    enc = tokenizer(
        texts, add_special_tokens=False, return_tensors="pt",
        padding=True, padding_side="left",
    ).to("cuda")
    out = model.generate(
        **enc, max_new_tokens=max_completion_length,
        temperature=0.0, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


def _aggregate_scores(rows: list[dict]) -> dict:
    """Turn per-row score dicts into pass-rate / mean aggregate."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    out: dict = {"n": n}
    pass_all = 0
    for row in rows:
        if all(row.get(k, 0.0) >= mx - 1e-6 for k, mx in _REWARD_MAX.items()):
            pass_all += 1
    out["pass_all"] = pass_all
    out["pass_all_pct"] = round(100 * pass_all / n, 1)
    # Per-component pass rate + mean
    for k, mx in _REWARD_MAX.items():
        vals = [row.get(k, 0.0) for row in rows]
        out[f"{k}_pass"] = sum(1 for v in vals if v >= mx - 1e-6)
        out[f"{k}_mean"] = round(sum(vals) / n, 3)
    out["mean_total"] = round(sum(sum(row.get(k, 0.0) for k in _REWARD_MAX) for row in rows) / n, 3)
    return out


def _score_items(model, tokenizer, items, max_completion_length: int,
                 batch_size: int, label: str) -> list[dict]:
    """Score (messages, gt_triggers, input_json) tuples in batches."""
    rows: list[dict] = []
    n = len(items)
    for start in range(0, n, batch_size):
        chunk = items[start:start + batch_size]
        try:
            completions = _generate_batch(
                model, tokenizer, [m for m, _, _ in chunk], max_completion_length,
            )
        except Exception as e:
            typer.echo(f"[{label}] batch starting at {start} skipped: {e}")
            continue
        for (_, gt, inp), completion in zip(chunk, completions):
            try:
                rows.append(score_completion(completion, gt, inp))
            except Exception as e:
                typer.echo(f"[{label}] scoring failed: {e}")
        done = min(start + batch_size, n)
        if done % (batch_size * 4) == 0 or done == n:
            typer.echo(f"[{label}] {done}/{n} done")
    return rows


def _run_heldout(model, tokenizer, heldout_ds, max_completion_length: int,
                 batch_size: int) -> dict:
    """Score on a held-out IID sample of the synthetic dataset (generalization).

    Heldout rows are dicts with `prompt` (chat messages), `ground_truth_triggers`,
    and `input_json` — straight from `dd_explainer_data_generator`.
    """
    if heldout_ds is None or len(heldout_ds) == 0:
        return {"n": 0}
    items = [
        (row["prompt"], sorted(row["ground_truth_triggers"]), row["input_json"])
        for row in heldout_ds
    ]
    typer.echo(f"[eval/heldout] scoring {len(items)} rows (temp=0, batch={batch_size})…")
    rows = _score_items(model, tokenizer, items, max_completion_length, batch_size, "eval/heldout")
    agg = _aggregate_scores(rows)
    typer.echo(f"[eval/heldout] pass_all={agg.get('pass_all',0)}/{agg.get('n',0)} "
               f"({agg.get('pass_all_pct',0)}%) mean={agg.get('mean_total',0)}")
    return agg


def _run_regression(model, tokenizer, trace_dir: Path, n_rows: int,
                    max_completion_length: int, max_seq_length: int,
                    batch_size: int) -> dict:
    """Score on previously-failed LangSmith production traces (regression gate).

    Reconstruct each prompt from the parquet via `reconstruct_pin_from_trace`
    so we can rebuild chat messages with `build_chat_messages`.
    """
    traces_path = trace_dir / "traces.parquet"
    if not traces_path.exists():
        typer.echo(f"[eval/regression] {traces_path} missing — skipped")
        return {"n": 0}
    df = pd.read_parquet(traces_path)
    failed = df[df["feedback.direct_debit_faithfulness"] < 1.0]
    budget = max_seq_length - max_completion_length - 64
    items = []
    for _, row in failed.iterrows():
        pin = reconstruct_pin_from_trace(row)
        if pin is None:
            continue
        if prompt_token_length(tokenizer, pin) > budget:
            continue
        items.append((
            build_chat_messages(pin),
            sorted(t.value for t in detect_triggers(pin)),
            pin.model_dump(mode="json"),
        ))
        if len(items) >= n_rows:
            break
    typer.echo(f"[eval/regression] scoring {len(items)} prior-failed traces "
               f"(temp=0, batch={batch_size})…")
    rows = _score_items(model, tokenizer, items, max_completion_length, batch_size, "eval/regression")
    agg = _aggregate_scores(rows)
    typer.echo(f"[eval/regression] pass_all={agg.get('pass_all',0)}/{agg.get('n',0)} "
               f"({agg.get('pass_all_pct',0)}%) mean={agg.get('mean_total',0)}")
    return agg


def _verify_adapter_nonzero(save_path: Path) -> None:

    adapter_file = save_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        typer.echo(f"warning: {adapter_file} not found — skipping zero-check")
        return
    with safe_open(str(adapter_file), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum().item()
            if n_zeros == tensor.numel():
                raise RuntimeError(f"Adapter zero-check failed: '{key}' is all zero.")


# =============================================================================
# Subcommand: infer
# =============================================================================


@app.command()
def infer(
    model_name: str = typer.Option("unsloth/gemma-4-E4B-it"),
    lora_path: Optional[Path] = typer.Option(
        None, help="Load a saved LoRA adapter directory. Omit to run the base model.",
    ),
    target_trigger: Optional[str] = typer.Option(
        None,
        help="Steer the synthetic example toward a specific Trigger enum value, "
             "e.g. 'Change in usage'.  Omit for a random single trigger.",
    ),
    max_seq_length: int = typer.Option(6144),
    max_new_tokens: int = typer.Option(1024),
    lora_rank: int = typer.Option(64),
    seed: int = typer.Option(2026),
) -> None:
    """Generate one fresh DDExplainerPromptInput and roll the model out on it."""
    _setup_workspace_env()


    model, tokenizer = _load_model(model_name, max_seq_length, lora_rank, lora_path=lora_path)

    rng = random.Random(seed)
    if target_trigger:
        try:
            target = {Trigger(target_trigger)}
        except ValueError:
            valid = ", ".join(t.value for t in Trigger)
            raise typer.BadParameter(f"Unknown trigger '{target_trigger}'. Valid: {valid}")
    else:
        non_terminal = [t for t in Trigger if t != Trigger.No_triggers_identified]
        target = {rng.choice(non_terminal)}
    pin = generate_dd_example(target, rng)

    typer.echo(f"\n--- Target triggers: {sorted(t.value for t in target)}")
    typer.echo(f"--- Oracle detects:  {sorted(t.value for t in detect_triggers(pin))}\n")

    # Two-step tokenize avoids the multimodal content-block parser choking on plain strings.
    text = tokenizer.apply_chat_template(
        build_chat_messages(pin), tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text=text, add_special_tokens=False, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        use_cache=True,
        streamer=streamer,
    )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )

    parsed = parse_response(completion)
    if parsed is None:
        typer.echo("\n(schema) ❌ Output did not parse as DirectDebitExplainerResponse.")
    else:
        typer.echo("\n(schema) ✓ Parsed. Triggers: " + ", ".join(e.trigger.value for e in parsed.explanations))


# =============================================================================
# Subcommand: regress
# =============================================================================


@app.command()
def regress(
    model_name: str = typer.Option("unsloth/gemma-4-E4B-it"),
    lora_path: Optional[Path] = typer.Option(
        None, help="Saved LoRA adapter dir (omit to score the base model).",
    ),
    trace_dir: Path = typer.Option(
        Path("/workspace/gemma4_rl/.error_analysis_cache/20260413T075447Z_20260420T075447Z"),
        help="Directory containing traces.parquet.",
    ),
    max_seq_length: int = typer.Option(6144),
    max_completion_length: int = typer.Option(1024),
    lora_rank: int = typer.Option(64),
    n_rows: int = typer.Option(20, help="How many previously-failed rows to re-score."),
) -> None:
    """Re-score the model against LangSmith's previously-failed traces."""
    _setup_workspace_env()

    model, tokenizer = _load_model(model_name, max_seq_length, lora_rank, lora_path=lora_path)
    agg = _run_regression(
        model, tokenizer, trace_dir, n_rows,
        max_completion_length, max_seq_length, batch_size=8,
    )
    typer.echo(f"\nAggregate: {json.dumps(agg, indent=2)}")


if __name__ == "__main__":
    app()
