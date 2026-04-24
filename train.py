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
    """Register wandb metric groupings and summary aggregations on train start.

    Effects on the wandb run:
      - Every `train/*` metric plots against `train/global_step` as the x-axis
        (regardless of log order).
      - `train/reward` and per-reward means are surfaced as "max so far" in the
        run's top-level summary panel.
      - `train/kl`, `train/grad_norm`, `train/loss` are surfaced as "last value".

    Produces the at-a-glance panel: best reward reached, final KL / loss /
    grad_norm — the metrics that tell you whether the run is healthy without
    opening the chart view.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        import wandb
        if wandb.run is None:
            return
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        # Climb-metrics: promote the max reached.
        wandb.define_metric("train/reward", summary="max")
        wandb.define_metric("train/rewards/*/mean", summary="max")
        # Last-value metrics: show the final state.
        wandb.define_metric("train/kl", summary="last")
        wandb.define_metric("train/loss", summary="last")
        wandb.define_metric("train/grad_norm", summary="last")
        wandb.define_metric("train/completions/clipped_ratio", summary="last")
        wandb.define_metric("train/completions/min_length", summary="min")


# =============================================================================
# Model / dataset loading
# =============================================================================


def _load_model(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_path: Optional[Path] = None,
):
    """Load Gemma 4 via Unsloth.

    - If `lora_path` is an existing directory, load the saved base + adapter from there.
    - Otherwise load `model_name` and attach a fresh LoRA (for training).
    """

    if lora_path is not None and Path(lora_path).exists():
        typer.echo(f"Loading saved model + LoRA adapter from {lora_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(lora_path),
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False,
        )
        return model, tokenizer

    typer.echo(f"Loading base model {model_name} and attaching fresh LoRA (r={lora_rank})")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
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
        use_gradient_checkpointing=False,  # KV cache off is ~5-10× slower on Gemma 4
        random_state=3407,
    )
    return model, tokenizer


def _load_dataset(data_dir: Path):
    """Glob the newest `dd_dataset_*_*rows.jsonl` in `data_dir`; drop the __meta__
    header row and row_index column. Raises if nothing found.
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
    return dataset


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

    # 3. Wire wandb via env vars before trl/transformers spin up its callback.
    settings.wandb.apply_env()
    report_to = "wandb" if settings.wandb.enabled else "none"


    model, tokenizer = _load_model(t.model_name, t.max_seq_length, t.lora_rank)
    dataset = _load_dataset(t.data_dir)

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
        mask_truncated_completions=True,
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
    trainer.train()

    t.save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(t.save_path))
    tokenizer.save_pretrained(str(t.save_path))
    _verify_adapter_nonzero(t.save_path)
    typer.echo(f"Saved LoRA adapter to {t.save_path}")


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

    df = pd.read_parquet(trace_dir / "traces.parquet")
    failed = df[df["feedback.direct_debit_faithfulness"] < 1.0]
    typer.echo(f"Loaded {len(failed)} previously-failed prompts from {trace_dir.name}")

    budget = max_seq_length - max_completion_length - 64
    candidates = []
    for _, row in failed.iterrows():
        pin = reconstruct_pin_from_trace(row)
        if pin is None:
            continue
        if prompt_token_length(tokenizer, pin) <= budget:
            candidates.append(pin)
    typer.echo(f"Candidates after token-budget filter: {len(candidates)} / {len(failed)}")

    @torch.no_grad()
    def _generate(pin) -> str:
        text = tokenizer.apply_chat_template(
            build_chat_messages(pin), tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text=text, add_special_tokens=False, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    rows_out = []
    for pin in candidates[:n_rows]:
        completion = _generate(pin)
        gt = sorted(t.value for t in detect_triggers(pin))
        rows_out.append(score_completion(completion, gt, pin.model_dump(mode="json")))

    regression_df = pd.DataFrame(rows_out)
    typer.echo(f"\nScored {len(regression_df)} rows. Mean reward by category:")
    typer.echo(regression_df.mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    app()
