"""Pydantic settings models — loaded from `configs/*.yaml` via Hydra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class WandbSettings(BaseModel):
    """Weights & Biases logging settings.

    `mode` is YAML-pinned per-config so smoke runs can force `disabled` even
    when WANDB_API_KEY is set; shell `WANDB_MODE=...` still overrides if you
    need a one-off bypass without editing YAML.
    """

    project: str = "gemma4-rlvr"
    entity: Optional[str] = "chaleong"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    mode: str = "online"  # "online" | "offline" | "disabled"
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled" and bool(self.project)

    def init_kwargs(self) -> dict:
        """Kwargs for `wandb.init()`. We init wandb ourselves (rather than letting
        HF's WandbCallback do it from env vars) so notes/tags/run_id/run_name —
        which `TrainingArguments` has no field for — flow straight from Hydra.
        """
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.run_name,
            "id": self.run_id,
            "mode": self.mode,
            "notes": self.notes,
            "tags": self.tags or None,
        }


class TrainSettings(BaseModel):
    """GRPO training hyper-parameters."""

    model_name: str = "unsloth/gemma-4-E4B-it"
    data_dir: Path = Path("/workspace/gemma4_rl/data")
    save_path: Path = Path("gemma_4_lora")
    output_dir: Path = Path("outputs")

    max_seq_length: int = 6144
    max_completion_length: int = 1024

    batch_size: int = 8
    grad_accum: int = 1
    num_generations: int = 4

    learning_rate: float = 1e-5
    beta: float = 0.04
    weight_decay: float = 0.001

    lora_rank: int = 64

    # --- Memory-efficient knobs ---------------------------------------
    # Quantize the base model to 4-bit on load (LoRA adapters stay in
    # bf16). Drops the 8B base from ~16 GB to ~4-5 GB of VRAM/host RAM
    # so the GRPO rollout buffer + reference-policy copy can breathe.
    load_in_4bit: bool = True
    # "unsloth" = Unsloth's offloaded gradient checkpointing (recommended).
    # Accepts "unsloth", "true", or "false" — coerced inside _load_model.
    use_gradient_checkpointing: str = "unsloth"

    max_steps: int = 300
    warmup_steps: int = 30
    save_steps: int = 50
    seed: int = 42

    # Early stopping (disabled by default; set patience > 0 to arm)
    patience: int = 0
    plateau_window: int = 10
    plateau_delta: float = 0.05

    # Post-train evaluation. Both run after a successful train pass and
    # write into results.jsonl `metrics`. Set to 0 to skip either eval.
    # heldout: random sample from the dataset (excluded from training).
    # regression: known-failed LangSmith traces from .error_analysis_cache/.
    eval_heldout_n: int = 1000      # ~18% of 5500-row dataset → ±1.6% std err on pass-rate
    eval_regression_n: int = 100    # ~54% of 187 failed-trace cache
    eval_batch_size: int = 32       # measured: bs=24 stable at 27.3/40 GB, 48% compute util
                                    # (CPU-bound on apply_chat_template). bs=32 → ~32.6 GB,
                                    # ~8 GB safety margin for variable-length completions.
    eval_trace_dir: Path = Path(
        "/workspace/gemma4_rl/.error_analysis_cache/20260413T075447Z_20260420T075447Z"
    )

    # Periodic completion preview to W&B. Generates completions for a fixed
    # train + heldout sample every N steps and logs them as a wandb Table so
    # you can scrub through training and see how outputs evolve. Set
    # `completion_preview_every=0` to disable. ~3-5s overhead per logged step.
    completion_preview_every: int = 25
    completion_preview_n_train: int = 4
    completion_preview_n_heldout: int = 4


class Settings(BaseModel):
    train: TrainSettings = Field(default_factory=TrainSettings)
    wandb: WandbSettings = Field(default_factory=WandbSettings)
