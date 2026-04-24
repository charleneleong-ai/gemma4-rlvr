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

    def apply_env(self) -> None:
        """Re-export to os.environ so the transformers/trl wandb callback picks them up."""
        if not self.enabled:
            os.environ["WANDB_MODE"] = "disabled"
            return
        os.environ["WANDB_PROJECT"] = self.project
        if self.entity:
            os.environ["WANDB_ENTITY"] = self.entity
        if self.run_name:
            os.environ["WANDB_NAME"] = self.run_name
        if self.run_id:
            os.environ["WANDB_RUN_ID"] = self.run_id
        os.environ["WANDB_MODE"] = self.mode


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

    max_steps: int = 300
    warmup_steps: int = 30
    save_steps: int = 50
    seed: int = 42

    # Early stopping (disabled by default; set patience > 0 to arm)
    patience: int = 0
    plateau_window: int = 10
    plateau_delta: float = 0.05


class Settings(BaseModel):
    train: TrainSettings = Field(default_factory=TrainSettings)
    wandb: WandbSettings = Field(default_factory=WandbSettings)
