"""Load a Hydra-composed `Settings` instance from `configs/<name>.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.base import Settings

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "configs"


def load_hydra_settings(config_name: str = "train") -> Settings:
    """Compose `configs/<config_name>.yaml` (with Hydra `defaults:` resolution)
    and coerce into a typed `Settings` instance.
    """
    with initialize_config_dir(version_base=hydra.__version__, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name)
        cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return Settings(**cfg_dict)
