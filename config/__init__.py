"""Hydra-composed Pydantic settings for the direct_debit_explainer RLVR mock.

YAMLs live in `configs/`, Pydantic models in `config/`. `load_hydra_settings`
returns a fully typed `Settings` instance combining both.
"""

from config.base import Settings, TrainSettings, WandbSettings
from config.utils import load_hydra_settings

__all__ = ["Settings", "TrainSettings", "WandbSettings", "load_hydra_settings"]
