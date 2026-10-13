"""Configuration loader and project paths."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
FIG_DIR = ROOT / "reports" / "figures"
METRIC_DIR = ROOT / "reports" / "metrics"
MODEL_DIR = ROOT / "models"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def path(rel: str) -> Path:
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True) if not p.suffix else p.parent.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # pragma: no cover
        pass


CFG = load_config()
CLASSES: list[str] = CFG["classes"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
for d in (FIG_DIR, METRIC_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
