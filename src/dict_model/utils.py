"""Shared helpers: seeding and YAML config loading."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Return the repo root (the directory that holds ``configs/``)."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return here.parent.parent.parent
