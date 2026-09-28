"""Project paths and configuration."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load_config(name: str = "pipeline") -> dict:
    with open(ROOT / "configs" / f"{name}.yml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in configs/pipeline.yml."""
    return ROOT / load_config()["paths"][key]
