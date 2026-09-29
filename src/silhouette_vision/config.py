"""Project paths and configuration.

ROOT is the repository (code, configs). DATA_ROOT holds data/ and artifacts/; it is the repository
too unless SILHOUETTE_DATA_ROOT points at a deploy bundle with the same layout
(scripts/10_build_public.py). SILHOUETTE_PUBLIC=1 runs the app without the Farfetch catalog,
whose license allows analysis and a private demo only.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SILHOUETTE_DATA_ROOT", ROOT))
PUBLIC = os.environ.get("SILHOUETTE_PUBLIC") == "1"


@lru_cache
def load_config(name: str = "pipeline") -> dict:
    with open(ROOT / "configs" / f"{name}.yml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in configs/pipeline.yml."""
    return DATA_ROOT / load_config()["paths"][key]
