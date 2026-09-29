"""Project paths and configuration.

ROOT is the repository (code, configs). DATA_ROOT holds data/ and artifacts/:
- SILHOUETTE_DATA_ROOT, if set (a deploy bundle with the same layout, scripts/10_build_public.py);
- else the repository, if the pipeline has been run here (local development);
- else HOSTED: a fresh host (e.g. Streamlit Cloud) with no data. The app then downloads the public
  bundle from DATA_REPO into a temp folder and runs in public mode, with no settings needed.
PUBLIC (SILHOUETTE_PUBLIC=1, or HOSTED) leaves out the Farfetch catalog, whose license allows
analysis and a private demo only.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_REPO = os.environ.get("DATA_REPO", "Sanvii/silhouette-vision-data")
if "SILHOUETTE_DATA_ROOT" in os.environ:
    DATA_ROOT, HOSTED = Path(os.environ["SILHOUETTE_DATA_ROOT"]), False
elif (ROOT / "data/processed/catalog.parquet").exists():
    DATA_ROOT, HOSTED = ROOT, False
else:
    DATA_ROOT, HOSTED = Path(tempfile.gettempdir()) / "silhouette_bundle", True
PUBLIC = os.environ.get("SILHOUETTE_PUBLIC") == "1" or HOSTED


@lru_cache
def load_config(name: str = "pipeline") -> dict:
    with open(ROOT / "configs" / f"{name}.yml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in configs/pipeline.yml."""
    return DATA_ROOT / load_config()["paths"][key]
