"""Project paths and configuration.

ROOT is the repository (code, configs). DATA_ROOT holds data/ and artifacts/:
- SILHOUETTE_DATA_ROOT, if set (a deploy bundle with the same layout, scripts/10_build_public.py);
- else the repository, if it holds raw data (local development);
- else HOSTED: a fresh host (e.g. Streamlit Cloud) with no data. The app then downloads the data
  bundle (scripts/10_build_bundle.py) from DATA_REPO into a temp folder, with no settings needed.
The app is the same locally and hosted: every catalog source may be shared.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Model roles (docs/phase2_results.md, October 2026 update). Photos are embedded with MODA (LookBench
# exact R@1 51.6 -> 56.5); its text tower is Marqo-FashionSigLIP's, unchanged, so text queries are
# matched against the Marqo catalog embeddings, where image and text are still aligned (text search
# against MODA image embeddings loses ~10 points of R@10). One model in memory serves both.
PHOTO_MODEL = "moda_fashion_distilled"
TEXT_MODEL = "marqo_fashion_siglip"
DATA_REPO = os.environ.get("DATA_REPO", "Sanvii/silhouette-vision-data")
if "SILHOUETTE_DATA_ROOT" in os.environ:
    DATA_ROOT, HOSTED = Path(os.environ["SILHOUETTE_DATA_ROOT"]), False
elif any((ROOT / "data/raw").glob("*/")):
    DATA_ROOT, HOSTED = ROOT, False
else:
    DATA_ROOT, HOSTED = Path(tempfile.gettempdir()) / "silhouette_bundle", True


@lru_cache
def load_config(name: str = "pipeline") -> dict:
    with open(ROOT / "configs" / f"{name}.yml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in configs/pipeline.yml."""
    return DATA_ROOT / load_config()["paths"][key]
