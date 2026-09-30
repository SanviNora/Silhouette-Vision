"""Fetch the data bundle on a fresh host (Streamlit Community Cloud).

The bundle (scripts/10_build_bundle.py) lives in a Hugging Face dataset repo, with its thumbnails
packed into one tar per source (scripts/11_publish.py). It carries a VERSION file: a host holding
an older bundle (the app's temp folder survives redeploys) wipes it and downloads again.
"""

import shutil
import tarfile
from pathlib import Path

BUNDLE_VERSION = "2026-09-30"  # bump when the bundle's contents change


def ensure_bundle(root: Path, repo: str) -> None:
    from huggingface_hub import snapshot_download

    version = root / "VERSION"
    is_bundle = version.exists() or root.name == "silhouette_bundle"  # never wipe anything else
    if root.exists() and is_bundle and (not version.exists() or version.read_text().strip() != BUNDLE_VERSION):
        shutil.rmtree(root)
    elif root.exists() and not is_bundle and any(root.iterdir()):
        raise RuntimeError(f"{root} is not a data bundle; refusing to download into it")
    if not version.exists():
        snapshot_download(repo, repo_type="dataset", local_dir=root)
    for archive in sorted((root / "archives").glob("thumbs_*.tar")):
        target = root / "data/processed/thumbs" / archive.stem.removeprefix("thumbs_")
        if not target.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")
