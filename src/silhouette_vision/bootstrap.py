"""Fetch the public data bundle on a fresh host (Streamlit Cloud, Hugging Face Space).

The bundle (scripts/10_build_public.py) lives in a Hugging Face dataset repo, with its ~47k
thumbnails packed into one tar per source (scripts/11_publish.py). Downloads once, then unpacks.
"""

import tarfile
from pathlib import Path


def ensure_bundle(root: Path, repo: str) -> None:
    from huggingface_hub import snapshot_download

    if not (root / "data/processed/catalog.parquet").exists():
        snapshot_download(repo, repo_type="dataset", local_dir=root)
    for archive in sorted((root / "archives").glob("thumbs_*.tar")):
        target = root / "data/processed/thumbs" / archive.stem.removeprefix("thumbs_")
        if not target.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")
