"""Stage 11: publish the data bundle to Hugging Face for the hosted app.

The app runs on Streamlit Community Cloud (deploy/STREAMLIT_CLOUD.md) and downloads this dataset
on first start (silhouette_vision.bootstrap). Needs a Hugging Face write token (`hf auth login`).
The dataset repo is made to mirror the local bundle exactly: files no longer in it are deleted.

Usage: python scripts/11_publish.py [--user Sanvii]
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import HfApi

from silhouette_vision.config import ROOT


def stage_data(bundle: Path, staging: Path) -> Path:
    """The bundle with its ~56k thumbnails packed into one tar per source: uploading that many
    small files hits Hugging Face's rate limit (HTTP 429), and downloading them slows start-up.
    bootstrap.ensure_bundle unpacks the archives."""
    if not bundle.exists():
        raise SystemExit("Build the bundle first: python scripts/10_build_bundle.py")
    if staging.exists():
        shutil.rmtree(staging)
    thumbs = bundle / "data/processed/thumbs"
    for f in bundle.rglob("*"):
        rel = f.relative_to(bundle)
        if f.is_file() and not f.is_relative_to(thumbs) and rel.parts[0] != ".cache":
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(f, staging / rel)
    (staging / "archives").mkdir(parents=True, exist_ok=True)
    for source in sorted(p.name for p in thumbs.iterdir()):
        archive = staging / "archives" / f"thumbs_{source}.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(thumbs / source, arcname=f"data/processed/thumbs/{source}")
        print(f"packed {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)")
    shutil.copy2(ROOT / "deploy/DATA_README.md", staging / "README.md")
    return staging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Sanvii")
    args = ap.parse_args()
    api = HfApi()
    repo = f"{args.user}/silhouette-vision-data"
    staging = stage_data(ROOT / "deploy_bundle", ROOT / "deploy_upload")
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=staging)
    local = {str(f.relative_to(staging)) for f in staging.rglob("*") if f.is_file()
             and f.relative_to(staging).parts[0] != ".cache"}
    stale = [f for f in api.list_repo_files(repo, repo_type="dataset")
             if f not in local and f != ".gitattributes"]
    if stale:
        from huggingface_hub import CommitOperationDelete

        api.create_commit(repo, repo_type="dataset", commit_message="Remove files no longer in the bundle",
                          operations=[CommitOperationDelete(path_in_repo=f) for f in stale])
    print(f"data: https://huggingface.co/datasets/{repo} ({len(local)} files, {len(stale)} stale removed)")


if __name__ == "__main__":
    main()
