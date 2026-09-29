"""Stage 11: publish the demo data to Hugging Face (and, optionally, a Docker Space).

The hosted app runs on Streamlit Community Cloud (deploy/STREAMLIT_CLOUD.md) and downloads this
dataset on first start. Docker Spaces now need a Hugging Face PRO plan, hence --space.

Needs a Hugging Face write token: `huggingface-cli login` (or HF_TOKEN set). Creates, if missing:
  datasets/<user>/silhouette-vision-data  the bundle from scripts/10_build_public.py (public)
  spaces/<user>/silhouette-vision         Docker Space running app/ui.py in public mode (--space)

Usage: python scripts/11_publish.py [--user Sanvii] [--skip-data] [--space]
"""

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from silhouette_vision.config import ROOT


def stage_data(bundle: Path, staging: Path) -> Path:
    """The bundle with its ~47k thumbnails packed into one tar per source: uploading that many
    small files hits Hugging Face's rate limit (HTTP 429), and downloading them slows start-up.
    deploy/start.sh unpacks the archives."""
    if not bundle.exists():
        raise SystemExit("Build the bundle first: python scripts/10_build_public.py")
    thumbs = bundle / "data/processed/thumbs"
    staging.mkdir(exist_ok=True)
    for f in bundle.rglob("*"):
        rel = f.relative_to(bundle)
        if f.is_file() and not f.is_relative_to(thumbs) and rel.parts[0] != ".cache":
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            if not (staging / rel).exists():
                os.link(f, staging / rel)
    (staging / "archives").mkdir(exist_ok=True)
    for source in sorted(p.name for p in thumbs.iterdir()):
        archive = staging / "archives" / f"thumbs_{source}.tar"
        if not archive.exists():
            with tarfile.open(archive, "w") as tar:
                tar.add(thumbs / source, arcname=f"data/processed/thumbs/{source}")
            print(f"packed {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)")
    shutil.copy2(ROOT / "deploy/DATA_README.md", staging / "README.md")
    return staging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Sanvii")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--space", action="store_true", help="also deploy a Docker Space (needs PRO)")
    args = ap.parse_args()
    api = HfApi()
    data_repo, space_repo = f"{args.user}/silhouette-vision-data", f"{args.user}/silhouette-vision"

    if not args.skip_data:
        staging = stage_data(ROOT / "deploy_bundle", ROOT / "deploy_upload")
        api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(repo_id=data_repo, repo_type="dataset", folder_path=staging)
        print(f"data: https://huggingface.co/datasets/{data_repo}")

    if not args.space:
        return
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ["Dockerfile", "requirements.txt", "start.sh"]:
            shutil.copy2(ROOT / "deploy" / name, stage / name)
        shutil.copy2(ROOT / "deploy/SPACE_README.md", stage / "README.md")
        for folder in ["src", "app", "configs"]:
            shutil.copytree(ROOT / folder, stage / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        dockerfile = (stage / "Dockerfile").read_text()
        (stage / "Dockerfile").write_text(dockerfile.replace("Sanvii/silhouette-vision-data", data_repo))
        api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=stage,
                          commit_message="Deploy Silhouette Vision")
    print(f"app: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
