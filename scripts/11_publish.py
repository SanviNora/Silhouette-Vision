"""Stage 11: publish the demo to Hugging Face (dataset repo for data, Space for the app).

Needs a Hugging Face write token: `huggingface-cli login` (or HF_TOKEN set). Creates, if missing:
  datasets/<user>/silhouette-vision-data  the bundle from scripts/10_build_public.py (public)
  spaces/<user>/silhouette-vision         Docker Space running app/ui.py in public mode

Usage: python scripts/11_publish.py [--user SanviNora] [--skip-data]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from silhouette_vision.config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="SanviNora")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    data_repo, space_repo = f"{args.user}/silhouette-vision-data", f"{args.user}/silhouette-vision"

    if not args.skip_data:
        bundle = ROOT / "deploy_bundle"
        if not bundle.exists():
            raise SystemExit("Build the bundle first: python scripts/10_build_public.py")
        shutil.copy2(ROOT / "deploy/DATA_README.md", bundle / "README.md")
        api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(repo_id=data_repo, repo_type="dataset", folder_path=bundle)
        print(f"data: https://huggingface.co/datasets/{data_repo}")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ["Dockerfile", "requirements.txt", "start.sh"]:
            shutil.copy2(ROOT / "deploy" / name, stage / name)
        shutil.copy2(ROOT / "deploy/SPACE_README.md", stage / "README.md")
        for folder in ["src", "app", "configs"]:
            shutil.copytree(ROOT / folder, stage / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        dockerfile = (stage / "Dockerfile").read_text()
        (stage / "Dockerfile").write_text(dockerfile.replace("SanviNora/silhouette-vision-data", data_repo))
        api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=stage,
                          commit_message="Deploy Silhouette Vision")
    print(f"app: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
