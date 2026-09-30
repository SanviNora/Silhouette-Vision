"""Stage 10: the app's data bundle, same layout as the repo's data/ and artifacts/.

A fresh host (Streamlit Community Cloud) downloads it from Hugging Face on first start
(silhouette_vision.bootstrap). Everything in it may be shared:
  Search catalog (ZooClaw-Fashion CC BY-NC 4.0, Second-Hand Fashion CC BY 4.0, LookBench
  Apache-2.0, Amazon Berkeley Objects CC BY 4.0): catalog rows, 384 px thumbnails, embeddings,
  colour histograms, attribute predictions, style clusters
  Visuelle 2.0 (CC BY-NC-SA 4.0): 384 px thumbnails, image embeddings, the demand model
  Models trained by us: attribute heads, calibrators
Not included: the legacy Myntra/Farfetch data (training and analysis only).

Usage: python scripts/10_build_bundle.py [--out deploy_bundle] [--side 384]
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from silhouette_vision.bootstrap import BUNDLE_VERSION
from silhouette_vision.config import ROOT
from silhouette_vision.images import load_rgb

FILES = [
    "data/processed/catalog.parquet",
    "data/processed/attribute_predictions.parquet",
    "data/processed/style_clusters.parquet",
    "artifacts/embeddings/marqo_fashion_siglip/embeddings.npy",
    "artifacts/embeddings/marqo_fashion_siglip/item_ids.parquet",
    "artifacts/embeddings/colour/hists.npy",
    "artifacts/embeddings/colour/item_ids.parquet",
    "artifacts/embeddings/visuelle/marqo_fashion_siglip.npy",
    "artifacts/models/attributes.joblib",
    "artifacts/models/match_confidence.json",
    "artifacts/models/model_recognition.json",
    "artifacts/models/demand.joblib",
    "artifacts/reports/style_clusters.json",
]
VIS_IMAGES = ROOT / "data/raw/visuelle2/visuelle2/images"


def thumb(src: Path, dst: Path, side: int) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = load_rgb(src)
    except OSError:  # Visuelle AI19/04442.png is truncated
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = load_rgb(src)
    img.thumbnail((side, side))
    img.save(dst, "JPEG", quality=85, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deploy_bundle"))
    ap.add_argument("--side", type=int, default=384)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for rel in FILES:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out / rel)

    catalog = pd.read_parquet(ROOT / "data/processed/catalog.parquet")
    vis = pd.read_parquet(ROOT / "data/processed/visuelle_products.parquet")
    jobs = [(ROOT / p, out / p, args.side) for p in catalog.image_path]
    jobs += [(VIS_IMAGES / p, out / "data/processed/thumbs/visuelle" / Path(p).with_suffix(".jpg"), args.side)
             for p in vis.image_path]
    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(lambda j: thumb(*j), jobs)):
            if (i + 1) % 10000 == 0:
                print(f"  thumbnails {i + 1:,}/{len(jobs):,}", flush=True)
    (out / "VERSION").write_text(BUNDLE_VERSION + "\n")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle {BUNDLE_VERSION}: {out} ({size / 1e9:.2f} GB, {len(jobs):,} thumbnails)")


if __name__ == "__main__":
    main()
