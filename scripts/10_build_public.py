"""Stage 10: the public deploy bundle (same layout as the repo's data/ and artifacts/).

Run the app on it with SILHOUETTE_DATA_ROOT=<bundle> SILHOUETTE_PUBLIC=1.

What goes in, by license:
  Myntra (Fashion Product Images, MIT)   catalog rows, 320 px thumbnails, embeddings (Marqo,
                                         GR-Lite, colour), attribute predictions
  Visuelle 2.0 (CC BY-NC-SA 4.0)         320 px thumbnails, embeddings, the demand model
  Farfetch (scraped, no redistribution)  nothing item-level: style-cluster aggregates
                                         (report JSON) and anonymous 2-D map positions only
  Models trained by us                   attribute heads, calibrators, demand model

Usage: python scripts/10_build_public.py [--out deploy_bundle] [--side 320]
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from silhouette_vision.config import ROOT
from silhouette_vision.images import load_rgb

VIS_IMAGES = ROOT / "data/raw/visuelle2/visuelle2/images"


def thumb(src: Path, dst: Path, side: int) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = load_rgb(src, max_side=side)
    except OSError:  # AI19/04442.png is truncated
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = load_rgb(src, max_side=side)
    img.thumbnail((side, side))
    img.save(dst, "JPEG", quality=82, optimize=True)


def copy(rel: str, out: Path) -> None:
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, out / rel)


def subset_embeddings(rel_dir: str, n: int, out: Path) -> None:
    """First n rows (Myntra) of an embedding folder, stored as float16 (loaded back as float32)."""
    src, dst = ROOT / rel_dir, out / rel_dir
    dst.mkdir(parents=True, exist_ok=True)
    name = "hists.npy" if (src / "hists.npy").exists() else "embeddings.npy"
    arr = np.load(src / name)[:n]
    np.save(dst / name, arr.astype(np.float16) if name == "embeddings.npy" else arr)
    pd.read_parquet(src / "item_ids.parquet").iloc[:n].to_parquet(dst / "item_ids.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deploy_bundle"))
    ap.add_argument("--side", type=int, default=320)
    args = ap.parse_args()
    out = Path(args.out)

    catalog = pd.read_parquet(ROOT / "data/processed/catalog.parquet")
    myntra = catalog[catalog.source == "myntra"]
    n = len(myntra)
    assert (catalog.source.values[:n] == "myntra").all(), "Myntra rows must come first"
    (out / "data/processed").mkdir(parents=True, exist_ok=True)
    myntra.to_parquet(out / "data/processed/catalog.parquet", index=False)
    preds = pd.read_parquet(ROOT / "data/processed/attribute_predictions.parquet")
    preds[preds.item_id.str.startswith("myn_")].to_parquet(
        out / "data/processed/attribute_predictions.parquet", index=False)
    print(f"catalog: {n:,} Myntra products")

    for rel in ["artifacts/embeddings/marqo_fashion_siglip", "artifacts/embeddings/colour"]:
        subset_embeddings(rel, n, out)
    shutil.copytree(ROOT / "artifacts/embeddings/gr_lite__myntra", out / "artifacts/embeddings/gr_lite__myntra",
                    ignore=shutil.ignore_patterns("chunks"), dirs_exist_ok=True)
    for rel in ["artifacts/models/attributes.joblib", "artifacts/models/match_confidence.json",
                "artifacts/models/model_recognition.json", "artifacts/models/demand.joblib",
                "artifacts/embeddings/visuelle/marqo_fashion_siglip.npy",
                "artifacts/reports/style_clusters.json"]:
        copy(rel, out)

    clusters = pd.read_parquet(ROOT / "data/processed/style_clusters.parquet")
    anonymous = (clusters.sample(frac=1, random_state=0).groupby("category").head(6000)
                 [["category", "cluster_name", "map_x", "map_y"]])
    anonymous.to_parquet(out / "data/processed/style_clusters.parquet", index=False)

    vis = pd.read_parquet(ROOT / "data/processed/visuelle_products.parquet")
    jobs = [(ROOT / p, out / p, args.side) for p in myntra.image_path]
    jobs += [(VIS_IMAGES / p, out / "data/processed/thumbs/visuelle" / Path(p).with_suffix(".jpg"), args.side)
             for p in vis.image_path]
    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(lambda j: thumb(*j), jobs)):
            if (i + 1) % 5000 == 0:
                print(f"  thumbnails {i + 1:,}/{len(jobs):,}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle: {out} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
