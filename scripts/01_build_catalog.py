"""Stage 1: build the search catalog (data/processed/catalog.parquet) and its 512 px thumbnails.

Sources (all 2022-2026, see catalog.py): ZooClaw-Fashion, LookBench's real studio gallery, and
Second-Hand Fashion. LookBench photos are stored inside its parquet files and are extracted
first. Second-Hand photos show garments lying sideways on a sorting table: they are rotated
upright and centre-cropped (images.secondhand_view; chosen in scripts/exp_secondhand_view.py).

Usage: python scripts/01_build_catalog.py [--legacy]   (--legacy rebuilds Myntra + Farfetch)
"""

import argparse
import io
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

from silhouette_vision.catalog import LOOKBENCH_DIR, build_catalog, build_legacy_catalog
from silhouette_vision.config import ROOT
from silhouette_vision.images import load_rgb, secondhand_view

THUMB_SIDE = 512


def make_thumb(job: tuple[str, str, str]) -> bool:
    source, src, dst = job
    src, dst = ROOT / src, ROOT / dst
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = load_rgb(src)
    if source == "secondhand":
        img = secondhand_view(img)
    img.thumbnail((THUMB_SIDE, THUMB_SIDE))
    img.save(dst, quality=90)
    return True


def extract_lookbench(catalog) -> None:
    """Write the chosen LookBench gallery photos to data/raw/lookbench/.../images/."""
    lb = catalog[catalog.source == "lookbench"]
    out = ROOT / LOOKBENCH_DIR / "images"
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f"{i}.jpg").exists() for i in lb.source_id):
        return
    images = pq.read_table(ROOT / LOOKBENCH_DIR / "real_studio_flat/gallery.parquet", columns=["image"]).column(0)
    for sid, row in zip(lb.source_id, lb.gallery_row.astype(int), strict=True):
        Image.open(io.BytesIO(images[row].as_py()["bytes"])).convert("RGB").save(out / f"{sid}.jpg", quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true")
    args = ap.parse_args()
    if args.legacy:
        legacy = build_legacy_catalog()
        legacy.to_parquet(ROOT / "data/processed/legacy_catalog.parquet", index=False)
        print(f"legacy catalog: {len(legacy):,} items")
        return

    catalog = build_catalog()
    extract_lookbench(catalog)
    jobs = list(zip(catalog.source, catalog.raw_image_path, catalog.image_path, strict=True))
    with ProcessPoolExecutor() as pool:
        made = sum(pool.map(make_thumb, jobs, chunksize=128))
    print(f"thumbnails: {made:,} new, {len(jobs) - made:,} existing")

    # A few source photos are thin slivers (bad crops upstream): drop anything beyond 3:1.
    ratio = [max(w / h, h / w) for w, h in (Image.open(ROOT / p).size for p in catalog.image_path)]
    odd = [r > 3 for r in ratio]
    print(f"dropped {sum(odd)} photos with aspect ratio > 3:1")
    catalog = catalog[[not o for o in odd]].reset_index(drop=True)

    out = ROOT / "data/processed/catalog.parquet"
    catalog.drop(columns=["gallery_row"]).to_parquet(out, index=False)
    print(f"catalog: {len(catalog):,} items -> {out.relative_to(ROOT)}")
    print(catalog.groupby(["category", "source"]).size().unstack(1).fillna(0).astype(int))
    print("years:", catalog.year.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
