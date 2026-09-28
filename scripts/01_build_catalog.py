"""Stage 1: build data/processed/catalog.parquet and Myntra thumbnails.

Myntra photos are up to 1800x2400; a 512 px copy is written once so embedding and the app
never decode the full-size files again. Farfetch images are already 300x400 and used as-is.
"""

from concurrent.futures import ProcessPoolExecutor

from silhouette_vision.catalog import build_catalog
from silhouette_vision.config import ROOT
from silhouette_vision.images import load_rgb

THUMB_SIDE = 512


def make_thumb(paths: tuple[str, str]) -> bool:
    src, dst = ROOT / paths[0], ROOT / paths[1]
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    load_rgb(src, max_side=THUMB_SIDE).save(dst, quality=90)
    return True


def main():
    catalog = build_catalog()
    myntra = catalog[catalog.source == "myntra"]
    jobs = list(zip(myntra.raw_image_path, myntra.image_path))
    with ProcessPoolExecutor() as pool:
        made = sum(pool.map(make_thumb, jobs, chunksize=256))
    print(f"thumbnails: {made} new, {len(jobs) - made} existing")

    out = ROOT / "data/processed/catalog.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out, index=False)
    print(f"catalog: {len(catalog):,} items -> {out.relative_to(ROOT)}")
    print(catalog.groupby(["source", "category"]).size().unstack(0).fillna(0).astype(int))


if __name__ == "__main__":
    main()
