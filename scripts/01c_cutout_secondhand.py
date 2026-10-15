"""Stage 1c: cut Second-Hand garments out of the sorting-table photos (BiRefNet, October 2026).

Writes data/processed/cutouts/secondhand/<source_id>.jpg (garment on white, square, up to 1024 px)
for every Second-Hand product in the catalog. Resumable: one file per photo, existing files are
skipped, so an interrupted overnight run continues where it stopped. A photo whose mask looks
wrong (under 3% or over 90% of the frame) gets no file and keeps its old thumbnail.
Then: python scripts/01_build_catalog.py --rethumb secondhand  and  02_embed.py --update
--refresh secondhand for each model (scripts 03-06 after that).
Usage: python scripts/01c_cutout_secondhand.py [--limit N]
"""

import argparse
import time

from silhouette_vision.catalog import load_catalog
from silhouette_vision.config import ROOT
from silhouette_vision.cutout import GarmentCutout
from silhouette_vision.images import load_rgb

OUT = ROOT / "data/processed/cutouts/secondhand"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    catalog = load_catalog()
    sh = catalog[catalog.source == "secondhand"]
    OUT.mkdir(parents=True, exist_ok=True)
    failed_log = OUT.parent / "secondhand_failed.txt"
    failed = set(failed_log.read_text().split()) if failed_log.exists() else set()
    todo = [(sid, raw) for sid, raw in zip(sh.source_id, sh.raw_image_path, strict=True)
            if not (OUT / f"{sid}.jpg").exists() and sid not in failed]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(sh):,} second-hand products, {len(todo):,} to cut out", flush=True)
    cut, t0 = GarmentCutout(), time.time()
    for n, (sid, raw) in enumerate(todo, 1):
        out = cut(load_rgb(ROOT / raw).rotate(90, expand=True))
        if out is None:
            failed.add(sid)
            with failed_log.open("a") as f:
                f.write(sid + "\n")
        else:
            out.thumbnail((1024, 1024))
            out.save(OUT / f"{sid}.jpg", quality=92)
        if n % 200 == 0 or n == len(todo):
            rate = n / (time.time() - t0)
            print(f"{n:,}/{len(todo):,}  {rate:.2f} photos/s  ~{(len(todo) - n) / rate / 3600:.1f} h left  "
                  f"failed so far {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
