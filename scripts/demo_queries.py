"""Run a folder of query photos through search and save a result sheet per photo.

Usage: python scripts/demo_queries.py "<folder of photos>" [--k 8] [--source zooclaw]

Each sheet shows the query on the left and the top-k matches with brand, title, price and
similarity. Sheets go to artifacts/reports/demo_queries/ (git-ignored: they contain
third-party images).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from silhouette_vision.config import ROOT, path
from silhouette_vision.images import load_rgb
from silhouette_vision.search import Filters, SearchEngine

TILE_W, TILE_H, CAPTION_H = 220, 293, 64
CURRENCY = {"SGD": "S$", "INR": "Rs "}


def font(size: int):
    for name in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((TILE_W, TILE_H))
    canvas = Image.new("RGB", (TILE_W, TILE_H), "white")
    canvas.paste(img, ((TILE_W - img.width) // 2, (TILE_H - img.height) // 2))
    return canvas


def sheet(query: Image.Image, results, title: str) -> Image.Image:
    n = len(results) + 1
    out = Image.new("RGB", (n * (TILE_W + 8) + 8, TILE_H + CAPTION_H + 44), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill="black", font=font(18))
    y = 36
    out.paste(fit(query), (8, y))
    draw.rectangle([8, y, 8 + TILE_W, y + TILE_H], outline="#d33", width=3)
    draw.text((8, y + TILE_H + 4), "QUERY", fill="#d33", font=font(14))
    small = font(12)
    for i, row in enumerate(results.itertuples(), start=1):
        x = 8 + i * (TILE_W + 8)
        out.paste(fit(load_rgb(ROOT / row.image_path)), (x, y))
        price = "" if row.price != row.price else f"{CURRENCY.get(row.currency, '')}{row.price:,.0f}"
        tag = f"{row.source}{' · pre-owned' if row.is_preowned else ''}"
        lines = [f"{(row.brand or '')[:28]}", f"{(row.title or '')[:34]}", f"{price} · {tag}",
                 f"sim {row.similarity:.3f}"]
        for j, line in enumerate(lines):
            draw.text((x, y + TILE_H + 4 + j * 15), line, fill="black", font=small)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--source", choices=["zooclaw", "lookbench", "secondhand"], default=None)
    ap.add_argument("--colour", action="store_true", help="colour-aware re-ranking")
    ap.add_argument("--precise", action="store_true", help="Marqo + GR-Lite blend (Myntra only)")
    args = ap.parse_args()

    engine = SearchEngine()
    filters = Filters(sources=[args.source]) if args.source else None
    out_dir = path("reports") / "demo_queries"
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in Path(args.folder).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    embed_ms, search_ms = [], []
    for i, photo in enumerate(photos, start=1):
        query = load_rgb(photo)
        t0 = time.perf_counter()
        vec = engine.image_vector(query)
        t1 = time.perf_counter()
        colour = engine.image_colour(query) if args.colour else None
        precise = engine.precise_vector(query) if args.precise else None
        results = engine.search(vec, args.k, filters, query_colour=colour, precise_query=precise)
        t2 = time.perf_counter()
        embed_ms.append((t1 - t0) * 1000)
        search_ms.append((t2 - t1) * 1000)
        suffix = (f"_{args.source}" if args.source else "") + ("_colour" if args.colour else "") + ("_precise" if args.precise else "")
        name = f"query_{i:02d}{suffix}.jpg"
        sheet(query, results, f"Query {i}: {photo.name}").save(out_dir / name, quality=88)
        print(f"{name}: top = {results.brand.iloc[0]} | {results.title.iloc[0]} "
              f"({results.category.iloc[0]}, sim {results.score.iloc[0]:.3f})", flush=True)

    print(f"\nlatency per query: embed {np.median(embed_ms[1:] or embed_ms):.0f} ms, "
          f"search {np.median(search_ms):.0f} ms (median; first query excluded as warm-up)")
    print(f"sheets: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
