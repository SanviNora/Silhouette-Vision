"""Evaluate encoders on LookBench (exact-item retrieval, Recall@K).

Usage:
  python scripts/eval_lookbench.py --models gr_lite marqo_fashion_siglip \
      --subsets real_studio_flat real_streetlook [--noise] [--limit 300]
"""

import argparse
import io
import json
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from silhouette_vision.config import ROOT, path
from silhouette_vision.encoders import load_encoder

LOOKBENCH = ROOT / "data/raw/lookbench/v20251201"
SUBSETS = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]


def read_split(file, limit=None):
    table = pq.read_table(file, columns=["image", "item_ID", "category"])
    if limit:
        table = table.slice(0, limit)
    rows = table.to_pylist()
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
    return images, np.array([r["item_ID"] for r in rows])


def embed(encoder, images, batch_size=32):
    out = [encoder.embed_images(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
    return np.concatenate(out)


def recall_at_k(q_emb, q_ids, g_emb, g_ids, ks=(1, 5, 10)):
    top = np.argsort(-(q_emb @ g_emb.T), axis=1)[:, : max(ks)]
    hits = g_ids[top] == q_ids[:, None]
    return {f"R@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gr_lite", "marqo_fashion_siglip"])
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--limit", type=int, default=None, help="max queries per subset")
    ap.add_argument("--noise", action="store_true", help="add the 58k shared distractors")
    args = ap.parse_args()

    data = {s: (read_split(LOOKBENCH / s / "query.parquet", args.limit),
                read_split(LOOKBENCH / s / "gallery.parquet")) for s in args.subsets}
    noise = None
    if args.noise:
        imgs, ids = [], []
        for f in sorted((LOOKBENCH / "noise").glob("*.parquet")):
            i, d = read_split(f)
            imgs += i
            ids.append(d)
        noise = (imgs, np.concatenate(ids))

    results = {}
    for name in args.models:
        encoder = load_encoder(name)
        noise_emb = embed(encoder, noise[0]) if noise else None
        for subset, ((q_img, q_ids), (g_img, g_ids)) in data.items():
            t = time.time()
            q_emb, g_emb = embed(encoder, q_img), embed(encoder, g_img)
            if noise:
                g_emb, g_ids = np.vstack([g_emb, noise_emb]), np.concatenate([g_ids, noise[1]])
            scores = recall_at_k(q_emb, q_ids, g_emb, g_ids)
            results[f"{name}/{subset}"] = scores
            print(f"{name:22s} {subset:18s} q={len(q_ids):5d} g={len(g_ids):6d} "
                  + " ".join(f"{k}={v:.3f}" for k, v in scores.items())
                  + f"  ({time.time() - t:.0f}s)", flush=True)

    out = path("reports") / f"lookbench{'_noise' if args.noise else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
