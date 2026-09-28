"""Evaluate encoders on LookBench (Recall@K under three definitions of a hit).

  exact  - the retrieved item is the same product (same item_ID)
  fine   - same category and same main attribute: the LookBench paper's "fine Recall",
           which this reproduces within ~1 point of the published numbers
  coarse - same category only

Usage:
  python scripts/eval_lookbench.py --models gr_lite marqo_fashion_siglip \
      --subsets real_studio_flat real_streetlook [--noise] [--limit 300]

Only compressed image bytes are held in memory; each batch is decoded by threads just before
embedding (decoding all ~69k benchmark images up front needs tens of GB).
Embeddings are cached per model and split in artifacts/embeddings/lookbench/<model>/, so the
58k shared distractors are embedded once per model and cached runs can come from another
machine (e.g. a Colab GPU).
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from silhouette_vision.config import ROOT, path
from silhouette_vision.encoders import load_encoder

LOOKBENCH = ROOT / "data/raw/lookbench/v20251201"
SUBSETS = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]


LABELS = ["item_ID", "category", "main_attribute"]


def read_split(files, limit=None) -> tuple[list[bytes], dict[str, np.ndarray]]:
    blobs, meta = [], {c: [] for c in LABELS}
    for file in files:
        table = pq.read_table(file, columns=["image", *LABELS])
        blobs += [img["bytes"] for img in table.column("image").to_pylist()]
        for c in LABELS:
            meta[c] += table.column(c).to_pylist()
    if limit:
        blobs, meta = blobs[:limit], {c: v[:limit] for c, v in meta.items()}
    return blobs, {c: np.array(v, dtype=object) for c, v in meta.items()}


def concat_meta(a: dict, b: dict) -> dict:
    return {c: np.concatenate([a[c], b[c]]) for c in LABELS}


def _batches(blobs, transform, batch_size, pool):
    """Decode batch i+1 in threads while the model runs on batch i (one batch prefetched)."""
    def decode(chunk):
        return torch.stack(list(pool.map(
            lambda b: transform(Image.open(io.BytesIO(b)).convert("RGB")), chunk)))
    starts = range(0, len(blobs), batch_size)
    with ThreadPoolExecutor(1) as prefetch:
        pending = None
        for s in starts:
            nxt = prefetch.submit(decode, blobs[s : s + batch_size])
            if pending is not None:
                yield pending.result()
            pending = nxt
        if pending is not None:
            yield pending.result()


def cached_embed(encoder, name: str, blobs: list[bytes], batch_size=32, threads=6,
                 chunk=4096) -> np.ndarray:
    """Embed with a cache; long runs are checkpointed every `chunk` images, so a crash or the
    laptop sleeping only loses the current chunk (the overnight GR-Lite run had no such safety)."""
    cache = path("embeddings") / "lookbench" / encoder.name / f"{name}.npy"
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(blobs):
            return emb.astype(np.float32)
    parts_dir = cache.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    # Threads, not DataLoader workers: macOS worker processes would each get a pickled copy
    # of every image's bytes (~2 GB for the distractors) and push the machine into swap.
    with ThreadPoolExecutor(threads) as pool:
        for c, start in enumerate(range(0, len(blobs), chunk)):
            part = parts_dir / f"{c:04d}_{len(blobs)}.npy"
            if not part.exists():
                chunk_blobs = blobs[start : start + chunk]
                emb = np.concatenate([encoder.embed_pixels(b) for b in
                                      _batches(chunk_blobs, encoder.transform, batch_size, pool)])
                np.save(part, emb.astype(np.float16))
                print(f"    {name}: {min(start + chunk, len(blobs))}/{len(blobs)}", flush=True)
            parts.append(np.load(part))
    emb = np.concatenate(parts)
    np.save(cache, emb)
    for part in parts_dir.glob("*.npy"):
        part.unlink()
    parts_dir.rmdir()
    return emb.astype(np.float32)


def recall_at_k(q_emb, q_meta, g_emb, g_meta, ks=(1, 5, 10)) -> dict[str, float]:
    scores = q_emb @ g_emb.T
    top = np.argpartition(-scores, max(ks), axis=1)[:, : max(ks)]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)

    def same(col):
        return g_meta[col][top] == q_meta[col][:, None]

    hits = {"exact": same("item_ID"),
            "fine": same("category") & same("main_attribute"),
            "coarse": same("category")}
    return {f"{kind} R@{k}": float(h[:, :k].any(axis=1).mean())
            for kind, h in hits.items() for k in ks}


def summary(scores: dict) -> str:
    """exact R@1/5/10 | fine R@1 | coarse R@1"""
    e = "/".join(f"{100 * scores[f'exact R@{k}']:.1f}" for k in (1, 5, 10))
    return f"exact R@1/5/10={e}  fine R@1={100 * scores['fine R@1']:.1f}  coarse R@1={100 * scores['coarse R@1']:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gr_lite", "marqo_fashion_siglip"])
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--limit", type=int, default=None, help="max queries per subset")
    ap.add_argument("--noise", action="store_true", help="add the 58k shared distractors")
    ap.add_argument("--out", default=None, help="report file name in artifacts/reports/")
    args = ap.parse_args()

    results = {}
    for name in args.models:
        encoder = load_encoder(name)
        noise = None
        if args.noise:
            blobs, meta = read_split(sorted((LOOKBENCH / "noise").glob("*.parquet")))
            t = time.time()
            noise = (cached_embed(encoder, "noise", blobs), meta)
            print(f"{name:22s} noise embedded: {len(blobs)} ({time.time() - t:.0f}s)", flush=True)
        for subset in args.subsets:
            t = time.time()
            q_blobs, q_meta = read_split([LOOKBENCH / subset / "query.parquet"], args.limit)
            g_blobs, g_meta = read_split([LOOKBENCH / subset / "gallery.parquet"])
            suffix = f"_first{args.limit}" if args.limit else ""
            q_emb = cached_embed(encoder, f"{subset}_query{suffix}", q_blobs)
            g_emb = cached_embed(encoder, f"{subset}_gallery", g_blobs)
            if noise:
                g_emb, g_meta = np.vstack([g_emb, noise[0]]), concat_meta(g_meta, noise[1])
            scores = recall_at_k(q_emb, q_meta, g_emb, g_meta)
            n_q, n_g = len(q_blobs), len(g_emb)
            results[f"{name}/{subset}"] = {"queries": n_q, "gallery": n_g, **scores}
            print(f"{name:22s} {subset:18s} q={n_q:5d} g={n_g:6d} " + summary(scores)
                  + f"  ({time.time() - t:.0f}s)", flush=True)
        n = sum(results[f"{name}/{s}"]["queries"] for s in args.subsets)
        metric_names = [k for k in results[f"{name}/{args.subsets[0]}"] if "R@" in k]
        overall = {k: sum(results[f"{name}/{s}"][k] * results[f"{name}/{s}"]["queries"]
                          for s in args.subsets) / n for k in metric_names}
        results[f"{name}/overall"] = {"queries": n, **overall}
        print(f"{name:22s} {'OVERALL':18s} q={n:5d}          " + summary(overall), flush=True)

    out = path("reports") / (args.out or f"lookbench{'_noise' if args.noise else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
