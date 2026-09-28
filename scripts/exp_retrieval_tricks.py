"""Experiments: can cheap, model-only tricks improve retrieval on top of cached embeddings?

Everything here uses only model outputs (never LookBench's gallery labels, which would make the
58k "noise data" distractors trivially separable). Hyperparameters are tuned on even-indexed
queries (dev) and reported on odd-indexed queries (test), per subset.

Tricks:
  cat   - soft category agreement: score += beta * <p_query, p_gallery>, where p is the model's
          zero-shot category distribution from text prompts
  dba   - database-side augmentation: each gallery vector averaged with its top-k neighbours
  aqe   - alpha query expansion: query averaged with its top-k retrieved gallery vectors
  flip  - test-time augmentation: average the query embedding with its horizontal flip

Usage: python scripts/exp_retrieval_tricks.py [--model marqo_fashion_siglip] [--flip]
"""

import argparse
import io
import json

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from silhouette_vision.config import ROOT, path
from silhouette_vision.encoders import load_encoder

LB = ROOT / "data/raw/lookbench/v20251201"
SUBSETS = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]
# Coarse garment vocabulary for zero-shot category prediction (prompted as "a photo of a ...").
CATEGORIES = ["coat", "jacket", "blazer", "vest", "cape", "sweater", "cardigan", "sweatshirt",
              "hoodie", "blouse", "shirt", "t-shirt", "top", "dress", "jumpsuit", "skirt", "pants",
              "jeans", "shorts", "shoe", "sneaker", "boot", "sandal", "high heel", "bag", "handbag",
              "tote bag", "backpack", "scarf", "hat", "glove", "tie", "belt", "sock", "glasses",
              "umbrella", "headband", "watch", "jewelry"]


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load(model, subset):
    e = path("embeddings") / "lookbench" / model
    q = np.load(e / f"{subset}_query.npy").astype(np.float32)
    g = np.vstack([np.load(e / f"{subset}_gallery.npy"), np.load(e / "noise.npy")]).astype(np.float32)
    q_ids = np.array(_ids(LB / subset / "query.parquet"), dtype=object)
    g_files = [LB / subset / "gallery.parquet", *sorted((LB / "noise").glob("*.parquet"))]
    g_ids = np.array([i for f in g_files for i in _ids(f)], dtype=object)
    return q, g, q_ids, g_ids


def _ids(file):
    return pq.read_table(file, columns=["item_ID"]).column("item_ID").to_pylist()


def recall(scores, q_ids, g_ids, ks=(1, 5, 10)):
    top = np.argpartition(-scores, max(ks), axis=1)[:, : max(ks)]
    top = np.take_along_axis(top, np.take_along_axis(scores, top, 1).argsort(1)[:, ::-1], 1)
    hits = g_ids[top] == q_ids[:, None]
    return {k: float(hits[:, :k].any(1).mean()) for k in ks}


def category_probs(emb, text_emb, temperature=100.0):
    logits = temperature * emb @ text_emb.T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def dba(g, k):
    """Replace each gallery vector by the mean of itself and its k nearest neighbours."""
    out = np.empty_like(g)
    for s in range(0, len(g), 4096):
        sims = g[s : s + 4096] @ g.T
        nn = np.argpartition(-sims, k, axis=1)[:, : k + 1]  # includes itself
        out[s : s + 4096] = g[nn].mean(1)
    return normalize(out)


def aqe(q, g, k, alpha=3.0):
    sims = q @ g.T
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(sims, nn, 1), 0, None) ** alpha
    return normalize(q + (w[..., None] * g[nn]).sum(1))


def flipped_query_embeddings(encoder, subset):
    table = pq.read_table(LB / subset / "query.parquet", columns=["image"])
    imgs = [ImageOps.mirror(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))
            for r in table.column("image").to_pylist()]
    return np.concatenate([encoder.embed_images(imgs[i : i + 32]) for i in range(0, len(imgs), 32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="marqo_fashion_siglip")
    ap.add_argument("--flip", action="store_true", help="also test flip TTA (re-embeds queries on CPU)")
    args = ap.parse_args()

    encoder = load_encoder(args.model, device="cpu")
    text = encoder.embed_texts([f"a photo of a {c}" for c in CATEGORIES])
    report = {}
    for subset in SUBSETS:
        q, g, q_ids, g_ids = load(args.model, subset)
        dev, test = np.arange(0, len(q), 2), np.arange(1, len(q), 2)
        pq_, pg = category_probs(q, text), category_probs(g, text)
        base = q @ g.T

        def evaluate(scores, idx, q_ids=q_ids, g_ids=g_ids):
            return recall(scores[idx], q_ids[idx], g_ids)

        rows = {"baseline": (evaluate(base, dev), evaluate(base, test), None)}

        # Category agreement: tune beta on dev.
        cat_scores = pq_ @ pg.T
        best = max([0.02, 0.05, 0.1, 0.2], key=lambda b: evaluate(base + b * cat_scores, dev)[1])
        rows["cat"] = (evaluate(base + best * cat_scores, dev),
                       evaluate(base + best * cat_scores, test), {"beta": best})

        # DBA and alpha-QE: tune k on dev.
        dbas = {k: dba(g, k) for k in (1, 2)}
        k_d = max(dbas, key=lambda k: evaluate(q @ dbas[k].T, dev)[1])
        rows["dba"] = (evaluate(q @ dbas[k_d].T, dev), evaluate(q @ dbas[k_d].T, test), {"k": k_d})
        k_q = max((1, 2, 3), key=lambda k: evaluate(aqe(q, g, k) @ g.T, dev)[1])
        qa = aqe(q, g, k_q)
        rows["aqe"] = (evaluate(qa @ g.T, dev), evaluate(qa @ g.T, test), {"k": k_q})

        if args.flip:
            qf = normalize(q + flipped_query_embeddings(encoder, subset))
            rows["flip"] = (evaluate(qf @ g.T, dev), evaluate(qf @ g.T, test), None)

        print(f"\n{subset} (exact-item Recall, test half = {len(test)} queries)")
        for name, (_, t, params) in rows.items():
            delta = "" if name == "baseline" else f"   Δ R@1 {100 * (t[1] - rows['baseline'][1][1]):+.1f}"
            print(f"  {name:9s} R@1/5/10 = {100 * t[1]:.1f}/{100 * t[5]:.1f}/{100 * t[10]:.1f}"
                  f"{delta}  {params or ''}", flush=True)
        report[subset] = {n: {"test": t, "params": p} for n, (_, t, p) in rows.items()}

    out = path("reports") / f"exp_retrieval_tricks_{args.model}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out.name}")


if __name__ == "__main__":
    main()
