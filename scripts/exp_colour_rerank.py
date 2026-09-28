"""Experiment: does colour-aware re-ranking fix the colour weakness without hurting type matching?

Colour descriptor: silhouette_vision.colour.colour_hist (CIELAB histogram of non-background
pixels). Re-ranking: take the
top-50 embedding neighbours and re-score with  cosine + gamma * histogram_intersection.
gamma is tuned on half the queries and reported on the other half, on Myntra's colour labels
(the weakness) and article-type labels (must not get worse).

Usage: python scripts/exp_colour_rerank.py [--queries 4000]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from silhouette_vision.catalog import load_catalog
from silhouette_vision.colour import colour_hist
from silhouette_vision.config import ROOT, path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.search import top_k

SHORTLIST = 50
K = 10


def hist_for(rel_path: str) -> np.ndarray:
    return colour_hist(Image.open(ROOT / rel_path))


def precision_at_k(labels, queries, ranked):
    lab = labels
    q = lab[queries][:, None]
    n = lab[ranked]
    known = pd.notna(n)
    return float(((n == q) & known).sum() / known.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=4000)
    args = ap.parse_args()

    catalog = load_catalog()
    emb = load_embeddings("marqo_fashion_siglip", catalog)
    idx = np.flatnonzero(catalog.source.values == "myntra")
    sub, sub_emb = catalog.iloc[idx].reset_index(drop=True), emb[idx]

    cache = path("embeddings") / "colour_hist_myntra.npy"
    if cache.exists():
        hists = np.load(cache)
    else:
        with ProcessPoolExecutor(6) as pool:
            hists = np.stack(list(pool.map(hist_for, sub.image_path, chunksize=256)))
        np.save(cache, hists)

    rng = np.random.default_rng(0)
    queries = rng.choice(len(sub), args.queries, replace=False)
    shortlists = []
    for q in queries:
        s = sub_emb @ sub_emb[q]
        s[q] = -np.inf
        shortlists.append(top_k(s, SHORTLIST))
    shortlists = np.array(shortlists)
    cos = np.take_along_axis(sub_emb[queries] @ sub_emb.T, shortlists, 1)
    inter = np.minimum(hists[queries][:, None, :], hists[shortlists]).sum(-1)

    colour = sub.colour.to_numpy(dtype=object)
    article = sub.article_type.to_numpy(dtype=object)
    dev, test = np.arange(0, len(queries), 2), np.arange(1, len(queries), 2)

    def evaluate(gamma, rows):
        order = np.argsort(-(cos[rows] + gamma * inter[rows]), axis=1)[:, :K]
        ranked = np.take_along_axis(shortlists[rows], order, 1)
        return (precision_at_k(colour, queries[rows], ranked),
                precision_at_k(article, queries[rows], ranked))

    gammas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    # Pick the gamma with the best colour precision that costs < 1 point of type precision.
    base_type = evaluate(0.0, dev)[1]
    ok = [g for g in gammas if evaluate(g, dev)[1] >= base_type - 0.01]
    best = max(ok, key=lambda g: evaluate(g, dev)[0])

    print(f"Myntra, {len(test)} test queries, precision@{K} (colour / article type)")
    report = {}
    for g in gammas:
        c, a = evaluate(g, test)
        report[g] = {"colour": c, "article_type": a}
        mark = "  <- chosen on dev" if g == best else ""
        print(f"  gamma {g:<4}  colour {100 * c:5.1f}%   type {100 * a:5.1f}%{mark}")
    (path("reports") / "exp_colour_rerank.json").write_text(json.dumps(
        {"chosen_gamma": best, "test": report}, indent=2))


if __name__ == "__main__":
    main()
