"""Catalog-level retrieval quality from the stored embeddings (no GPU needed).

For sampled query products, take the 10 nearest neighbours in the same source (excluding the
query) and measure how often they share a labelled attribute: precision@10. Chance level is
the probability that a random catalog item shares the attribute (sum of squared frequencies),
so "lift" = precision / chance shows how much the embedding knows beyond base rates.

Usage: python scripts/eval_catalog.py [--model marqo_fashion_siglip] [--queries 5000] [--legacy]
"""

import argparse
import json

import numpy as np
import pandas as pd

from silhouette_vision.catalog import load_catalog, load_legacy_catalog
from silhouette_vision.config import path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.search import top_k

K = 10
MYNTRA_ATTRS = ["Pattern", "Fabric", "Sleeve Length", "Fit", "Neck", "Material"]


def neighbours(emb: np.ndarray, queries: np.ndarray, k: int = K) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=int)
    for i, q in enumerate(queries):
        scores = emb @ emb[q]
        scores[q] = -np.inf
        out[i] = top_k(scores, k)
    return out


def precision(labels: pd.Series, queries: np.ndarray, nbrs: np.ndarray) -> dict | None:
    """Precision@K over queries whose label is known, only counting neighbours with a known label."""
    lab = labels.to_numpy(dtype=object)
    known = pd.notna(lab)
    rows = [i for i, q in enumerate(queries) if known[q]]
    if len(rows) < 50:
        return None
    q_lab = lab[queries[rows]][:, None]
    n_lab = lab[nbrs[rows]]
    mask = pd.notna(n_lab)
    if mask.sum() == 0:
        return None
    prec = ((n_lab == q_lab) & mask).sum() / mask.sum()
    freq = labels.dropna().value_counts(normalize=True)
    chance = float((freq**2).sum())
    return {"precision@10": float(prec), "chance": chance, "lift": float(prec / chance),
            "queries": len(rows), "classes": int(freq.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="marqo_fashion_siglip")
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--legacy", action="store_true", help="Myntra + Farfetch (Phases 1-4)")
    args = ap.parse_args()

    if args.legacy:
        catalog = load_legacy_catalog()
        emb = load_embeddings(f"{args.model}__legacy", catalog)
        sources = [("myntra", ["article_type", "category", "colour", "gender", *MYNTRA_ATTRS]),
                   ("farfetch", ["category", "brand", "is_preowned", "on_sale"])]
    else:
        catalog = load_catalog()
        emb = load_embeddings(args.model, catalog)
        sources = [("secondhand", ["item_type", "category", "pattern", "colour", "brand", "gender"]),
                   ("zooclaw", ["item_type", "brand", "gender"]),
                   ("abo", ["item_type", "brand", "gender"]),
                   ("lookbench", ["item_type"])]
    rng = np.random.default_rng(0)
    report = {}

    for source, fields in sources:
        idx = np.flatnonzero(catalog.source.values == source)
        sub, sub_emb = catalog.iloc[idx].reset_index(drop=True), emb[idx]
        if source == "myntra":
            attrs = sub.attributes.map(json.loads)
            for a in MYNTRA_ATTRS:
                sub[a] = attrs.map(lambda d, a=a: d.get(a))
        queries = rng.choice(len(sub), size=min(args.queries, len(sub)), replace=False)
        nbrs = neighbours(sub_emb, queries)
        print(f"\n{source}: {len(queries):,} queries against {len(sub):,} items")
        print(f"  {'label':14s} {'P@10':>6s} {'chance':>7s} {'lift':>6s} {'classes':>8s}")
        for field in fields:
            r = precision(sub[field], queries, nbrs)
            if r is None:
                continue
            report[f"{source}/{field}"] = r
            print(f"  {field:14s} {100 * r['precision@10']:5.1f}% {100 * r['chance']:6.1f}% "
                  f"{r['lift']:5.1f}x {r['classes']:8d}")

    out = path("reports") / f"catalog_eval_{args.model}{'_legacy' if args.legacy else ''}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out.name}")


if __name__ == "__main__":
    main()
