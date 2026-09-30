"""Text-to-image search on the ZooClaw-Fashion benchmark (2,000 queries with known products, 2026).

For each model: Recall@1/@10 and MRR, for short and long (LLM-written) queries, split into
zero-shot queries (products unseen by ZooClaw's own model) and in-domain ones. Two galleries:
the benchmark's 12k products (comparable to its published table) and our whole 51k catalog
(what the app searches).
Usage: python scripts/eval_zooclaw.py --model marqo_fashion_siglip [--model zooclaw_fashionsiglip2]
       [--zooclaw-only]   (embed/search only the 12k benchmark products: a quick first look)
"""

import argparse
import json

import numpy as np
import pandas as pd

from silhouette_vision.catalog import load_catalog
from silhouette_vision.config import ROOT, path
from silhouette_vision.embed import embed_catalog, load_embeddings
from silhouette_vision.encoders import load_encoder

ZC = ROOT / "data/raw/zooclaw"


def catalog_embeddings(name: str, catalog: pd.DataFrame, encoder) -> np.ndarray:
    try:
        return load_embeddings(name, catalog)
    except (FileNotFoundError, ValueError):
        return embed_catalog(encoder, catalog)


def metrics(sims: np.ndarray, truth: list[set[int]]) -> dict:
    order = np.argsort(-sims, 1)[:, :100]
    ranks = [next((r for r, j in enumerate(row) if j in t), None) for row, t in zip(order, truth, strict=True)]
    r = np.array([np.inf if x is None else x for x in ranks])
    return {"R@1": float((r < 1).mean()), "R@10": float((r < 10).mean()), "MRR": float(np.where(r < 100, 1 / (r + 1), 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--zooclaw-only", action="store_true")
    args = ap.parse_args()
    catalog = load_catalog()
    queries = pd.read_parquet(ZC / "queries.parquet")
    truth = pd.read_parquet(ZC / "ground_truth.parquet").set_index("query_id").corpus_ids
    row_of = pd.Series(np.arange(len(catalog)), index=catalog.item_id)
    gt = [{int(row_of[f"zc_{c}"]) for c in truth[q] if f"zc_{c}" in row_of} for q in queries.query_id]
    keep = np.array([len(t) > 0 for t in gt])
    queries, gt = queries[keep].reset_index(drop=True), [t for t in gt if t]
    zc_rows = np.flatnonzero(catalog.source.values == "zooclaw")
    local = {r: i for i, r in enumerate(zc_rows)}
    gt_zc = [{local[r] for r in t} for t in gt]
    report = {}
    for name in args.model:
        encoder = load_encoder(name)
        if args.zooclaw_only:
            sub = catalog.iloc[zc_rows].reset_index(drop=True)
            encoder.name = f"{name}__zooclaw"
            emb = np.zeros((len(catalog), encoder.dim), np.float32)
            emb[zc_rows] = catalog_embeddings(encoder.name, sub, encoder)
        else:
            emb = catalog_embeddings(name, catalog, encoder)
        for kind in ["short_query", "long_query"]:
            q = encoder.embed_texts(queries[kind].fillna("").tolist())
            galleries = [("zooclaw_12k", q @ emb[zc_rows].T, gt_zc)]
            if not args.zooclaw_only:
                galleries.append(("catalog_51k", q @ emb.T, gt))
                if name == "marqo_fashion_siglip":  # the app's hybrid text search
                    from silhouette_vision.search import SearchEngine

                    eng = SearchEngine.__new__(SearchEngine)
                    eng.catalog = catalog
                    kw = np.stack([eng.keyword_scores(t) for t in queries[kind].fillna("")])
                    galleries.append(("catalog_51k_hybrid", q @ emb.T + SearchEngine.KEYWORD_WEIGHT * kw, gt))
            for gallery, sims, t in galleries:
                for split in ["zero-shot", "in-domain"]:
                    m = (queries.query_type == split).values
                    res = metrics(sims[m], [x for x, k in zip(t, m, strict=True) if k])
                    report.setdefault(name, {}).setdefault(kind, {}).setdefault(gallery, {})[split] = res
                    print(f"{name:24s} {kind:11s} {gallery:11s} {split:9s} R@1 {100 * res['R@1']:5.1f}  "
                          f"R@10 {100 * res['R@10']:5.1f}  MRR {res['MRR']:.3f}", flush=True)
    print(f"({len(queries)} queries with their product in the catalog)")
    out = path("reports") / "eval_zooclaw.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps(old | report, indent=2))


if __name__ == "__main__":
    main()
