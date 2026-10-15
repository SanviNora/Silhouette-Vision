"""Stage 2b: remove duplicate photos from the search catalog (October 2026).

2.1% of products had a twin at cosine >= 0.99 (practically the same image listed twice: a
second-hand garment registered twice, a LookBench image under two names, a ZooClaw listing
repeated). Duplicates waste a result slot and, worse, hide exact matches: when the exact product
is listed twice, its #1 and #2 scores are equal, so the exact-match calibrator (which rewards a
result that stands out) reports low confidence. Pairs must be >= THRESHOLD in BOTH embedding
spaces (MODA and Marqo), so lookalike products that one model happens to merge are kept.
One item per group is kept: one with a brand, then the longest title, then source priority.

Output: data/processed/catalog.parquet (rewritten) and data/processed/duplicates.parquet
(removed_id -> kept_id; scripts/01_build_catalog.py re-applies it, 06 maps benchmark targets).
Then re-run: 02_embed.py --update for each model (drops the rows, no re-embedding), 03-06.
Usage: python scripts/02b_dedupe.py [--threshold 0.99]
"""

import argparse

import numpy as np
import pandas as pd

from silhouette_vision.catalog import load_catalog
from silhouette_vision.config import PHOTO_MODEL, ROOT, TEXT_MODEL
from silhouette_vision.embed import load_embeddings

SOURCE_PRIORITY = {"zooclaw": 0, "abo": 1, "lookbench": 2, "secondhand": 3}


def twin_pairs(emb: np.ndarray, threshold: float, chunk: int = 4096) -> set[tuple[int, int]]:
    pairs = set()
    for i in range(0, len(emb), chunk):
        s = emb[i:i + chunk] @ emb.T
        for a, b in zip(*np.nonzero(s >= threshold), strict=True):
            a = int(a) + i
            if a < b:
                pairs.add((a, int(b)))
    return pairs


def groups(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return [g for g in out.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()
    catalog = load_catalog()
    pairs = twin_pairs(load_embeddings(PHOTO_MODEL, catalog), args.threshold) & twin_pairs(
        load_embeddings(TEXT_MODEL, catalog), args.threshold)
    rank = pd.DataFrame({"no_brand": catalog.brand.isna(), "title_len": -catalog.title.fillna("").str.len(),
                         "source": catalog.source.map(SOURCE_PRIORITY)})
    removed, kept = [], []
    for g in groups(len(catalog), pairs):
        order = rank.iloc[g].sort_values(["no_brand", "title_len", "source"]).index
        keep = catalog.item_id.iloc[order[0]]
        for i in order[1:]:
            removed.append(catalog.item_id.iloc[i])
            kept.append(keep)
    dup = pd.DataFrame({"removed_id": removed, "kept_id": kept})
    out = ROOT / "data/processed/duplicates.parquet"
    old = pd.read_parquet(out) if out.exists() else dup.iloc[:0]
    pd.concat([old, dup]).drop_duplicates("removed_id").to_parquet(out, index=False)
    catalog = catalog[~catalog.item_id.isin(removed)].reset_index(drop=True)
    catalog.to_parquet(ROOT / "data/processed/catalog.parquet", index=False)
    by_src = pd.Series(removed).str.split("_").str[0].value_counts().to_dict()
    print(f"{len(pairs):,} twin pairs in both spaces -> removed {len(removed):,} duplicates ({by_src}); "
          f"catalog: {len(catalog):,} items")


if __name__ == "__main__":
    main()
