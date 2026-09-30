"""Stage 5: visual style clusters in the search catalog, with a 2-D style map.

Per category (dresses with dresses, bags with bags):
  UMAP (cosine, 10-d) -> k-means, k = n_items / 1500 clamped to 6..16  => style clusters
  UMAP (2-d)                                                          => map coordinates
Sources photograph products differently (Second-Hand: flat lays on a tiled sorting table), and
raw embeddings clustered by *photo setup*: 97% source purity, clusters either ~0% or ~98%
second-hand. Each source's mean embedding is therefore removed within a category first, which
brings purity down to the majority baseline (76.6% for tops). Caveat: this also removes any
average style difference between sources, so the map shows relative differences.
HDBSCAN was rejected earlier (inconsistent granularity, up to 32% unassigned).

Naming: Marqo shares an image-text space, so each style word is scored against the cluster's mean
image embedding *minus* the category average (raw similarity names everything after hub words
like "logo print"). Each cluster is summarised with signature brands (by lift) and, where both
sides are large, its **donated share**: second-hand garments given away in 2022-24 vs products
sold new in 2025-26. A style over-represented among donations relative to its category is
plausibly fading; under-represented, plausibly current (an association, not a sales measure).

Outputs:
  data/processed/style_clusters.parquet     item_id, cluster, cluster_name, map_x, map_y
  artifacts/reports/style_clusters.json     per-cluster summary + quality metrics
"""

import json
import time

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from silhouette_vision.catalog import load_catalog
from silhouette_vision.config import ROOT, path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.encoders import load_encoder

CATEGORIES = ["top", "bottoms", "dress", "outerwear", "shoes", "bag"]  # >= 1,000 products each
MIN_SIDE = 300  # donated vs new is only reported when both sides have this many products
STYLE_WORDS = [
    "minimalist", "logo print", "monogram canvas", "floral print", "animal print", "graphic print",
    "striped", "polka dot", "checked plaid", "tie-dye", "sequin embellished", "crystal embellished",
    "lace", "sheer", "ruffled", "pleated", "draped", "tailored", "oversized", "cropped", "denim",
    "leather", "patent leather", "suede", "quilted", "metallic", "velvet", "satin silk", "knitted",
    "fur", "shearling", "fringe", "studded", "chain strap", "sporty", "athleisure", "bohemian",
    "romantic", "evening", "western", "military", "preppy", "gothic", "vintage", "colour block",
    "neon", "pastel", "all black", "all white", "gold-tone", "pearl", "statement", "chunky",
]


# Words that describe bags, shoes or jewellery, not garments: they named tops "chain strap".
NOT_FOR_CLOTHING = {"chain strap", "patent leather", "gold-tone", "monogram canvas", "pearl", "studded"}
CLOTHING = {"top", "bottoms", "dress", "outerwear"}


def name_clusters(emb, labels, text_emb, words):
    """Two most *distinctive* style words per cluster, unique within the category."""
    baseline = text_emb @ emb.mean(0)
    names, used = {}, set()
    for c in sorted(set(labels)):
        centroid = emb[labels == c].mean(0)
        ranked = [words[i] for i in np.argsort(-(text_emb @ centroid - baseline))]
        name = " · ".join(ranked[:2])
        j = 2
        while name in used and j < len(ranked):  # swap the second word until the name is unique
            name = f"{ranked[0]} · {ranked[j]}"
            j += 1
        used.add(name)
        names[c] = name
    return names


def centre_per_source(emb: np.ndarray, sources: np.ndarray) -> np.ndarray:
    out = emb.copy()
    for src in set(sources):
        m = sources == src
        if m.sum() >= 100:
            out[m] -= out[m].mean(0)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    catalog = load_catalog()
    emb_all = load_embeddings("marqo_fashion_siglip", catalog)
    encoder = load_encoder("marqo_fashion_siglip", device="cpu")
    rows, summary = [], {}

    for category in CATEGORIES:
        t0 = time.time()
        idx = np.flatnonzero(catalog.category.values == category)
        emb, items = emb_all[idx], catalog.iloc[idx]
        sources = items.source.values
        X = centre_per_source(emb, sources)
        reduced = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, metric="cosine",
                            random_state=0).fit_transform(X)
        k = int(np.clip(round(len(idx) / 1500), 6, 16))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(reduced)
        xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=0).fit_transform(X)

        noun = {"bottoms": "trousers or skirt"}.get(category, category)
        words = [w for w in STYLE_WORDS if category not in CLOTHING or w not in NOT_FOR_CLOTHING]
        text = encoder.embed_texts([f"a photo of a {w} {noun}" for w in words])
        names = name_clusters(emb, labels, text, words)

        sample = np.random.default_rng(0).choice(len(idx), min(5000, len(idx)), replace=False)
        sil = float(silhouette_score(reduced[sample], labels[sample]))
        donated = sources == "secondhand"
        compare = MIN_SIDE <= donated.sum() and MIN_SIDE <= (~donated).sum()
        purity = sum(max(np.sum((labels == c) & (sources == s)) for s in set(sources)) for c in set(labels))
        brand_share = items.brand.value_counts(normalize=True)
        clusters = {}
        for c, name in names.items():
            m = labels == c
            members = items[m]
            lift = (members.brand.value_counts(normalize=True) / brand_share).dropna()
            lift = lift[members.brand.value_counts() >= 10].sort_values(ascending=False)
            clusters[int(c)] = {
                "name": name, "size": int(m.sum()),
                "signature_brands": lift.head(3).index.tolist(),
                "sources": members.source.value_counts().to_dict(),
                "donated_share": float(donated[m].mean()) if compare else None,
                "donated_lift": float(donated[m].mean() / donated.mean()) if compare else None,
            }
        summary[category] = {"items": len(idx), "clusters": len(names), "silhouette_umap10": sil,
                             "source_purity": float(purity / len(idx)),
                             "donated_share": float(donated.mean()) if compare else None, "detail": clusters}
        rows.append(pd.DataFrame({
            "item_id": items.item_id.values, "category": category,
            "cluster": [f"{category}_{c}" for c in labels],
            "cluster_name": [names[c] for c in labels],
            "map_x": xy[:, 0].astype(np.float32), "map_y": xy[:, 1].astype(np.float32),
        }))
        print(f"{category:10s} {len(idx):6d} items -> {k:2d} clusters, silhouette {sil:.2f}, "
              f"source purity {100 * purity / len(idx):.0f}%  ({time.time() - t0:.0f}s)", flush=True)
        if compare:
            for v in sorted(clusters.values(), key=lambda v: -v["donated_lift"]):
                print(f"     {v['name']:38s} n={v['size']:5d}  donated {100 * v['donated_share']:4.0f}% "
                      f"(lift {v['donated_lift']:.2f})  brands: {', '.join(v['signature_brands'][:3])}")

    pd.concat(rows).to_parquet(ROOT / "data/processed/style_clusters.parquet", index=False)
    (path("reports") / "style_clusters.json").write_text(json.dumps(summary, indent=2))
    print("saved data/processed/style_clusters.parquet, artifacts/reports/style_clusters.json")


if __name__ == "__main__":
    main()
