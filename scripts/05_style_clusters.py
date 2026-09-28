"""Stage 5: visual style clusters in the luxury (Farfetch) catalog, with a 2-D style map.

Per category (dresses with dresses, bags with bags):
  UMAP (cosine, 10-d) -> k-means, k = n_items / 1500 clamped to 8..20  => style clusters
  UMAP (2-d)                                                           => map coordinates
HDBSCAN was tried first and rejected for this use: granularity was inconsistent (2 clusters for
21k dresses, 35 for 25k bags) and up to 32% of items were left unassigned, which a merchandising
map can't use.

Naming: Marqo shares an image-text space, so each style word is scored against the cluster's mean
image embedding *minus* the category average. Raw similarity named almost every cluster
"logo print · statement" (hub words close to every fashion image); centring keeps what is
distinctive. Names are made unique within a category. Each cluster is summarised with signature
brands (by lift), median price, markdown rate and pre-owned share.

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

CATEGORIES = ["dress", "top", "bottoms", "outerwear", "shoes", "bag", "jewellery", "accessory"]
STYLE_WORDS = [
    "minimalist", "logo print", "monogram canvas", "floral print", "animal print", "graphic print",
    "striped", "polka dot", "checked plaid", "tie-dye", "sequin embellished", "crystal embellished",
    "lace", "sheer", "ruffled", "pleated", "draped", "tailored", "oversized", "cropped", "denim",
    "leather", "patent leather", "suede", "quilted", "metallic", "velvet", "satin silk", "knitted",
    "fur", "shearling", "fringe", "studded", "chain strap", "sporty", "athleisure", "bohemian",
    "romantic", "evening", "western", "military", "preppy", "gothic", "vintage", "colour block",
    "neon", "pastel", "all black", "all white", "gold-tone", "pearl", "statement", "chunky",
]


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


def main():
    catalog = load_catalog()
    emb_all = load_embeddings("marqo_fashion_siglip", catalog)
    ff = catalog.source.values == "farfetch"
    encoder = load_encoder("marqo_fashion_siglip", device="cpu")
    rows, summary = [], {}

    for category in CATEGORIES:
        t0 = time.time()
        idx = np.flatnonzero(ff & (catalog.category.values == category))
        emb, items = emb_all[idx], catalog.iloc[idx]
        reduced = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, metric="cosine",
                            random_state=0).fit_transform(emb)
        k = int(np.clip(round(len(idx) / 1500), 8, 20))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(reduced)
        xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=0).fit_transform(emb)

        noun = {"bottoms": "trousers or skirt", "jewellery": "piece of jewellery"}.get(category, category)
        text = encoder.embed_texts([f"a photo of a {w} {noun}" for w in STYLE_WORDS])
        names = name_clusters(emb, labels, text, STYLE_WORDS)

        sample = np.random.default_rng(0).choice(len(idx), min(5000, len(idx)), replace=False)
        sil = float(silhouette_score(reduced[sample], labels[sample]))
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
                "median_price_sgd": float(members.price.median()),
                "on_sale_share": float(members.on_sale.mean()),
                "preowned_share": float(members.is_preowned.mean()),
            }
        summary[category] = {"items": len(idx), "clusters": len(names),
                             "silhouette_umap10": sil, "detail": clusters}
        rows.append(pd.DataFrame({
            "item_id": items.item_id.values, "category": category,
            "cluster": [f"{category}_{c}" for c in labels],
            "cluster_name": [names[c] for c in labels],
            "map_x": xy[:, 0].astype(np.float32), "map_y": xy[:, 1].astype(np.float32),
        }))
        print(f"{category:10s} {len(idx):6d} items -> {k:2d} clusters, silhouette {sil:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    pd.concat(rows).to_parquet(ROOT / "data/processed/style_clusters.parquet", index=False)
    (path("reports") / "style_clusters.json").write_text(json.dumps(summary, indent=2))
    print("saved data/processed/style_clusters.parquet, artifacts/reports/style_clusters.json")


if __name__ == "__main__":
    main()
