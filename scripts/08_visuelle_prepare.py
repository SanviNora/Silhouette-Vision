"""Stage 8: Visuelle 2.0 tables and image embeddings for cold-start demand forecasting.

Outputs
  data/processed/visuelle_rows.parquet      one row per product x store: tags, price, release
                                            date, store, units sold in weeks 1-12
  data/processed/visuelle_products.parquet  one row per product (tags, mean price, n_stores,
                                            first release, total units)
  artifacts/embeddings/visuelle/marqo_fashion_siglip.npy  row-aligned with visuelle_products

`restock` (stock actually delivered during the season) and the weekly discounts are kept out of
the modelling tables' features: neither is known when a new product is being planned.
Usage: python scripts/08_visuelle_prepare.py
"""

import numpy as np
import pandas as pd

from silhouette_vision.config import ROOT, path
from silhouette_vision.encoders import load_encoder
from silhouette_vision.images import load_rgb

VIS = ROOT / "data/raw/visuelle2/visuelle2"
WEEKS = [str(i) for i in range(12)]


def load_visuelle(image_path: str):
    """One Visuelle photo (AI19/04442.png is truncated: loaded partially rather than dropped)."""
    try:
        return load_rgb(VIS / "images" / image_path)
    except OSError:
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            print(f"  truncated image loaded partially: {image_path}")
            return load_rgb(VIS / "images" / image_path)
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False


def main():
    sales = pd.read_csv(VIS / "sales.csv", index_col=0, parse_dates=["release_date"])
    prices = pd.read_csv(VIS / "price_discount_series.csv", index_col=0)
    assert (prices.retail.values == sales.retail.values).all()  # same row order
    rows = sales.drop(columns=["restock"]).assign(price=prices.price.values)
    rows = rows.rename(columns={w: f"w{w}" for w in WEEKS})
    rows = rows.assign(total=rows[[f"w{w}" for w in WEEKS]].sum(1))

    products = (rows.groupby("external_code")
                .agg(season=("season", "first"), category=("category", "first"),
                     color=("color", "first"), fabric=("fabric", "first"),
                     image_path=("image_path", "first"), price=("price", "mean"),
                     n_stores=("retail", "nunique"), release_date=("release_date", "min"),
                     total=("total", "sum"))
                .reset_index())
    print(f"{len(rows):,} product-store rows, {len(products):,} products, {rows.retail.nunique()} stores")
    print(products.groupby("season").agg(products=("external_code", "size"),
                                         median_units=("total", "median")).to_string())

    out = path("processed")
    rows.to_parquet(out / "visuelle_rows.parquet", index=False)
    products.to_parquet(out / "visuelle_products.parquet", index=False)

    encoder = load_encoder("marqo_fashion_siglip")
    vecs = []
    for i in range(0, len(products), 64):
        batch = [load_visuelle(p) for p in products.image_path.iloc[i:i + 64]]
        vecs.append(encoder.embed_images(batch))
    emb_dir = path("embeddings") / "visuelle"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "marqo_fashion_siglip.npy", np.vstack(vecs).astype(np.float32))
    print(f"saved {len(products):,} image embeddings")


if __name__ == "__main__":
    main()
