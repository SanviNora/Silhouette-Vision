"""Stage 3: colour histograms for every catalog image (CPU), for colour-aware re-ranking.

Output: artifacts/embeddings/colour/hists.npy (float16, row i = catalog row i) + item_ids.parquet
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from silhouette_vision.catalog import load_catalog
from silhouette_vision.colour import colour_hist
from silhouette_vision.config import ROOT, path


def hist_for(rel_path: str) -> np.ndarray:
    return colour_hist(Image.open(ROOT / rel_path))


def main():
    catalog = load_catalog()
    with ProcessPoolExecutor(6) as pool:
        hists = np.stack(list(pool.map(hist_for, catalog.image_path, chunksize=512)))
    out = path("embeddings") / "colour"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "hists.npy", hists.astype(np.float16))
    catalog[["item_id"]].to_parquet(out / "item_ids.parquet", index=False)
    print(f"colour histograms: {hists.shape} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
