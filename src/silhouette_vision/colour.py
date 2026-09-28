"""Colour descriptor for colour-aware re-ranking.

A normalized CIELAB histogram (4 L x 8 a x 8 b bins) over the non-background pixels.
Compared with histogram intersection (1 = identical colour distribution, 0 = disjoint).

Why: the embedding model captures type, pattern and brand far better than exact colour
(Myntra neighbour precision@10: type 88%, colour 42%). Blending in this descriptor with
weight 0.1 raised colour precision to 47% for a 0.9-point drop in type precision
(scripts/exp_colour_rerank.py).
"""

import numpy as np
from PIL import Image

BINS = (4, 8, 8)
DEFAULT_WEIGHT = 0.1


def colour_hist(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img.thumbnail((128, 128))
    lab = np.asarray(img.convert("LAB"), dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    # Background = very light, low-saturation pixels (studio white/grey).
    bg = (rgb.min(1) > 225) | ((rgb.max(1) - rgb.min(1) < 12) & (rgb.mean(1) > 200))
    fg = lab[~bg] if (~bg).sum() > 50 else lab
    hist, _ = np.histogramdd(fg, bins=BINS, range=[(0, 256)] * 3)
    hist = hist.ravel().astype(np.float32)
    return hist / hist.sum()


def intersection(query: np.ndarray, hists: np.ndarray) -> np.ndarray:
    return np.minimum(query[None, :], hists).sum(1)
