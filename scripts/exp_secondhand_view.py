"""Experiment: how to prepare Second-Hand Fashion photos (garments lie sideways on a sorting table).

Variants: raw; rotated upright; rotated + fixed centre crop; rotated + adaptive crop (garment mask
against the table, see images.garment_view). Scored on ~2,000 garments by (1) leave-one-out
10-NN vote accuracy of the labelled type and pattern, (2) zero-shot type accuracy, i.e. how well
the photo matches "a photo of a {type}" (what text search relies on).
Usage: python scripts/exp_secondhand_view.py [--n 2000]
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from silhouette_vision.config import ROOT, path
from silhouette_vision.encoders import load_encoder
from silhouette_vision.images import secondhand_view

IMAGES = ROOT / "data/raw/secondhand/images"


fixed_crop = secondhand_view  # rotated + centre crop: the chosen variant


def garment_mask_crop(img: Image.Image, pad: float = 0.06) -> Image.Image:
    """Rotated, cropped to the largest region differing from the table colour (grid lines removed
    by a morphological opening); centre crop if no clear region."""
    from scipy import ndimage

    img = img.convert("RGB").rotate(90, expand=True)
    w, h = img.size
    small = np.asarray(img.resize((w * 320 // max(w, h), h * 320 // max(w, h))), dtype=np.float32)
    border = np.concatenate([small[:8].reshape(-1, 3), small[-8:].reshape(-1, 3),
                             small[:, :8].reshape(-1, 3), small[:, -8:].reshape(-1, 3)])
    mask = ndimage.binary_opening(np.abs(small - np.median(border, 0)).max(-1) > 38, np.ones((7, 7)))
    labels, n = ndimage.label(ndimage.binary_closing(mask, np.ones((9, 9))))
    if n:
        sizes = ndimage.sum(labels > 0, labels, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        if 0.03 < sizes[k - 1] / labels.size < 0.85:
            ys, xs = np.nonzero(labels == k)
            sy, sx = h / small.shape[0], w / small.shape[1]
            y0, y1, x0, x1 = ys.min() * sy, (ys.max() + 1) * sy, xs.min() * sx, (xs.max() + 1) * sx
            py, px = pad * (y1 - y0), pad * (x1 - x0)
            return img.crop((max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py))))
    return img.crop((int(0.08 * w), int(0.18 * h), int(0.92 * w), int(0.82 * h)))


_DETECTOR = None


def detector_crop(img: Image.Image) -> Image.Image:
    """Rotated, cropped to the largest garment box from the YOLOS-Fashionpedia detector."""
    global _DETECTOR
    from silhouette_vision.detect import GarmentDetector

    _DETECTOR = _DETECTOR or GarmentDetector(threshold=0.3)
    img = img.convert("RGB").rotate(90, expand=True)
    boxes = [d for d in _DETECTOR.detect(img) if d.label not in {"sleeve", "neckline", "pocket", "zipper", "collar",
                                                                 "lapel", "buckle", "bead", "applique", "bow"}]
    if not boxes:
        return fixed_crop(img.rotate(-90, expand=True))
    best = max(boxes, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
    return best.crop(img, pad=0.06)


VARIANTS = {"raw": lambda im: im, "rotated": lambda im: im.rotate(90, expand=True),
            "rotated+centre": fixed_crop, "rotated+adaptive": garment_mask_crop,
            "rotated+detector": detector_crop}


def knn_vote(emb, labels, k=10):
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1)
    nn = np.argsort(-sims, 1)[:, :k]
    return float(np.mean([Counter(labels[n]).most_common(1)[0][0] == y for n, y in zip(nn, labels, strict=True)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    labels = pd.read_parquet(ROOT / "data/raw/secondhand/labels_sample.parquet")
    labels = labels[labels.garment_id.map(lambda i: (IMAGES / f"{i}.jpg").exists())]
    top = labels["type"].value_counts()
    labels = labels[labels["type"].isin(top[top >= 20].index)]
    labels = labels.sample(min(args.n, len(labels)), random_state=0)
    types = labels["type"].values
    pattern = labels.pattern.fillna("None").values
    enc = load_encoder("marqo_fashion_siglip")
    names = sorted(set(types))
    text = enc.embed_texts([f"a photo of a {t.lower()}" for t in names])
    report = {}
    for name, fn in VARIANTS.items():
        vecs = []
        for i in range(0, len(labels), 64):
            ims = [fn(Image.open(IMAGES / f"{g}.jpg").convert("RGB")) for g in labels.garment_id.iloc[i:i + 64]]
            vecs.append(enc.embed_images(ims))
        emb = np.vstack(vecs)
        zs = np.array(names)[(emb @ text.T).argmax(1)]
        report[name] = {"type_knn": knn_vote(emb, types), "pattern_knn": knn_vote(emb, pattern),
                        "type_zero_shot": float((zs == types).mean())}
        print(f"{name:17s} type 10-NN {100 * report[name]['type_knn']:.1f}%  pattern 10-NN "
              f"{100 * report[name]['pattern_knn']:.1f}%  type zero-shot {100 * report[name]['type_zero_shot']:.1f}%",
              flush=True)
    report["n"], report["types"] = len(labels), len(names)
    (path("reports") / "exp_secondhand_view.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
