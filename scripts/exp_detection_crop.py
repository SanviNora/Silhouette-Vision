"""Experiment: does detecting and cropping the garment improve retrieval from street photos?

LookBench's RealStreetLook queries come with the full photo (`raw_image`) and the official crop.
For each query we retrieve against the cached Marqo gallery (+ 58k distractors) using:
  full        - the whole street photo (what a user uploads today)
  det_cat     - our detector's box for the query's garment type (as if the user tapped the item)
  det_top     - our detector's highest-scoring garment box (no user input)
  official    - LookBench's own crop (upper bound)
Detector: YOLOS trained on Fashionpedia (valentinafevu/yolos-fashionpedia, MIT), CPU.

Usage: python scripts/exp_detection_crop.py [--limit 981]
"""

import argparse
import io
import json

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from silhouette_vision.config import ROOT, path
from silhouette_vision.detect import GarmentDetector, lookbench_to_fashionpedia
from silhouette_vision.encoders import load_encoder

LB = ROOT / "data/raw/lookbench/v20251201"
SUBSET = "real_streetlook"


def gallery():
    e = path("embeddings") / "lookbench" / "marqo_fashion_siglip"
    g = np.vstack([np.load(e / f"{SUBSET}_gallery.npy"), np.load(e / "noise.npy")]).astype(np.float32)
    files = [LB / SUBSET / "gallery.parquet", *sorted((LB / "noise").glob("*.parquet"))]
    ids = np.array([i for f in files
                    for i in pq.read_table(f, columns=["item_ID"]).column("item_ID").to_pylist()],
                   dtype=object)
    return g, ids


def recall(q, q_ids, g, g_ids, ks=(1, 5, 10)):
    s = q @ g.T
    top = np.argpartition(-s, max(ks), 1)[:, : max(ks)]
    top = np.take_along_axis(top, np.take_along_axis(s, top, 1).argsort(1)[:, ::-1], 1)
    hits = g_ids[top] == q_ids[:, None]
    return {f"R@{k}": round(100 * float(hits[:, :k].any(1).mean()), 1) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t = pq.read_table(LB / SUBSET / "query.parquet", columns=["image", "raw_image", "item_ID", "category"])
    rows = t.to_pylist()[: args.limit]
    rows = [r for r in rows if r["raw_image"] and r["raw_image"].get("bytes")]
    q_ids = np.array([r["item_ID"] for r in rows], dtype=object)
    encoder, detector = load_encoder("marqo_fashion_siglip", device="cpu"), GarmentDetector()
    g, g_ids = gallery()

    variants = {"full": [], "det_cat": [], "det_top": [], "official": []}
    found_cat = found_any = 0
    for r in rows:
        raw = Image.open(io.BytesIO(r["raw_image"]["bytes"])).convert("RGB")
        dets = detector.detect(raw)
        wanted = lookbench_to_fashionpedia(r["category"])
        match = [d for d in dets if d.label in wanted]
        found_cat += bool(match)
        found_any += bool(dets)
        variants["full"].append(raw)
        variants["det_cat"].append(match[0].crop(raw) if match else raw)
        variants["det_top"].append(dets[0].crop(raw) if dets else raw)
        variants["official"].append(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))

    report = {"queries": len(rows), "detector_found_query_type": round(found_cat / len(rows), 3),
              "detector_found_any_garment": round(found_any / len(rows), 3)}
    print(f"{len(rows)} street photos; detector found the query's garment type in "
          f"{100 * found_cat / len(rows):.1f}%, any garment in {100 * found_any / len(rows):.1f}%")
    for name, imgs in variants.items():
        emb = np.concatenate([encoder.embed_images(imgs[i : i + 32]) for i in range(0, len(imgs), 32)])
        report[name] = recall(emb, q_ids, g, g_ids)
        print(f"  {name:9s} exact R@1/5/10 = {report[name]['R@1']}/{report[name]['R@5']}/{report[name]['R@10']}",
              flush=True)
    (path("reports") / "exp_detection_crop.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
