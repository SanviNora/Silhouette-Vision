"""Experiment: which heatmap points at the garment that drives the match, occlusion or attribution?

LookBench RealStreetLook queries are full street photos with a box around the garment whose
product is the answer. For each query we compute two maps against the true product's embedding
(occlusion, 12x12, ~120 forward passes; patch attribution, 14x14, one forward + backward) and check:
  pointing  - is the hottest cell inside the garment box? (vs a random cell, vs the centre cell)
  deletion  - hiding the hottest 25% of cells should hurt the similarity more than hiding a
              random 25% or the coldest 25% (the map is faithful if it ranks regions correctly)

Output: artifacts/reports/exp_explain_heatmap.json
Usage: python scripts/exp_explain_heatmap.py [--limit 200]
"""

import argparse
import io
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from silhouette_vision.config import ROOT, path
from silhouette_vision.encoders import load_encoder
from silhouette_vision.explain import input_gradient_map, occlusion_map, patch_attribution

LB = ROOT / "data/raw/lookbench/v20251201/real_streetlook"
EMB = path("embeddings") / "lookbench" / "marqo_fashion_siglip"
# name -> (grid, map function)
METHODS = {
    "occlusion": (12, lambda enc, img, t: occlusion_map(enc, img, t, grid=12)),
    "occlusion8": (8, lambda enc, img, t: occlusion_map(enc, img, t, grid=8)),  # the app's setting
    "attribution": (14, patch_attribution),
    "input_gradient": (14, input_gradient_map),
}


def hide(encoder, image, cells, grid):
    x = encoder.preprocess(image)
    edges = np.linspace(0, x.shape[-1], grid + 1).round().astype(int)
    for r, c in cells:
        x[:, edges[r]:edges[r + 1], edges[c]:edges[c + 1]] = 0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    args = ap.parse_args()

    encoder = load_encoder("marqo_fashion_siglip")
    q = pq.read_table(LB / "query.parquet", columns=["raw_image", "bbox", "item_ID"])
    g_ids = np.array(pq.read_table(LB / "gallery.parquet", columns=["item_ID"]).column("item_ID").to_pylist())
    g_emb = np.load(EMB / "real_streetlook_gallery.npy").astype(np.float32)
    rng = np.random.default_rng(0)
    picks = rng.choice(q.num_rows, min(args.limit, q.num_rows), replace=False)

    res = {m: {"hit": [], "rand": [], "centre": [], "top": [], "random": [], "bottom": [], "time": []}
           for m in args.methods}
    for n, i in enumerate(picks):
        row = q.slice(int(i), 1).to_pylist()[0]
        image = Image.open(io.BytesIO(row["raw_image"]["bytes"])).convert("RGB")
        x0, y0, x1, y1 = json.loads(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"]
        target = g_emb[g_ids == row["item_ID"]].mean(0)
        target /= np.linalg.norm(target)
        for method in args.methods:
            grid, fn = METHODS[method]
            t = time.perf_counter()
            heat = fn(encoder, image, target)
            r_ = res[method]
            r_["time"].append(time.perf_counter() - t)
            w, h = image.size
            cx, cy = (np.arange(grid) + 0.5) / grid * w, (np.arange(grid) + 0.5) / grid * h
            inside = (cy[:, None] >= y0) & (cy[:, None] <= y1) & (cx[None, :] >= x0) & (cx[None, :] <= x1)
            r, c = np.unravel_index(heat.argmax(), heat.shape)
            r_["hit"].append(bool(inside[r, c]))
            r_["rand"].append(float(inside.mean()))
            r_["centre"].append(bool(inside[grid // 2, grid // 2]))
            k = grid * grid // 4
            order = np.argsort(-heat.ravel())
            sets = {"top": order[:k], "bottom": order[-k:], "random": rng.choice(grid * grid, k, replace=False)}
            variants = torch.stack([encoder.preprocess(image)] +
                                   [hide(encoder, image, [divmod(int(j), grid) for j in s_], grid)
                                    for s_ in sets.values()])
            sims = encoder.embed_pixels(variants) @ target
            for name, s_ in zip(sets, sims[1:], strict=True):
                r_[name].append(float(sims[0] - s_))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(picks)}  " + "  ".join(
                f"{m}: pointing {100 * np.mean(v['hit']):.0f}% drop top/rand/bottom "
                f"{np.mean(v['top']):.3f}/{np.mean(v['random']):.3f}/{np.mean(v['bottom']):.3f}"
                for m, v in res.items()), flush=True)

    report = {"n": len(picks)}
    for m, v in res.items():
        report[m] = {"grid": METHODS[m][0], "pointing_hit": float(np.mean(v["hit"])),
                     "pointing_random_cell": float(np.mean(v["rand"])),
                     "pointing_centre_cell": float(np.mean(v["centre"])),
                     "deletion_drop_hide_25pct": {k_: float(np.mean(v[k_])) for k_ in ("top", "random", "bottom")},
                     "top_beats_random": float(np.mean(np.array(v["top"]) > np.array(v["random"]))),
                     "seconds_per_map_median": float(np.median(v["time"]))}
    print(json.dumps(report, indent=2))
    file = path("reports") / "exp_explain_heatmap.json"
    old = json.loads(file.read_text()) if file.exists() else {}
    file.write_text(json.dumps(old | report, indent=2))


if __name__ == "__main__":
    main()
