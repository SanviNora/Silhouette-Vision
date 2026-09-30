"""Stage 6: calibrated exact-match confidence ("94% likely the exact product").

Raw cosine similarity is not a probability: 0.80 can be the same product or merely a similar one.
We learn P(same product | similarity) by logistic regression on LookBench, where every query's
true products are known: for the top-10 results of each query, label = result is the exact item.
Fitted on even-indexed queries, evaluated on odd-indexed ones (reliability table, ECE, and
precision/recall of "exact match" claims at several thresholds).

Checks on the search catalog itself:
  - LookBench's studio products are in the catalog, so its real studio query photos are searched
    against all 51k products: is the calibration still right there? Its street queries show
    *other* products (LookBench numbers items per subset: street item 3 is not studio item 3), so
    they test the opposite case, a product the catalog does not have: how often is "exact" claimed?
  - "a photo found online": catalog images are cropped, shrunk, JPEG-compressed and brightened,
    then searched; the original should come back first with high confidence.

Output: artifacts/models/match_confidence.json
Usage: python scripts/06_match_confidence.py [--robustness 400]
"""

import argparse
import io
import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression

from silhouette_vision.catalog import load_catalog
from silhouette_vision.config import ROOT, path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.images import load_rgb
from silhouette_vision.search import SearchEngine

LB = ROOT / "data/raw/lookbench/v20251201"
SUBSETS = ["real_studio_flat", "aigen_studio", "real_streetlook", "aigen_streetlook"]
K = 10


def features(sorted_scores: np.ndarray) -> np.ndarray:
    """[top-1 similarity, gap to #2, gap to #5] from each row of descending scores."""
    s = sorted_scores
    return np.column_stack([s[:, 0], s[:, 0] - s[:, 1], s[:, 0] - s[:, 4]])


def lookbench_top1(model: str, blend_with: str | None = None, w: float = 0.5):
    """Features of each LookBench query's top-5 scores, and whether its top-1 is the exact item."""
    def ids(file):
        return pq.read_table(file, columns=["item_ID"]).column("item_ID").to_pylist()

    noise_ids = [i for f in sorted((LB / "noise").glob("*.parquet")) for i in ids(f)]
    X, y = [], []
    for subset in SUBSETS:
        s = _scores(model, subset)
        if blend_with:
            s = (1 - w) * s + w * _scores(blend_with, subset)
        q_ids = np.array(ids(LB / subset / "query.parquet"), dtype=object)
        g_ids = np.array(ids(LB / subset / "gallery.parquet") + noise_ids, dtype=object)
        top = np.argpartition(-s, 5, 1)[:, :5]
        top = np.take_along_axis(top, np.take_along_axis(s, top, 1).argsort(1)[:, ::-1], 1)
        X.append(features(np.take_along_axis(s, top, 1)))
        y.append(g_ids[top[:, 0]] == q_ids)
    return np.vstack(X), np.concatenate(y)


def _scores(model: str, subset: str) -> np.ndarray:
    e = path("embeddings") / "lookbench" / model
    q = np.load(e / f"{subset}_query.npy").astype(np.float32)
    g = np.vstack([np.load(e / f"{subset}_gallery.npy"), np.load(e / "noise.npy")]).astype(np.float32)
    return q @ g.T


def reliability(prob, label, bins=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001)):
    table = []
    for lo, hi in pairwise(bins):
        m = (prob >= lo) & (prob < hi)
        if m.sum():
            table.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()),
                          "predicted": float(prob[m].mean()), "actual": float(label[m].mean())})
    ece = sum(t["n"] * abs(t["predicted"] - t["actual"]) for t in table) / len(prob)
    return table, float(ece)


def calibrate(X, y):
    idx = np.arange(len(y))
    dev, test = idx % 2 == 0, idx % 2 == 1
    lr = LogisticRegression(C=10).fit(X[dev], y[dev])
    p = lr.predict_proba(X[test])[:, 1]
    table, ece = reliability(p, y[test])
    claims = {}
    for t in (0.5, 0.7, 0.8, 0.9):
        m = p >= t
        claims[t] = {"share_of_queries": float(m.mean()),
                     "precision": float(y[test][m].mean()) if m.any() else None}
    return lr, {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                "features": ["similarity", "gap_to_2nd", "gap_to_5th"],
                "base_rate": float(y.mean()), "test_ece": ece, "reliability": table, "claims": claims}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def catalog_check(lr) -> dict:
    """Studio queries (product in the catalog as lb_<item_ID>): exact top-1 rate and calibration.
    Street queries (product not in the catalog): share of photos wrongly claimed exact."""
    catalog = load_catalog()
    emb = load_embeddings("marqo_fashion_siglip", catalog)
    row_of = pd.Series(np.arange(len(catalog)), index=catalog.item_id)
    result = {}
    for subset in ["real_studio_flat", "real_streetlook"]:
        q = np.load(path("embeddings") / "lookbench" / "marqo_fashion_siglip" / f"{subset}_query.npy").astype(np.float32)
        ids = pq.read_table(LB / subset / "query.parquet", columns=["item_ID"]).column(0).to_pylist()
        keep = np.array([f"lb_{i}" in row_of for i in ids])
        q, target = q[keep], row_of[[f"lb_{i}" for i, k in zip(ids, keep, strict=True) if k]].values
        sims = q @ emb.T
        top = np.argsort(-sims, 1)[:, :5]
        exact = top[:, 0] == target
        p = lr.predict_proba(features(np.take_along_axis(sims, top, 1)))[:, 1]
        table, ece = reliability(p, exact)
        claims = {t: {"share": float((p >= t).mean()),
                      "precision": float(exact[p >= t].mean()) if (p >= t).any() else None} for t in (0.5, 0.8, 0.9)}
        if subset == "real_streetlook":  # product absent: any "exact" claim is false
            result["absent_product_street"] = {"queries": int(keep.sum()),
                                               "false_claims": {t: c["share"] for t, c in claims.items()}}
            print(f"[catalog] street photos of products NOT in the catalog ({keep.sum()}): claimed exact "
                  + ", ".join(f"P>={t}: {100 * c['share']:.1f}%" for t, c in claims.items()))
            continue
        result["in_catalog_studio"] = {"queries": int(keep.sum()), "exact_top1": float(exact.mean()),
                                       "ece": ece, "claims": claims, "reliability": table}
        print(f"[catalog] studio photos of products in the catalog ({keep.sum()} vs {len(catalog):,}): exact #1 "
              f"{100 * exact.mean():.1f}%, ECE {ece:.3f}, "
              + ", ".join(f"P>={t}: {100 * c['share']:.0f}% of queries, precision {_pct(c['precision'])}"
                          for t, c in claims.items()))
    return result


def degrade(img: Image.Image, rng) -> Image.Image:
    w, h = img.size
    f = rng.uniform(0.85, 0.95)
    x0, y0 = rng.uniform(0, 1 - f) * w, rng.uniform(0, 1 - f) * h
    img = img.crop((x0, y0, x0 + f * w, y0 + f * h))
    img = img.resize((max(64, int(img.width * 0.7)), max(64, int(img.height * 0.7))))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", type=int, default=400)
    args = ap.parse_args()

    out, models = {}, {}
    for mode, model, other in [("marqo", "marqo_fashion_siglip", None)]:
        lr, res = calibrate(*lookbench_top1(model, other))
        out[mode], models[mode] = res, lr
        print(f"\n[{mode}] top-1 is exact for {100 * res['base_rate']:.1f}% of queries; "
              f"held-out ECE {res['test_ece']:.3f}")
        for r in res["reliability"]:
            print(f"   predicted {r['bin']:9s} n={r['n']:5d}  mean predicted {100 * r['predicted']:5.1f}%"
                  f"  actually exact {100 * r['actual']:5.1f}%")
        for t, c in res["claims"].items():
            prec = "-" if c["precision"] is None else f"{100 * c['precision']:.1f}%"
            print(f"   say 'exact match' when P >= {t}: {100 * c['share_of_queries']:5.1f}% of queries, "
                  f"precision {prec}")

    out["catalog_lookbench"] = catalog_check(models["marqo"])

    # Robustness: degraded catalog photos should find themselves, confidently.
    engine = SearchEngine(precise=False)
    rng = np.random.default_rng(0)
    catalog = load_catalog()
    picks = rng.choice(len(catalog), args.robustness, replace=False)
    found, feats = [], []
    for i in picks:
        img = degrade(load_rgb(ROOT / catalog.image_path.iloc[i]), rng)
        res = engine.search(engine.image_vector(img), k=5)
        found.append(res.item_id.iloc[0] == catalog.item_id.iloc[i])
        feats.append(features(res.similarity.values[None, :])[0])
    found, conf = np.array(found), models["marqo"].predict_proba(np.array(feats))[:, 1]
    out["robustness"] = {"n": len(found), "top1_is_original": float(found.mean()),
                         "median_confidence_when_found": float(np.median(conf[found])),
                         "share_found_and_confident_0.8": float((found & (conf >= 0.8)).mean())}
    print(f"\n[robustness] {len(found)} degraded catalog photos (crop, 70% resize, JPEG q60, brightness): "
          f"original #1 in {100 * found.mean():.1f}%, median confidence {100 * np.median(conf[found]):.0f}%, "
          f"found with >=80% confidence in {100 * out['robustness']['share_found_and_confident_0.8']:.1f}%")
    (path("models") / "match_confidence.json").write_text(json.dumps(out, indent=2))
    print("saved artifacts/models/match_confidence.json")


if __name__ == "__main__":
    main()
