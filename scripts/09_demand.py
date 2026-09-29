"""Stage 9: cold-start demand forecasting on Visuelle 2.0 (strict season split).

Task: for each product x store, predict units sold in each of the 12 weeks after launch, for
products never seen in training. Train = SS17, AW17, SS18, AW18; test = all SS19 + AW19 products.
Settings (k, boosting iterations) are chosen by training on SS17-SS18 and validating on AW18.

Known at planning time and used: store, launch date, tags (category, colour, fabric), price,
number of stores, product photo. Not used: restock (stock delivered during the season) and
weekly discounts, both only known afterwards.

Simple models (per-store total units spread over a weekly shape):
  global      - average training row
  tags        - average of past products with the same tags (backing off to category), x store
  lookalike   - average of the k visually most similar past products, x store
Gradient boosting on log(units per store), feature sets added one at a time:
  basics      - store, store average, launch week/month, number of stores, price
  +tags       - category, colour, fabric (+ their past average)
  +image      - look-alike sales + 16 principal components of the embedding  (= final model)
and the same without "number of stores", which carries the planners' own expectations, to
measure what the product itself (tags, image) says about demand.

Metrics: WAPE of weekly product sales (summed over stores; primary), WAPE of product-store-week
values (very noisy: ~1 unit per store-week), WAPE of 12-week product totals, bias, and Spearman
rank correlation of units *per store* (does it rank best-sellers? Product totals would mostly
rank how many stores a product is in). A paired bootstrap over products gives a 95% interval for
the image's effect.
Outputs: artifacts/reports/demand.json, and artifacts/models/demand.joblib (the final model
refitted on all six seasons for the app, with the test error range and an in-range threshold)
Usage: python scripts/09_demand.py
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from silhouette_vision.config import path
from silhouette_vision.demand import (
    FEATURE_SETS,
    FINAL,
    N_PCA,
    TAGS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    WEEK_COLS,
    Experiment,
    fit_gbm,
    gbm_matrix,
    load_visuelle,
    wape,
)


def product_level(Y, P, product_ids):
    """Weekly sales summed over stores, one row per product."""
    return (pd.DataFrame(Y).groupby(product_ids).sum().values,
            pd.DataFrame(P).groupby(product_ids).sum().values)


def evaluate(name, Y, P, product_ids, n_stores):
    wy, wp = product_level(Y, P, product_ids)
    ty, tp = wy.sum(1), wp.sum(1)
    return {"model": name, "product_week_wape": wape(wy, wp), "store_week_wape": wape(Y, P),
            "product_total_wape": wape(ty, tp),
            "rank_units_per_store": float(spearmanr(ty / n_stores, tp / n_stores).statistic),
            "bias_pct": float(100 * (P.sum() - Y.sum()) / Y.sum())}


def bootstrap_gain(Y, P_without, P_with, product_ids, n=1000, seed=0):
    """95% interval of the drop in product-week WAPE from adding the image (paired, by product)."""
    wy, w0 = product_level(Y, P_without, product_ids)
    _, w1 = product_level(Y, P_with, product_ids)
    err0, err1, tot = np.abs(wy - w0).sum(1), np.abs(wy - w1).sum(1), wy.sum(1)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n):
        b = rng.integers(0, len(tot), len(tot))
        gains.append(100 * (err0[b].sum() - err1[b].sum()) / tot[b].sum())
    return float(np.mean(gains)), [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]


def run(rows, products, emb, train_seasons, test_seasons, k, max_iter, sets=FEATURE_SETS):
    ex = Experiment(rows, products, emb, train_seasons, test_seasons, k)
    Xtr, Ytr, _ = ex.row_frame(ex.train_r)
    Xte, Yte, tag_shape = ex.row_frame(ex.test_r)
    pid = Xte["product"].values
    n_stores = products.n_stores.values[np.unique(pid)]
    preds = {n: P for n, P in ex.simple_predictions(Xte, tag_shape).items() if n != "store"}
    pca = PCA(N_PCA, random_state=0).fit(emb[ex.train_p])
    categories = {c: sorted(products[c].unique()) for c in TAGS}
    Mtr = gbm_matrix(Xtr, pca.transform(emb[Xtr["product"].values]), categories)
    Mte = gbm_matrix(Xte, pca.transform(emb[Xte["product"].values]), categories)
    ytr = np.log1p(Ytr.sum(1))
    for name in sets:
        cols = FEATURE_SETS[name]
        shape = ex.knn_shape[pid] if "knn_log" in cols else tag_shape
        model = fit_gbm(Mtr[cols], ytr, max_iter)
        preds[name] = np.expm1(model.predict(Mte[cols])).clip(0)[:, None] * shape
    results = [evaluate(n, Yte, P, pid, n_stores) for n, P in preds.items()]
    return results, preds, (ex, Yte, pid)


def main():
    rows, products, emb = load_visuelle()

    print("Tuning on AW18 (train SS17-SS18), final model, product-week WAPE:")
    best = None
    for k in (20, 50, 100):
        for max_iter in (200, 500):
            res, _, _ = run(rows, products, emb, TRAIN_SEASONS[:3], ["AW18"], k, max_iter, sets=[FINAL])
            w = {x["model"]: x["product_week_wape"] for x in res}[FINAL]
            print(f"  k={k:3d} iterations={max_iter}: {w:.1f}")
            if best is None or w < best[0]:
                best = (w, k, max_iter)
    _, k, max_iter = best
    print(f"chosen: k={k}, iterations={max_iter}")

    val, _, _ = run(rows, products, emb, TRAIN_SEASONS[:3], ["AW18"], k, max_iter)
    print("\nValidation (AW18):")
    print(pd.DataFrame(val).set_index("model").round(3).to_string())

    res, preds, (ex, Yte, pid) = run(rows, products, emb, TRAIN_SEASONS, TEST_SEASONS, k, max_iter)
    table = pd.DataFrame(res).set_index("model")
    print(f"\nTest: {int(ex.test_p.sum())} new products ({', '.join(TEST_SEASONS)}), "
          f"{int(ex.test_r.sum()):,} product-store rows")
    print(table.round(3).to_string())
    gain = {"with n_stores": bootstrap_gain(Yte, preds["+tags"], preds["+tags+image"], pid),
            "without n_stores": bootstrap_gain(Yte, preds["no n_stores: tags"],
                                               preds["no n_stores: tags+image"], pid)}
    for key, (mean, ci) in gain.items():
        print(f"image effect ({key}): product-week WAPE -{mean:.2f} points, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    per_season = {}
    for season in TEST_SEASONS:
        r_, _, _ = run(rows, products, emb, TRAIN_SEASONS, [season], k, max_iter,
                       sets=[FINAL, "raw n_stores: +tags+image"])
        per_season[season] = {x["model"]: {"product_week_wape": x["product_week_wape"],
                                           "bias_pct": x["bias_pct"]} for x in r_}
    print("per test season:", json.dumps(per_season, indent=1))
    report = {"k": k, "max_iter": max_iter, "validation_aw18": val, "per_season": per_season, "n_test_products": int(ex.test_p.sum()),
              "n_test_rows": int(ex.test_r.sum()), "results": res,
              "image_gain_product_week_wape": {key: {"mean": m, "ci95": ci} for key, (m, ci) in gain.items()}}
    (path("reports") / "demand.json").write_text(json.dumps(report, indent=2))

    # Error range from the test: actual / predicted 12-week product totals, final model.
    wy, wp = product_level(Yte, preds[FINAL], pid)
    ratio = wy.sum(1) / np.clip(wp.sum(1), 1e-6, None)
    q10, q90 = np.percentile(ratio, [10, 90])
    print(f"test products: actual/predicted total between {q10:.2f} and {q90:.2f} for 80% of products")
    deploy(rows, products, emb, k, max_iter, q10, q90)


def deploy(rows, products, emb, k, max_iter, q10, q90):
    """Refit the final model on all six seasons and save what the app needs."""
    import joblib

    seasons = TRAIN_SEASONS + TEST_SEASONS
    ex = Experiment(rows, products, emb, seasons, [], k)
    X, Y, _ = ex.row_frame(ex.train_r)
    pca = PCA(N_PCA, random_state=0).fit(emb)
    categories = {c: sorted(products[c].unique()) for c in TAGS}
    model = fit_gbm(gbm_matrix(X, pca.transform(emb[X["product"].values]), categories)[FEATURE_SETS[FINAL]],
                    np.log1p(Y.sum(1)), max_iter)
    # Reference for the app: 1st percentile of each product's similarity to its nearest *other*
    # product. Shoppers' photos (people, backgrounds) score lower even for similar clothes, so the
    # app gates on garment type instead and only reports this similarity.
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1)
    min_similarity = float(np.percentile(sims.max(1), 1))
    latest = products[products.season == "AW19"]
    bundle = {
        "model": model, "pca": pca, "categories": categories, "k": k,
        "log_units": ex.stats.log_units.values, "shape": ex.stats[WEEK_COLS].values,
        "products": products[["external_code", "season", "category", "color", "fabric", "image_path",
                              "n_stores", "total"]],
        "store_log": ex.store_log, "global_log": ex.global_log, "global_shape": ex.global_shape,
        "store_order": rows.retail.value_counts().index.values,  # stores by number of launches
        "latest_n_stores": latest.n_stores.values, "latest_prices": latest.price.values,
        "ratio_q10": float(q10), "ratio_q90": float(q90), "min_similarity": min_similarity,
    }
    joblib.dump(bundle, path("models") / "demand.joblib")
    print(f"saved artifacts/models/demand.joblib (in-range similarity >= {min_similarity:.3f})")


if __name__ == "__main__":
    main()
