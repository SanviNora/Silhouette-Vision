"""Cold-start demand: 12-week sales of a new product from its photo, tags and price (Visuelle 2.0).

A new product has no sales history, so it borrows from past products: its visual look-alikes
(nearest neighbours in Marqo embedding space) and past products with the same tags. The final
model is gradient boosting on log(units per store) with store, launch timing, tags, price and
image features (look-alike sales + a few principal components of the embedding); the weekly
curve is the predicted total spread over the look-alikes' average weekly shape.

Evaluated with a strict season split in scripts/09_demand.py: train on SS17-AW18, test on every
SS19/AW19 product, none of which appears in training.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from silhouette_vision.config import path

WEEK_COLS = [f"w{i}" for i in range(12)]
TRAIN_SEASONS = ["SS17", "AW17", "SS18", "AW18"]
TEST_SEASONS = ["SS19", "AW19"]
TAGS = ["category", "color", "fabric"]


def load_visuelle() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rows = pd.read_parquet(path("processed") / "visuelle_rows.parquet")
    products = pd.read_parquet(path("processed") / "visuelle_products.parquet")
    emb = np.load(path("embeddings") / "visuelle" / "marqo_fashion_siglip.npy")
    assert len(emb) == len(products)
    return rows, products, emb


def wape(y: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |error| / sum actual, in %."""
    return float(100 * np.abs(y - pred).sum() / y.sum())


def lookalikes(query: np.ndarray, pool: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    """Indices of the k most similar pool embeddings for each query row, best first."""
    sims = query @ pool.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(sims, idx, 1), 1), 1)


N_PCA = 16


def product_stats(rows: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Per product: log mean units per store, and the normalised weekly shape."""
    g = rows.groupby("external_code")
    weekly = g[WEEK_COLS].sum()
    out = pd.DataFrame(index=products.external_code)
    out["log_units"] = np.log1p(g.total.mean()).reindex(out.index).values
    shape = weekly.div(weekly.sum(1).clip(lower=1), axis=0).reindex(out.index).astype(float)
    out[WEEK_COLS] = shape.to_numpy()
    return out


class Experiment:
    def __init__(self, rows, products, emb, train_seasons, test_seasons, k=20):
        self.rows, self.products, self.emb, self.k = rows, products, emb, k
        self.pidx = pd.Series(np.arange(len(products)), index=products.external_code)
        self.train_p = products.season.isin(train_seasons).values
        self.test_p = products.season.isin(test_seasons).values
        self.train_r = rows.season.isin(train_seasons).values
        self.test_r = rows.season.isin(test_seasons).values
        self.stats = product_stats(rows[self.train_r], products)  # NaN for non-training products
        tr = rows[self.train_r]
        self.store_log = np.log1p(tr.groupby("retail").total.mean())
        self.global_log = float(np.log1p(tr.total.mean()))
        self.global_shape = self.stats.loc[self.train_p, WEEK_COLS].mean().values
        # Breadth of distribution relative to the season's assortment: the brand widened
        # distribution in 2019 (mean 23 stores vs 14-21 before) while per-store sales of widely
        # distributed products fell, so the raw count drifts; its within-season rank does not.
        self.store_pct = products.groupby("season").n_stores.rank(pct=True).values
        self._lookalike_features()

    def _lookalike_features(self):
        """Look-alike sales and shape per product. Training products borrow only from *other*
        training seasons (as a new season would), test products from all training seasons."""
        p = self.products
        log_units = np.full(len(p), np.nan)
        shape = np.full((len(p), 12), np.nan)
        self.neighbours = {}
        seasons = p.season.values
        for i in np.flatnonzero(self.train_p | self.test_p):
            pool = np.flatnonzero(self.train_p & (seasons != seasons[i]))
            nn = pool[lookalikes(self.emb[i:i + 1], self.emb[pool], self.k)[0]]
            self.neighbours[i] = nn
            log_units[i] = self.stats.log_units.values[nn].mean()
            shape[i] = self.stats[WEEK_COLS].values[nn].mean(0)
        self.knn_log, self.knn_shape = log_units, shape

    def _tag_log(self, i, seasons_ok):
        p, st = self.products, self.stats
        for keys in (TAGS, ["category", "color"], ["category"]):
            m = seasons_ok.copy()
            for key in keys:
                m &= p[key].values == p[key].values[i]
            if m.sum() >= 5:
                return st.log_units.values[m].mean(), st[WEEK_COLS].values[m].mean(0)
        return self.global_log, self.global_shape

    def row_frame(self, mask):
        r = self.rows[mask]
        i = self.pidx.loc[r.external_code].values
        rel = pd.to_datetime(r.release_date)
        seasons = self.products.season.values
        tag = [self._tag_log(j, self.train_p & (seasons != seasons[j])) for j in np.unique(i)]
        tag_map = dict(zip(np.unique(i), tag, strict=True))
        return pd.DataFrame({
            "product": i, "retail": r.retail.values,
            "store_log": self.store_log.reindex(r.retail).fillna(self.global_log).values,
            "category": self.products.category.values[i], "color": self.products.color.values[i],
            "fabric": self.products.fabric.values[i], "price": r.price.values,
            "n_stores": self.products.n_stores.values[i],
            "n_stores_pct": self.store_pct[i],
            "week_of_year": rel.dt.isocalendar().week.values.astype(int), "month": rel.dt.month.values,
            "tag_log": [tag_map[j][0] for j in i], "knn_log": self.knn_log[i],
        }), r[WEEK_COLS].values, np.array([tag_map[j][1] for j in i])

    def simple_predictions(self, X, tag_shape):
        store_adj = X.store_log.values - self.global_log
        n = len(X)
        shape_knn = self.knn_shape[X["product"].values]
        return {
            "global": np.outer(np.full(n, np.expm1(self.global_log)), self.global_shape),
            "store": np.expm1(X.store_log.values)[:, None] * self.global_shape,
            "tags": np.expm1(X.tag_log.values + store_adj)[:, None] * tag_shape,
            "lookalike": np.expm1(X.knn_log.values + store_adj)[:, None] * shape_knn,
        }


BASICS = ["retail", "store_log", "week_of_year", "month", "n_stores_pct", "price"]
TAG_FEATURES = [*TAGS, "tag_log"]
IMAGE_FEATURES = ["knn_log"] + [f"pc{j}" for j in range(N_PCA)]
FEATURE_SETS = {
    "basics": BASICS,
    "+tags": BASICS + TAG_FEATURES,
    "+tags+image": BASICS + TAG_FEATURES + IMAGE_FEATURES,
    "raw n_stores: +tags+image": [f if f != "n_stores_pct" else "n_stores" for f in BASICS]
    + TAG_FEATURES + IMAGE_FEATURES,
    "no n_stores: tags": [f for f in BASICS if f != "n_stores_pct"] + TAG_FEATURES,
    "no n_stores: image": [f for f in BASICS if f != "n_stores_pct"] + IMAGE_FEATURES,
    "no n_stores: tags+image": [f for f in BASICS if f != "n_stores_pct"] + TAG_FEATURES + IMAGE_FEATURES,
}
FINAL = "+tags+image"


def gbm_matrix(X, comps, categories):
    cols = {c: X[c].values for c in ["retail", "store_log", "price", "n_stores", "n_stores_pct",
                                     "week_of_year", "month", "tag_log", "knn_log"]}
    for c in TAGS:
        cols[c] = pd.Categorical(X[c], categories=categories[c]).codes
    for j in range(N_PCA):
        cols[f"pc{j}"] = comps[:, j]
    return pd.DataFrame(cols)


def fit_gbm(M, y, max_iter):
    cat = [c in ("retail", *TAGS) for c in M.columns]
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         categorical_features=cat, random_state=0).fit(M, y)


class DemandForecaster:
    """12-week forecast for one new product in the brand's stores (model from scripts/09_demand.py,
    trained on all six seasons). Also returns the look-alike past products it borrowed from."""

    def __init__(self):
        import joblib

        self.b = joblib.load(path("models") / "demand.joblib")
        self.emb = np.load(path("embeddings") / "visuelle" / "marqo_fashion_siglip.npy")
        self.products = self.b["products"]

    def nearest_similarity(self, image_vec: np.ndarray) -> float:
        """Similarity to the closest past product (the brand's own new products: >= min_similarity)."""
        return float((self.emb @ image_vec).max())

    def suggest_tags(self, image_vec: np.ndarray, k: int = 15) -> dict:
        """Category, colour and fabric by vote of the k most similar past products (on Visuelle,
        leave-one-out: category 77%, colour 66%, fabric 64%; zero-shot text got 31% on category)."""
        nn = lookalikes(image_vec[None, :], self.emb, k)[0]
        return {key: self.products[key].iloc[nn].mode().iloc[0] for key in TAGS}

    def _tag_log(self, category, color, fabric):
        p, logu, shape = self.products, self.b["log_units"], self.b["shape"]
        for keys, values in ((TAGS, (category, color, fabric)), (["category", "color"], (category, color)),
                             (["category"], (category,))):
            m = np.ones(len(p), bool)
            for key, value in zip(keys, values, strict=True):
                m &= p[key].values == value
            if m.sum() >= 5:
                return logu[m].mean(), shape[m].mean(0)
        return self.b["global_log"], self.b["global_shape"]

    def forecast(self, image_vec, category, color, fabric, price_pct, n_stores, launch_date) -> dict:
        b = self.b
        nn = lookalikes(image_vec[None, :], self.emb, b["k"])[0]
        knn_log, shape = b["log_units"][nn].mean(), b["shape"][nn].mean(0)
        tag_log, _ = self._tag_log(category, color, fabric)
        stores = b["store_order"][:n_stores]
        dist = b["latest_n_stores"]
        launch = pd.Timestamp(launch_date)
        X = pd.DataFrame({
            "retail": stores, "store_log": b["store_log"].reindex(stores).fillna(b["global_log"]).values,
            "category": category, "color": color, "fabric": fabric,
            "price": float(np.quantile(b["latest_prices"], price_pct)),
            "n_stores": n_stores, "n_stores_pct": (dist < n_stores).mean() + 0.5 * (dist == n_stores).mean(),
            "week_of_year": int(launch.isocalendar().week), "month": launch.month,
            "tag_log": tag_log, "knn_log": knn_log,
        })
        comps = np.repeat(b["pca"].transform(image_vec[None, :]), len(X), 0)
        M = gbm_matrix(X, comps, b["categories"])[FEATURE_SETS[FINAL]]
        per_store = np.expm1(b["model"].predict(M)).clip(0)
        weekly = per_store.sum() * shape
        total = float(weekly.sum())
        look = self.products.iloc[nn[:8]].assign(units_per_store=np.expm1(b["log_units"][nn[:8]]))
        return {"weekly": weekly, "total": total, "per_store": total / len(stores),
                "low": total * b["ratio_q10"], "high": total * b["ratio_q90"], "lookalikes": look}
