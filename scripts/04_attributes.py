"""Stage 4: attribute prediction — evaluate zero-shot vs linear probe vs MLP, then enrich the catalog.

Method selection on Myntra labels (legacy catalog). The final heads for colour, pattern and
article type are then refitted on Myntra + the recent sources' own labels (recent_labels.py): on
held-out recent photos this raised colour 41.6 -> 85.2%, pattern 71.7 -> 78.4%, garment type
57.5 -> 79.1% and footwear type 14.4 -> 99.4%, for < 1.5 points on Myntra
(scripts/exp_attributes_joint.py). Predictions are written for the search catalog, with article
types constrained to each product's category (known from its source).

For each attribute (labels from Myntra), on a group-aware train/val/test split:
  1. zero-shot: Marqo text prompts, no training
  2. linear probe: logistic regression (class-balanced), C chosen on val
  3. MLP: one hidden layer (512), early stopping
The best method on *val* is refit on train+val and used to predict every catalog item for which
the attribute applies (e.g. sleeve length only for tops/dresses/outerwear). Test is only reported.

Outputs:
  artifacts/reports/attributes_<features>.json      per-attribute metrics
  data/processed/attribute_predictions.parquet      search catalog: item_id + pred_<attr> + conf_<attr>
  artifacts/models/attributes.joblib                fitted predictors, for the app

Usage: python scripts/04_attributes.py [--features marqo|gr_lite|concat]
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from silhouette_vision.attributes import (
    ATTRIBUTES,
    applicable_categories,
    expected_calibration_error,
    group_split,
    label_frame,
    prompt_texts,
    zero_shot,
)
from silhouette_vision.catalog import load_catalog, load_legacy_catalog
from silhouette_vision.config import ROOT, path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.encoders import load_encoder
from silhouette_vision.recent_labels import recent_labels


def features(kind: str, catalog: pd.DataFrame) -> np.ndarray:
    marqo = load_embeddings("marqo_fashion_siglip__legacy", catalog)
    if kind == "marqo":
        return marqo
    myntra = catalog[catalog.source == "myntra"].reset_index(drop=True)
    grl = np.full((len(catalog), 1024), np.nan, np.float32)
    grl[np.flatnonzero(catalog.source.values == "myntra")] = load_embeddings("gr_lite__myntra", myntra)
    return grl if kind == "gr_lite" else np.hstack([marqo, grl])


RECENT_ATTRS = {"colour", "pattern", "article_type"}


def type_to_category(catalog, labels) -> pd.Series:
    """Myntra article type -> its most common shared category (+ "Boots", from Amazon labels)."""
    myntra_cat = catalog.loc[labels.index, "category"]
    counts = labels.article_type.dropna().groupby(myntra_cat).value_counts().reset_index()
    out = counts.sort_values("count").drop_duplicates("article_type", keep="last").set_index(
        "article_type")["category"]
    return pd.concat([out, pd.Series({"Boots": "shoes"})])


def constrain_to_category(probs, classes, catalog, labels, item_categories):
    """Only allow article types consistent with each item's category.

    Myntra has almost no outerwear (its only kept outerwear type is "Jackets"), so an
    unconstrained classifier called Farfetch coats "Handbags" (black, shiny texture). Farfetch
    categories come from its titles, which agreed with the images for 89.6% of items, so they
    are a reliable second signal. Items in "other" (or with no compatible class) are left as is.
    """
    type_to_cat = type_to_category(catalog, labels)
    allowed = np.array([[type_to_cat.get(c) == cat for c in classes] for cat in item_categories])
    usable = allowed.any(1) & (item_categories != "other")
    out = probs.copy()
    out[usable] = np.where(allowed[usable], probs[usable], 0)
    out[usable] /= out[usable].sum(1, keepdims=True)
    return out


def scores(y_true, y_pred) -> dict:
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, y_pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["marqo", "gr_lite", "concat"], default="marqo")
    args = ap.parse_args()

    catalog = load_legacy_catalog()
    X_all = features(args.features, catalog)
    labels = label_frame(catalog)
    split = group_split(catalog.loc[labels.index])
    marqo = load_encoder("marqo_fashion_siglip", device="cpu")
    search = load_catalog()
    X_search = load_embeddings("marqo_fashion_siglip", search)
    recent = recent_labels(search, ROOT)
    report, predictions = {}, pd.DataFrame({"item_id": search.item_id})
    bundle = {"type_to_category": type_to_category(catalog, labels).to_dict(), "attributes": {}}

    for attr in ATTRIBUTES:
        t0 = time.time()
        y = labels[attr.name].dropna()
        X = X_all[y.index]
        s = split[y.index].values
        tr, va, te = s == "train", s == "val", s == "test"
        classes = sorted(y.unique())
        res = {"classes": len(classes), "n_train": int(tr.sum()), "n_test": int(te.sum())}

        # 1. zero-shot (only possible with the image-text model's embeddings)
        if args.features == "marqo":
            text = marqo.embed_texts(prompt_texts(attr, classes))
            zs_pred, _ = zero_shot(X, text)
            zs = np.array(classes)[zs_pred]
            res["zero_shot"] = {"val": scores(y[va], zs[va]), "test": scores(y[te], zs[te])}

        # 2. linear probe, C chosen on val
        probes = {}
        for c in (0.1, 1.0):
            m = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, max_iter=1000, class_weight="balanced"))
            m.fit(X[tr], y[tr])
            probes[c] = (m, f1_score(y[va], m.predict(X[va]), average="macro"))
        best_c = max(probes, key=lambda c: probes[c][1])
        lr = probes[best_c][0]
        lr_conf = lr.predict_proba(X[te]).max(1)
        lr_test = lr.predict(X[te])
        res["linear_probe"] = {"C": best_c, "val": scores(y[va], lr.predict(X[va])),
                               "test": scores(y[te], lr_test),
                               "ece": expected_calibration_error(lr_conf, lr_test == y[te].values)}

        # 3. MLP
        mlp = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(512,), early_stopping=True, max_iter=200, random_state=0))
        mlp.fit(X[tr], y[tr])
        res["mlp"] = {"val": scores(y[va], mlp.predict(X[va])), "test": scores(y[te], mlp.predict(X[te]))}

        methods = [m for m in ("zero_shot", "linear_probe", "mlp") if m in res]
        chosen = max(methods, key=lambda m: res[m]["val"]["macro_f1"])
        res["chosen"] = chosen
        report[attr.name] = res

        # Refit the chosen trained method on train+val (+ recent labels) and predict the
        # applicable items of the search catalog.
        if args.features == "marqo":
            categories = applicable_categories(catalog, labels, attr.name)
            if attr.name == "article_type":
                categories = sorted(set(categories) | {"shoes"})
            applies = search.category.isin(categories).values
            fit_rows = tr | va
            if chosen == "zero_shot":
                pred_idx, probs = zero_shot(X_search[applies], text)
                pred, conf = np.array(classes)[pred_idx], probs.max(1)
            else:
                final = lr if chosen == "linear_probe" else mlp
                Xf, yf = X[fit_rows], y[fit_rows].values
                if attr.name in RECENT_ATTRS:
                    r = recent[recent[attr.name].notna()]
                    r_tr, r_te = r[r.split == "train"], r[r.split == "test"]
                    final.fit(np.vstack([Xf, X_search[r_tr.row]]), np.concatenate([yf, r_tr[attr.name].values]))
                    res["recent_test"] = {src: scores(g[attr.name].values, final.predict(X_search[g.row]))
                                          for src, g in r_te.groupby("source")}
                    res["myntra_test_with_recent"] = scores(y[te], final.predict(X[te]))
                else:
                    final.fit(Xf, yf)
                probs = final.predict_proba(X_search[applies])
                if attr.name == "article_type":
                    probs = constrain_to_category(probs, final.classes_, catalog, labels,
                                                  search.category.values[applies])
                pred, conf = final.classes_[probs.argmax(1)], probs.max(1)
            bundle["attributes"][attr.name] = {
                "method": chosen, "classes": list(classes),
                "model": None if chosen == "zero_shot" else final,
                "text_emb": text if chosen == "zero_shot" else None,
                "categories": categories,
                "test_macro_f1": res[chosen]["test"]["macro_f1"],
                "test_accuracy": res[chosen]["test"]["accuracy"],
            }
            predictions[f"pred_{attr.name}"] = pd.Series(dtype=object)
            predictions.loc[applies, f"pred_{attr.name}"] = pred
            predictions[f"conf_{attr.name}"] = np.nan
            predictions.loc[applies, f"conf_{attr.name}"] = conf.astype(np.float32)

        line = " | ".join(f"{m} {100 * res[m]['test']['macro_f1']:.1f}" for m in methods)
        recent_line = "".join(f"  {k} acc {100 * v['accuracy']:.1f}%" for k, v in res.get("recent_test", {}).items())
        print(f"{attr.name:14s} {len(classes):3d} classes  test macro-F1: {line}  "
              f"-> {chosen}{recent_line}  ({time.time() - t0:.0f}s)", flush=True)

    out = path("reports") / f"attributes_{args.features}.json"
    out.write_text(json.dumps(report, indent=2))
    if args.features == "marqo":
        predictions.to_parquet(ROOT / "data/processed/attribute_predictions.parquet", index=False)
        models = path("models")
        models.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, models / "attributes.joblib")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
