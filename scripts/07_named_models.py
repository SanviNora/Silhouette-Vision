"""Stage 7: named-model recognition ("this is a Louis Vuitton Pochette Félicie").

Zero-shot: the photo embedding is compared with text prompts for each model in
configs/iconic_models.yml. Evaluated on catalog photos whose titles name one listed model of
their brand (positives), and on bags/shoes/accessories from unlisted brands (negatives, which
should be rejected as "none of these"). A logistic calibrator on [top score, gap to 2nd] gives
P(named model is right); fitted on even-indexed items, evaluated on odd-indexed ones.

Output: artifacts/models/model_recognition.json
Usage: python scripts/07_named_models.py
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from silhouette_vision.catalog import load_legacy_catalog
from silhouette_vision.config import path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.encoders import load_encoder
from silhouette_vision.named_models import (
    TEMPLATES,
    features,
    gated_scores,
    kind_matrix,
    load_models,
    prompt_matrix,
    title_model,
)

MODEL = "marqo_fashion_siglip"
KIND_CATEGORIES = {"bag": {"bag"}, "wallet": {"bag", "accessory", "other"}, "shoes": {"shoes"},
                   "belt": {"accessory", "other"}, "outerwear": {"outerwear"}}
N_NEGATIVES = 6000


def labelled_items(catalog, models):
    brands = {b for m in models for b in m.catalog_brands}
    ff = catalog[catalog.source == "farfetch"]
    pos = []
    for i, row in ff[ff.brand.isin(brands)].iterrows():
        m = title_model(models, row.brand, row.title)
        if m is not None and row.category in KIND_CATEGORIES[m.kind]:
            pos.append((i, models.index(m)))
    rng = np.random.default_rng(0)
    pool = catalog[catalog.category.isin(["bag", "shoes", "accessory"]) & ~catalog.brand.isin(brands)]
    neg = rng.choice(pool.index.values, min(N_NEGATIVES, len(pool)), replace=False)
    return np.array([p[0] for p in pos]), np.array([p[1] for p in pos]), neg


def main():
    catalog = load_legacy_catalog()
    emb = load_embeddings(f"{MODEL}__legacy", catalog)
    encoder = load_encoder(MODEL)
    models = load_models()
    embed_text = lambda t: encoder.embed_texts([t])[0]

    pos_rows, pos_label, neg_rows = labelled_items(catalog, models)
    print(f"{len(models)} named models; {len(pos_rows)} catalog photos name one "
          f"({len(set(pos_label))} distinct models); {len(neg_rows)} negatives from unlisted brands")

    # Prompt templates: each alone and averaged (pick on even-indexed positives).
    dev = np.arange(len(pos_rows)) % 2 == 0
    for name, tmpl in [("t0", TEMPLATES[:1]), ("t1", TEMPLATES[1:]), ("avg", TEMPLATES)]:
        P = np.stack([np.mean([embed_text(t.format(brand=m.brand, name=m.name, kind=m.kind))
                               for t in tmpl], 0) for m in models])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        acc = ((emb[pos_rows[dev]] @ P.T).argmax(1) == pos_label[dev]).mean()
        print(f"   prompts {name:3s}: dev top-1 {100 * acc:.1f}%")

    prompts = prompt_matrix(embed_text, models)
    kinds, kind_vecs = kind_matrix(embed_text)
    S_pos = gated_scores(emb[pos_rows], prompts, models, kinds, kind_vecs)
    S_neg = gated_scores(emb[neg_rows], prompts, models, kinds, kind_vecs)
    top1 = S_pos.argmax(1) == pos_label
    top3 = (np.argsort(-S_pos, 1)[:, :3] == pos_label[:, None]).any(1)
    print(f"all positives: top-1 {100 * top1.mean():.1f}%, top-3 {100 * top3.mean():.1f}%")

    X = np.vstack([features(S_pos), features(S_neg)])
    y = np.concatenate([top1, np.zeros(len(neg_rows), bool)])
    is_pos = np.concatenate([np.ones(len(pos_rows), bool), np.zeros(len(neg_rows), bool)])
    idx = np.arange(len(y))
    fit, test = idx % 2 == 0, idx % 2 == 1
    lr = LogisticRegression(C=10).fit(X[fit], y[fit])
    p = lr.predict_proba(X[test])[:, 1]
    claims = {}
    for t in (0.5, 0.7, 0.8, 0.9):
        m = p >= t
        claims[t] = {"precision": float(y[test][m].mean()) if m.any() else None,
                     "named_models_recognised": float(m[is_pos[test]].mean()),
                     "false_alarm_on_unlisted": float(m[~is_pos[test]].mean())}
        print(f"   name a model when P >= {t}: precision {100 * claims[t]['precision']:.1f}%, "
              f"recognises {100 * claims[t]['named_models_recognised']:.1f}% of listed-model photos, "
              f"false alarm {100 * claims[t]['false_alarm_on_unlisted']:.1f}% on unlisted brands")

    per_brand = {}
    for b in sorted({models[i].brand for i in pos_label}):
        m = np.array([models[i].brand == b for i in pos_label])
        per_brand[b] = {"n": int(m.sum()), "top1": float(top1[m].mean())}
    print("per brand top-1:", {b: f"{100 * v['top1']:.0f}% (n={v['n']})" for b, v in per_brand.items()})
    confused = {}
    for i in np.flatnonzero(~top1):
        key = f"{models[pos_label[i]].label} -> {models[S_pos[i].argmax()].label}"
        confused[key] = confused.get(key, 0) + 1
    print("most confused:", sorted(confused.items(), key=lambda kv: -kv[1])[:8])

    out = {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
           "features": ["top_score", "gap_to_2nd"], "n_models": len(models),
           "n_positives": len(pos_rows), "n_distinct_models": len(set(pos_label)),
           "n_negatives": len(neg_rows), "top1": float(top1.mean()), "top3": float(top3.mean()),
           "claims": claims, "per_brand": per_brand}
    (path("models") / "model_recognition.json").write_text(json.dumps(out, indent=2))
    print("saved artifacts/models/model_recognition.json")


if __name__ == "__main__":
    main()
