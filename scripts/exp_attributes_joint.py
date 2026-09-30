"""Experiment: do the attribute heads improve on recent photos if trained on Myntra + the recent
sources' own labels, instead of Myntra alone?

Labels mapped into Myntra's vocabulary: Second-Hand colour, pattern and garment type; Amazon
(ABO) footwear type adds "Boots" and "Sandals". Test sets: Myntra's group-aware test split and
Second-Hand's official test split (disjoint garments); ABO split by photo id parity. Same model
for both arms (class-balanced logistic regression on frozen Marqo embeddings).
Usage: python scripts/exp_attributes_joint.py
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from silhouette_vision.attributes import group_split, label_frame
from silhouette_vision.catalog import load_catalog, load_legacy_catalog
from silhouette_vision.config import ROOT, path
from silhouette_vision.embed import load_embeddings
from silhouette_vision.explain import COLOUR_FAMILY
from silhouette_vision.recent_labels import recent_labels


def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X, y)


def main():
    legacy = load_legacy_catalog()
    lx = load_embeddings("marqo_fashion_siglip__legacy", legacy)
    labels = label_frame(legacy)
    split = group_split(legacy.loc[labels.index])
    cat = load_catalog()
    cx = load_embeddings("marqo_fashion_siglip", cat)
    recent = recent_labels(cat, ROOT)
    report = {}
    for attr in ["colour", "pattern", "article_type"]:
        y = labels[attr].dropna()
        s = split[y.index].values
        m_tr, m_te = y.index[s != "test"], y.index[s == "test"]
        r = recent[recent[attr].notna()]
        r_tr, r_te = r[r.split == "train"], r[r.split == "test"]
        arms = {"myntra only": (lx[m_tr], y[m_tr].values),
                "myntra + recent": (np.vstack([lx[m_tr], cx[r_tr.row]]), np.concatenate([y[m_tr].values, r_tr[attr].values]))}
        for arm, (X, Y) in arms.items():
            model = fit(X, Y)
            out = {}
            for test_name, Xt, Yt in [("myntra", lx[m_te], y[m_te].values),
                                       *[(src, cx[g.row], g[attr].values) for src, g in r_te.groupby("source")]]:
                p = model.predict(Xt)
                out[test_name] = {"accuracy": float(accuracy_score(Yt, p)),
                                  "macro_f1": float(f1_score(Yt, p, average="macro")), "n": len(Yt)}
                if attr == "colour":
                    fam = lambda v: COLOUR_FAMILY.get(v, v)
                    out[test_name]["family"] = float(np.mean([fam(a) == fam(b) for a, b in zip(p, Yt, strict=True)]))
            report.setdefault(attr, {})[arm] = out
            print(f"{attr:13s} {arm:16s} " + "  ".join(f"{k}: {100 * v['accuracy']:.1f}% (F1 {100 * v['macro_f1']:.1f})"
                                                     for k, v in out.items()), flush=True)
    (path("reports") / "exp_attributes_joint.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
