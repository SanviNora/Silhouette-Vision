"""Attribute predictions for the app: for an uploaded photo (live) and catalog items (precomputed).

Models come from scripts/04_attributes.py. An attribute is only shown where it applies (sleeve
length for tops/dresses/outerwear, material for bags/shoes, ...), judged from the predicted
article type's category.

Trust: the predictors are trained on Myntra (mass-market). On luxury Farfetch items, material,
fabric, usage, season and fit showed clear domain shift (e.g. 59% of luxury bags/shoes called
"synthetic", 95% of items "casual"), so only the attributes below are displayed for Farfetch.
"""

import numpy as np
import pandas as pd

from silhouette_vision.config import ROOT, path

# Shown in the app. Hidden: gender (learned bias, e.g. women's wide-leg jeans -> "Men" at 100%),
# usage and season (weakest test macro-F1, yet predicted at ~100% confidence), fabric (44 F1).
DISPLAY_ORDER = ["article_type", "colour", "pattern", "sleeve_length", "neck", "fit", "material"]
TRUSTED_ON_FARFETCH = {"article_type", "colour", "pattern", "sleeve_length", "neck"}
LABELS = {"article_type": "Type", "colour": "Colour", "pattern": "Pattern",
          "sleeve_length": "Sleeves", "neck": "Neck", "fit": "Fit", "material": "Material",
          "fabric": "Fabric", "gender": "For", "usage": "Occasion", "season": "Season"}


class AttributePredictor:
    def __init__(self):
        import joblib

        bundle = joblib.load(path("models") / "attributes.joblib")
        self.attributes = bundle["attributes"]
        self.type_to_category = bundle["type_to_category"]

    def predict(self, embedding: np.ndarray, min_confidence: float = 0.35) -> list[dict]:
        """Attributes for one image embedding, most important first."""
        x = embedding[None, :].astype(np.float32)
        out, category = [], None
        for name in DISPLAY_ORDER:
            spec = self.attributes.get(name)
            if spec is None:
                continue
            if spec["method"] == "zero_shot":
                logits = 100.0 * x @ spec["text_emb"].T
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                classes = np.array(spec["classes"])
            else:
                probs = spec["model"].predict_proba(x)
                classes = spec["model"].classes_
            label, conf = str(classes[probs.argmax()]), float(probs.max())
            if name == "article_type":
                category = self.type_to_category.get(label)
            elif category is not None and category not in spec["categories"]:
                continue
            if conf >= min_confidence:
                out.append({"attribute": name, "label": LABELS[name], "value": label,
                            "confidence": conf, "test_accuracy": spec["test_accuracy"]})
        return out


def load_catalog_predictions() -> pd.DataFrame | None:
    file = ROOT / "data/processed/attribute_predictions.parquet"
    return pd.read_parquet(file).set_index("item_id") if file.exists() else None


def item_tags(pred_row: pd.Series, source: str, n: int = 3) -> list[str]:
    """Short attribute tags for a result card (trusted attributes only on Farfetch)."""
    tags = []
    for name in ["colour", "pattern", "sleeve_length", "neck"]:
        if source == "farfetch" and name not in TRUSTED_ON_FARFETCH:
            continue
        value = pred_row.get(f"pred_{name}")
        if isinstance(value, str) and pred_row.get(f"conf_{name}", 0) >= 0.5 and value != "Solid":
            tags.append(value)
    return tags[:n]
