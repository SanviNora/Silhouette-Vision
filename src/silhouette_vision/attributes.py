"""Attribute prediction from frozen image embeddings (catalog enrichment).

Labels come from Myntra (styles metadata + the structured `articleAttributes` in its style JSONs).
Three predictors are compared on a group-aware split:
  zero-shot     - nearest text prompt in the shared image-text space (no training)
  linear probe  - logistic regression on the embeddings
  mlp           - one hidden layer on the embeddings
The chosen predictor is then applied to every catalog item, including Farfetch, which has
no attribute labels at all.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

MIN_CLASS_COUNT = 50  # rarer classes are left out of an attribute's label set


@dataclass(frozen=True)
class Attribute:
    name: str
    prompt: str  # zero-shot template; {} is the class name, lower-cased
    json_key: str | None = None  # read from the Myntra articleAttributes JSON instead of a column


ATTRIBUTES = [
    Attribute("article_type", "a product photo of {}"),
    Attribute("colour", "a photo of a {} fashion item"),
    Attribute("gender", "a photo of clothing for {}"),
    Attribute("season", "a photo of {} clothing"),
    Attribute("usage", "a photo of {} wear"),
    Attribute("pattern", "a photo of a {} garment", "Pattern"),
    Attribute("fabric", "a photo of a garment made of {}", "Fabric"),
    Attribute("sleeve_length", "a photo of a top with {}", "Sleeve Length"),
    Attribute("neck", "a photo of a top with a {}", "Neck"),
    Attribute("fit", "a photo of a {} garment", "Fit"),
    Attribute("material", "a photo of a shoe or bag made of {}", "Material"),
]
MISSING = {"", "NA", "N/A", "None", None}


# Indian ethnic-wear types exist in Myntra but in none of the 2022-2026 search catalog sources:
# kept as classes they labelled western tops "Kurtis" on uploaded photos.
NOT_IN_CATALOG_TYPES = {"Kurtas", "Kurtis", "Kurta Sets", "Sarees", "Dupatta", "Lehenga Choli",
                        "Salwar", "Churidar", "Patiala", "Salwar and Dupatta"}


def label_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """One column per attribute for Myntra rows; rare or missing labels become NaN."""
    myntra = catalog[catalog.source == "myntra"]
    parsed = myntra.attributes.map(json.loads)
    out = pd.DataFrame(index=myntra.index)
    for attr in ATTRIBUTES:
        raw = parsed.map(lambda d, k=attr.json_key: d.get(k)) if attr.json_key else myntra[attr.name]
        raw = raw.where(~raw.isin(MISSING))
        if attr.name == "article_type":
            raw = raw.where(~raw.isin(NOT_IN_CATALOG_TYPES))
        counts = raw.value_counts()
        out[attr.name] = raw.where(raw.isin(counts[counts >= MIN_CLASS_COUNT].index))
    return out


def group_split(catalog: pd.DataFrame, fractions=(0.7, 0.15, 0.15)) -> pd.Series:
    """train / val / test by product group, so colour variants of one product never straddle splits.

    Group = brand + title with colour and gender words removed ("Puma Men Miami Black Slipper"
    and "Puma Unisex Miami Purple Slipper" share a group). Deterministic: hash of the group key.
    """
    colours = {c.lower() for c in catalog.colour.dropna().unique()}
    colour_words = {w for c in colours for w in c.split()} | {"multi", "coloured", "colored"}
    drop = colour_words | {"men", "women", "unisex", "boys", "girls", "kids", "s"}

    def key(row):
        words = re.findall(r"[a-z]+", str(row.title).lower())
        return f"{row.brand}|{' '.join(w for w in words if w not in drop)}"

    def bucket(k):
        return int(hashlib.md5(k.encode()).hexdigest(), 16) % 1000 / 1000

    u = catalog.apply(key, axis=1).map(bucket)
    return pd.Series(np.where(u < fractions[0], "train",
                              np.where(u < fractions[0] + fractions[1], "val", "test")),
                     index=catalog.index)


def prompt_texts(attr: Attribute, classes: list[str]) -> list[str]:
    return [attr.prompt.format(c.lower()) for c in classes]


def zero_shot(emb: np.ndarray, text_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = 100.0 * emb @ text_emb.T
    logits -= logits.max(1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    return probs.argmax(1), probs


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def applicable_categories(catalog: pd.DataFrame, labels: pd.DataFrame, attr: str,
                          min_share: float = 0.2) -> list[str]:
    """Categories where this attribute is labelled often enough to predict it (e.g. sleeve
    length for tops and dresses, not for shoes)."""
    cat = catalog.loc[labels.index, "category"]
    share = labels[attr].notna().groupby(cat).mean()
    return sorted(share[share >= min_share].index)
