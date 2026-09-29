"""Build the unified product catalog from Myntra and Farfetch.

One row per product, with image paths relative to the project root:

item_id | source | source_id | image_path | title | brand | gender | category | article_type |
colour | price | currency | on_sale | discount_pct | is_preowned | stock | label | description |
attributes (JSON) | year | season | usage
"""

import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from silhouette_vision.config import DATA_ROOT, load_config

MYNTRA_DIR = Path("data/raw/fashion_product_images/fashion-dataset")
FARFETCH_DIR = Path("data/raw/farfetch")
MYNTRA_THUMBS = Path("data/processed/thumbs/myntra")

# Myntra articleAttributes kept as labelled "why it matches" features.
MYNTRA_ATTRIBUTES = ["Pattern", "Fabric", "Sleeve Length", "Fit", "Neck", "Material", "Occasion", "Type"]


def _strip_html(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip() or None


def _read_myntra_json(file: Path) -> dict:
    d = json.loads(file.read_text())["data"]
    attrs = d.get("articleAttributes") or {}
    return {
        "source_id": str(d["id"]),
        "title": d.get("productDisplayName"),
        "brand": d.get("brandName"),
        "gender": d.get("gender"),
        "master_category": (d.get("masterCategory") or {}).get("typeName"),
        "sub_category": (d.get("subCategory") or {}).get("typeName"),
        "article_type": (d.get("articleType") or {}).get("typeName"),
        "colour": d.get("baseColour"),
        "price": d.get("price"),
        "discounted_price": d.get("discountedPrice"),
        "year": d.get("year"),
        "season": d.get("season") or None,
        "usage": d.get("usage") or None,
        "description": _strip_html(((d.get("productDescriptors") or {}).get("description") or {}).get("value")),
        "attributes": json.dumps({k: attrs[k] for k in MYNTRA_ATTRIBUTES if attrs.get(k)}),
    }


def build_myntra() -> pd.DataFrame:
    tax = load_config("taxonomy")["myntra"]
    files = sorted((DATA_ROOT / MYNTRA_DIR / "styles").glob("*.json"))
    with ThreadPoolExecutor(8) as pool:
        df = pd.DataFrame(pool.map(_read_myntra_json, files))

    has_image = df.source_id.map(lambda i: (DATA_ROOT / MYNTRA_DIR / "images" / f"{i}.jpg").exists())
    df = df[has_image & ~df.master_category.isin(tax["exclude_master_categories"])].copy()

    df["category"] = (
        df.article_type.map(tax["article_type_map"])
        .fillna(df.sub_category.map(tax["subcategory_map"]))
        .fillna("other")
    )
    df["price"] = pd.to_numeric(df.price, errors="coerce").where(lambda p: p > 0)
    discounted = pd.to_numeric(df.discounted_price, errors="coerce")
    df["on_sale"] = discounted < df.price
    df["discount_pct"] = (1 - discounted / df.price).where(df.on_sale, 0.0).round(3)
    df["year"] = pd.to_numeric(df.year, errors="coerce").astype("Int64")
    return df.assign(
        source="myntra",
        item_id="myn_" + df.source_id,
        raw_image_path=str(MYNTRA_DIR / "images") + "/" + df.source_id + ".jpg",
        image_path=str(MYNTRA_THUMBS) + "/" + df.source_id + ".jpg",
        currency="INR",
        is_preowned=False,
        stock=pd.NA,
        label=None,
    )


def _farfetch_category(titles: pd.Series) -> pd.Series:
    rules = load_config("taxonomy")["farfetch_title_rules"]
    lower = titles.fillna("").str.lower()
    category = pd.Series("other", index=titles.index)
    unmatched = pd.Series(True, index=titles.index)
    for name, pattern in rules:
        hit = unmatched & lower.str.contains(rf"\b(?:{pattern})(?:e?s)?\b", regex=True)
        category[hit] = name
        unmatched &= ~hit
    return category


def build_farfetch() -> pd.DataFrame:
    raw = pd.read_csv(DATA_ROOT / FARFETCH_DIR / "current_farfetch_listings.csv")
    raw = raw.drop_duplicates("id").reset_index(drop=True)
    source_id = raw["id"].astype(str)
    cutout = raw["images.cutOut"].str.rsplit("/", n=1).str[-1]
    brand = raw["brand.name"]
    initial, final = raw["priceInfo.initialPrice"], raw["priceInfo.finalPrice"]
    df = pd.DataFrame(
        {
            "item_id": "ff_" + source_id,
            "source": "farfetch",
            "source_id": source_id,
            "image_path": str(FARFETCH_DIR / "cutout-img/cutout") + "/" + cutout,
            "raw_image_path": str(FARFETCH_DIR / "cutout-img/cutout") + "/" + cutout,
            "model_image_path": str(FARFETCH_DIR / "model-img/model") + "/"
            + raw["images.model"].str.rsplit("/", n=1).str[-1],
            "title": raw["shortDescription"],
            "brand": brand.str.replace(r"\s+Pre-Owned$", "", regex=True),
            "is_preowned": brand.str.endswith("Pre-Owned", na=False),
            "gender": raw["gender"].str.title(),
            "category": _farfetch_category(raw["shortDescription"]),
            "article_type": None,
            "colour": None,
            "price": final.astype(float),
            "initial_price": initial.astype(float),
            "currency": raw["priceInfo.currencyCode"],
            "on_sale": raw["priceInfo.isOnSale"].astype(bool),
            "discount_pct": (1 - final / initial).round(3),
            "stock": raw["stockTotal"].astype("Int64"),
            "label": raw["merchandiseLabel"],
            "merchant_id": raw["merchantId"],
            "description": None,
            "attributes": "{}",
            "year": pd.array([2019] * len(raw), dtype="Int64"),
        }
    )
    return df[df.image_path.map(lambda p: (DATA_ROOT / p).exists())]


COLUMNS = [
    "item_id", "source", "source_id", "image_path", "raw_image_path", "model_image_path",
    "title", "brand", "gender", "category", "article_type", "colour", "price", "initial_price",
    "currency", "on_sale", "discount_pct", "is_preowned", "stock", "label", "merchant_id",
    "description", "attributes", "year", "season", "usage",
]


def build_catalog() -> pd.DataFrame:
    df = pd.concat([build_myntra(), build_farfetch()], ignore_index=True)
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    return df[COLUMNS].reset_index(drop=True)


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(DATA_ROOT / "data/processed/catalog.parquet")
