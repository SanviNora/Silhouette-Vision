import json

import numpy as np
import pandas as pd
import pytest

from silhouette_vision.catalog import _farfetch_category
from silhouette_vision.config import ROOT, load_config

CATALOG = ROOT / "data/processed/catalog.parquet"


def test_farfetch_title_rules_prefer_specific_matches():
    titles = pd.Series([
        "silk shirt dress", "leather jacket and trousers suit", "Superstar sneakers",
        "Teddy Bear logo shoulder bag", "gold hoop earrings", "cashmere sweater",
        "high-waisted jeans", "Nautico tiger print bikini", "Metropolis",
    ])
    assert _farfetch_category(titles).tolist() == [
        "dress", "outerwear", "shoes", "bag", "jewellery", "top", "bottoms", "other", "other",
    ]


def test_taxonomy_only_uses_declared_categories():
    tax = load_config("taxonomy")
    allowed = set(tax["categories"])
    myntra = tax["myntra"]
    assert set(myntra["article_type_map"].values()) <= allowed
    assert set(myntra["subcategory_map"].values()) <= allowed
    assert {name for name, _ in tax["farfetch_title_rules"]} <= allowed


@pytest.fixture(scope="module")
def catalog():
    if not CATALOG.exists():
        pytest.skip("catalog not built; run scripts/01_build_catalog.py")
    return pd.read_parquet(CATALOG)


def test_catalog_ids_unique_and_prefixed(catalog):
    assert catalog.item_id.is_unique
    assert set(catalog.item_id.str.split("_").str[0]) == {"myn", "ff"}


def test_catalog_categories_and_currency(catalog):
    assert set(catalog.category) <= set(load_config("taxonomy")["categories"])
    currency = catalog.groupby("source").currency.unique().apply(list).to_dict()
    assert currency == {"farfetch": ["SGD"], "myntra": ["INR"]}


def test_catalog_images_exist(catalog):
    sample = catalog.sample(200, random_state=0)
    missing = [p for p in sample.image_path if not (ROOT / p).exists()]
    assert not missing, missing[:5]


def test_catalog_attributes_are_json(catalog):
    parsed = catalog.attributes.head(500).map(json.loads)
    assert parsed.map(lambda d: isinstance(d, dict)).all()


def test_prices_positive(catalog):
    assert (catalog.price.dropna() > 0).all()
    assert np.isfinite(catalog.price.dropna()).all()
