import pandas as pd
import pytest

from silhouette_vision.catalog import category_from_text
from silhouette_vision.config import ROOT, load_config

CATALOG = ROOT / "data/processed/catalog.parquet"


def test_farfetch_title_rules_prefer_specific_matches():
    titles = pd.Series([
        "silk shirt dress", "leather jacket and trousers suit", "Superstar sneakers",
        "Teddy Bear logo shoulder bag", "gold hoop earrings", "cashmere sweater",
        "high-waisted jeans", "Nautico tiger print bikini", "Metropolis",
    ])
    assert category_from_text(titles).tolist() == [
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
    assert set(catalog.item_id.str.split("_").str[0]) == {"zc", "lb", "sh"}


def test_catalog_categories_licenses_and_years(catalog):
    assert set(catalog.category) <= set(load_config("taxonomy")["categories"])
    assert catalog.license.notna().all()
    assert catalog.year.min() >= 2022  # recent products only


def test_catalog_images_exist(catalog):
    sample = catalog.sample(200, random_state=0)
    missing = [p for p in sample.image_path if not (ROOT / p).exists()]
    assert not missing, missing[:5]


def test_brand_names_are_display_ready():
    from silhouette_vision.catalog import _brand

    assert _brand("h&m") == "H&M"
    assert _brand("mm6 maison margiela") == "MM6 Maison Margiela"
    assert _brand("Not in the list") is None and _brand("") is None
