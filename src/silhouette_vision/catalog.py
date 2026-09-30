"""Product catalogs.

Search catalog (data/processed/catalog.parquet): recent products, all shareable:
  ZooClaw-Fashion (2026, CC BY-NC 4.0), LookBench gallery (2025, Apache-2.0),
  Second-Hand Fashion (2022-24, CC BY 4.0), and footwear from Amazon Berkeley Objects
  (c. 2019-2021, CC BY 4.0), because the recent sources carry almost no shoes.
Legacy catalog (data/processed/legacy_catalog.parquet): Myntra (2007-2018, MIT) + Farfetch (2019,
  scraped, analysis only). Kept locally to train the attribute heads and reproduce Phases 1-4.

One row per product, image paths relative to the project root.
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


def category_from_text(titles: pd.Series) -> pd.Series:
    """Shared category from product words (taxonomy.yml title rules; first matching rule wins)."""
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
            "category": category_from_text(raw["shortDescription"]),
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


LEGACY_COLUMNS = [
    "item_id", "source", "source_id", "image_path", "raw_image_path", "model_image_path",
    "title", "brand", "gender", "category", "article_type", "colour", "price", "initial_price",
    "currency", "on_sale", "discount_pct", "is_preowned", "stock", "label", "merchant_id",
    "description", "attributes", "year", "season", "usage",
]


def build_legacy_catalog() -> pd.DataFrame:
    df = pd.concat([build_myntra(), build_farfetch()], ignore_index=True)
    for col in LEGACY_COLUMNS:
        if col not in df:
            df[col] = None
    return df[LEGACY_COLUMNS].reset_index(drop=True)


def load_legacy_catalog() -> pd.DataFrame:
    return pd.read_parquet(DATA_ROOT / "data/processed/legacy_catalog.parquet")


# --- search catalog: recent sources ---------------------------------------------------------
ZOOCLAW_DIR = Path("data/raw/zooclaw")
LOOKBENCH_DIR = Path("data/raw/lookbench/v20251201")
SECONDHAND_DIR = Path("data/raw/secondhand")
THUMBS = Path("data/processed/thumbs")

COLUMNS = [
    "item_id", "source", "source_id", "image_path", "raw_image_path", "title", "brand", "gender",
    "category", "item_type", "colour", "pattern", "material", "price_band", "is_preowned", "year",
]
LICENSES = {"zooclaw": "CC BY-NC 4.0", "lookbench": "Apache-2.0", "secondhand": "CC BY 4.0", "abo": "CC BY 4.0"}
KIDS = {"kids", "boys", "girls", "teen", "children", "baby", "youth", "toddler", "babies", "child",
        "juniors"}
UPPER = {"Mm6": "MM6", "Jw": "JW", "Dkny": "DKNY", "Ck": "CK", "Msgm": "MSGM", "Apc": "APC",
         "A.p.c.": "A.P.C.", "Hugo Boss": "Hugo Boss"}


def _brand(name) -> str | None:
    if not isinstance(name, str) or name.strip().lower() in {"", "not in the list", "unknown", "missing",
                                                              "not applicable"}:
        return None
    cap = lambda w: w[:1].upper() + w[1:].lower()  # noqa: E731  (not str.title: "Levi'S")
    words = [UPPER.get(cap(w), cap(w)) for w in name.strip().split()]
    return " ".join(words).replace("H&m", "H&M").replace("J.crew", "J.Crew")


def _gender(value) -> str | None:
    v = (value or "").strip().lower()
    if v in {"women", "woman", "ladies", "ladys", "female"}:
        return "Women"
    if v in {"men", "man", "male"}:
        return "Men"
    if v in {"unisex", "adults", "adult"}:
        return "Unisex"
    return "Kids" if v in KIDS else None


def _clean_title(title: str, brand: str | None) -> str:
    """Display title: without a leading brand name or a trailing size ("... - size m")."""
    t = re.sub(r"\s+-\s+size\s+\S+$", "", title.strip(), flags=re.IGNORECASE)
    if isinstance(brand, str) and t.lower().startswith(brand.lower() + " "):
        t = t[len(brand) + 1:]
    return t[:1].upper() + t[1:]


def build_zooclaw() -> pd.DataFrame:
    z = pd.read_parquet(DATA_ROOT / ZOOCLAW_DIR / "corpus.parquet")
    sid = z.corpus_id.astype(str)
    kind = z.category.map({"one-piece": "dress", "bottom": "bottoms"}).fillna(z.category)
    return pd.DataFrame({
        "item_id": "zc_" + sid, "source": "zooclaw", "source_id": sid,
        "raw_image_path": str(ZOOCLAW_DIR / "images") + "/" + sid + ".jpg",
        "title": [_clean_title(t, b) for t, b in zip(z.title, z.brand.map(_brand), strict=True)],
        "brand": z.brand.map(_brand), "gender": z.demographic.map(_gender),
        "category": kind.where(kind.isin(load_config("taxonomy")["categories"]), "other"),
        "item_type": z.category, "is_preowned": False, "year": 2026,
    })


def build_lookbench() -> pd.DataFrame:
    """The 1,078 products of LookBench's real studio gallery, one photo each. Its 58k "noise"
    images are left out: they match Fashion200k (2017) near-exactly, so they are not recent."""
    import pyarrow.parquet as pq

    g = pq.read_table(DATA_ROOT / LOOKBENCH_DIR / "real_studio_flat/gallery.parquet",
                      columns=["item_ID", "category", "main_attribute"]).to_pandas()
    g = g.reset_index().drop_duplicates("item_ID")
    words = g.main_attribute.fillna("").str.replace("_", " ")
    kind = g.category.str.lower()
    title = (words.where(~words.isin(["", "plain", "casual", "elegant"]), "") + " " + kind).str.strip()
    sid = g.item_ID.astype(str)
    return pd.DataFrame({
        "item_id": "lb_" + sid, "source": "lookbench", "source_id": sid,
        "raw_image_path": str(LOOKBENCH_DIR / "images") + "/" + sid + ".jpg",
        "title": title, "brand": None, "gender": None, "category": category_from_text(kind),
        "item_type": kind, "is_preowned": False, "year": 2025, "gallery_row": g["index"].values,
    })


def build_secondhand() -> pd.DataFrame:
    d = pd.read_parquet(DATA_ROOT / SECONDHAND_DIR / "labels.parquet")
    d = d[d.garment_id.map(lambda i: (DATA_ROOT / SECONDHAND_DIR / "images" / f"{i}.jpg").exists())]
    sid = d.garment_id.astype(str)
    kind = d["type"].fillna("")
    pattern = d.pattern.where(~d.pattern.isin([None, "None", "Plain", "Other"]))
    colour = d.colors.map(lambda c: str(c[0]).title() if c is not None and len(c) else None)
    return pd.DataFrame({
        "item_id": "sh_" + sid, "source": "secondhand", "source_id": sid,
        "raw_image_path": str(SECONDHAND_DIR / "images") + "/" + sid + ".jpg",
        "title": (pattern.fillna("").str.lower() + " " + kind.str.lower()).str.strip().str.capitalize(),
        "brand": d.brand.map(_brand), "gender": d.category.map(_gender),
        "category": category_from_text(kind), "item_type": kind, "colour": colour,
        "pattern": pattern, "material": d.material, "price_band": d.price, "is_preowned": True,
        "year": pd.to_numeric(d.timestamp.str[:4], errors="coerce").astype("Int64"),
    })


ABO_DIR = Path("data/raw/abo")
GENDER_WORDS = {"women's": "Women", "womens": "Women", "men's": "Men", "mens": "Men", "girls'": "Kids",
                "girl's": "Kids", "boys'": "Kids", "boy's": "Kids", "kids'": "Kids", "unisex": "Unisex",
                "unisex-adult": "Unisex", "unisex-child": "Kids", "baby": "Kids"}


def _abo_name(name: str, brand: str | None) -> dict:
    """"Amazon Brand - The Fix Women's Kennedi Slouch Boot, Black, 8.5 B US" ->
    brand The Fix, title "Kennedi Slouch Boot", gender Women, colour Black."""
    brand = re.sub(r"^Amazon Brand\s*-?\s*", "", brand or "").strip() or None
    name = re.sub(r"^Amazon Brand\s*-?\s*", "", name)
    # Indian listings: "Beige Formal Shoes-9 UK (43 EU) (10 US) (AZ-SY-435)": drop size and codes
    name = re.sub(r"[-\s]*\d+(\.\d+)?\s*(UK|US|EU)\b.*$", "", name)
    name = re.sub(r"\s*\([^)]*\d[^)]*\)\s*$", "", name)
    head, *rest = name.split(",")
    if brand and head.lower().startswith(brand.lower()):
        head = head[len(brand):]
    gender = None
    for word, g in GENDER_WORDS.items():
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", head, re.IGNORECASE):
            gender = gender or g
            head = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", "", head, flags=re.IGNORECASE)
    colour = rest[0].strip(" ()") if rest and not re.search(r"\d", rest[0]) else None
    title = re.sub(r"\s+", " ", head).strip(" -")
    return {"brand": _brand(brand), "title": title[:1].upper() + title[1:], "gender": gender,
            "colour": colour.split("(")[0].strip().title() if colour else None}


def build_abo() -> pd.DataFrame:
    """Footwear from Amazon Berkeley Objects (CC BY 4.0; listings c. 2019-2021)."""
    a = pd.read_parquet(DATA_ROOT / ABO_DIR / "footwear.parquet")
    a = a[a.main_image_id.map(lambda i: (DATA_ROOT / ABO_DIR / "images" / f"{i}.jpg").exists())]
    parsed = pd.DataFrame([_abo_name(n, b) for n, b in zip(a.name, a.brand, strict=True)], index=a.index)
    sid = a.main_image_id.astype(str)
    return pd.DataFrame({
        "item_id": "abo_" + sid, "source": "abo", "source_id": sid,
        "raw_image_path": str(ABO_DIR / "images") + "/" + sid + ".jpg",
        "title": parsed.title, "brand": parsed.brand, "gender": parsed.gender, "category": "shoes",
        "item_type": a["type"].str.lower(), "colour": parsed.colour, "is_preowned": False, "year": 2021,
    })


def build_catalog() -> pd.DataFrame:
    df = pd.concat([build_zooclaw(), build_lookbench(), build_secondhand(), build_abo()], ignore_index=True)
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    df["image_path"] = str(THUMBS) + "/" + df.source + "/" + df.source_id + ".jpg"
    df["license"] = df.source.map(LICENSES)
    return df[[*COLUMNS, "license", "gallery_row"]].reset_index(drop=True)


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(DATA_ROOT / "data/processed/catalog.parquet")
