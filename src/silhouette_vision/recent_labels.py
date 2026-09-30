"""Labels of the recent catalog sources, mapped into the Myntra vocabulary the attribute heads use.

Second-Hand Fashion: colour (single-colour garments only; 40% list several), pattern and garment
type, with the dataset's own train/test split (eval gold set counts as test). Amazon footwear:
"Boots" (a class Myntra lacks) and "Sandals"; plain "shoes" is too vague to map.
"""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

SH_COLOUR = {c: c for c in ["Black", "Blue", "White", "Grey", "Pink", "Beige", "Green", "Red", "Brown",
                            "Purple", "Yellow", "Orange"]} | {"Multicolor": "Multi", "Turquoise": "Turquoise Blue"}
SH_PATTERN = {"None": "Solid", "Plain": "Solid", "Striped": "Striped", "Checkered print": "Checked",
              "Floral print": "Printed", "Animal print": "Printed", "Logo print": "Printed",
              "Geometric print": "Printed", "Dots": "Printed", "Lace": "Lace", "Embroidered": "Embellished"}
SH_TYPE = {"T-shirt": "Tshirts", "Top": "Tops", "Tank top": "Tops", "Blouse": "Tops", "Training top": "Tops",
           "Shirt": "Shirts", "Sweater": "Sweaters", "Cardigan": "Sweaters", "Hoodie": "Sweatshirts",
           "Dress": "Dresses", "Trousers": "Trousers", "Jeans": "Jeans", "Skirt": "Skirts",
           "Shorts": "Shorts", "Jacket": "Jackets", "Jacker": "Jackets", "Blazer": "Jackets",
           "Winter jacket": "Jackets", "Outerwear": "Jackets", "Denim jacket": "Jackets",
           "Tights": "Leggings", "Tunic": "Tunics", "Pajamas": "Night suits",
           "Night gown": "Nightdress", "Nightgown": "Nightdress"}
ABO_TYPE = {"boot": "Boots", "sandal": "Sandals"}


def recent_labels(catalog: pd.DataFrame, root: Path) -> pd.DataFrame:
    """One row per labelled recent product: catalog row, source, split, colour, pattern, article_type."""
    row_of = pd.Series(np.arange(len(catalog)), index=catalog.item_id)
    raw = pd.read_parquet(root / "data/raw/secondhand/labels.parquet")
    raw["item_id"] = "sh_" + raw.garment_id.astype(str)
    raw = raw[raw.item_id.isin(row_of.index)]
    colours = raw.colors.map(lambda c: list(c) if c is not None else [])
    sh = pd.DataFrame({
        "row": row_of[raw.item_id].values, "source": "secondhand",
        "split": raw.split.map({"train": "train"}).fillna("test").values,
        "colour": [SH_COLOUR.get(c[0]) if len(c) == 1 else None for c in colours],
        "pattern": raw.pattern.fillna("None").map(SH_PATTERN).values,
        "article_type": raw["type"].map(SH_TYPE).values,
    })
    abo = catalog[catalog.source == "abo"]
    ab = pd.DataFrame({
        "row": row_of[abo.item_id].values, "source": "abo",
        "split": np.where(abo.source_id.map(lambda s: zlib.crc32(s.encode()) % 5 == 0), "test", "train"),
        "colour": None, "pattern": None, "article_type": abo.item_type.map(ABO_TYPE).values,
    })
    return pd.concat([sh, ab], ignore_index=True)
