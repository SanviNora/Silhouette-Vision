import numpy as np
import pandas as pd
from PIL import Image

from silhouette_vision.attributes import expected_calibration_error, group_split
from silhouette_vision.detect import Detection, _iou, lookbench_to_fashionpedia
from silhouette_vision.enrich import item_tags


def test_crop_pads_and_stays_inside_image():
    img = Image.new("RGB", (100, 200))
    crop = Detection("dress", 0.9, (10, 20, 60, 180)).crop(img, pad=0.1)
    assert crop.size == (60, 192)  # 50 + 2*5 wide, 160 + 2*16 tall
    edge = Detection("shoe", 0.9, (0, 0, 100, 200)).crop(img, pad=0.2)
    assert edge.size == (100, 200)


def test_iou():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_lookbench_category_mapping():
    assert lookbench_to_fashionpedia("Blouse") == {"shirt, blouse"}
    assert lookbench_to_fashionpedia("jeans") == {"pants"}
    assert lookbench_to_fashionpedia("coat") == {"coat"}


def test_group_split_keeps_colour_and_gender_variants_together():
    df = pd.DataFrame({
        "title": ["Puma Men Miami Black Slipper", "Puma Unisex Miami Purple Slipper",
                  "Nike Women Air Max Red", "Nike Women Air Max Blue"],
        "brand": ["Puma", "Puma", "Nike", "Nike"],
        "colour": ["Black", "Purple", "Red", "Blue"],
    })
    split = group_split(df)
    assert split.iloc[0] == split.iloc[1]
    assert split.iloc[2] == split.iloc[3]


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2, dtype=bool)
    assert expected_calibration_error(conf, correct) < 1e-9


def test_item_tags_hide_untrusted_attributes_on_farfetch():
    row = pd.Series({"pred_colour": "Black", "conf_colour": 0.9, "pred_pattern": "Printed",
                     "conf_pattern": 0.8, "pred_sleeve_length": "Long Sleeves",
                     "conf_sleeve_length": 0.3, "pred_neck": None})
    assert item_tags(row, "farfetch") == ["Black", "Printed"]  # low-confidence sleeves dropped
