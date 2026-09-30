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


def test_item_tags_skip_low_confidence_and_solid():
    row = pd.Series({"pred_colour": "Black", "conf_colour": 0.9, "pred_pattern": "Printed",
                     "conf_pattern": 0.8, "pred_sleeve_length": "Long Sleeves",
                     "conf_sleeve_length": 0.3, "pred_neck": None})
    assert item_tags(row) == ["Black", "Printed"]  # low-confidence sleeves dropped


def test_detector_items_drop_parts_inside_dress_and_whole_image_boxes():
    from silhouette_vision.detect import GarmentDetector

    det = GarmentDetector.__new__(GarmentDetector)
    img = Image.new("RGB", (100, 200))
    dress = Detection("dress", 0.99, (10, 10, 90, 190))
    skirt = Detection("skirt", 0.95, (12, 100, 88, 188))  # inside the dress -> a part
    bag = Detection("bag, wallet", 0.9, (0, 150, 20, 199))  # separate item
    weak = Detection("belt", 0.55, (30, 90, 70, 100))  # below min_score
    det.detect = lambda image: [dress, skirt, bag, weak]
    assert [d.label for d in det.items(img)] == ["dress", "bag, wallet"]
    det.detect = lambda image: [Detection("skirt", 0.96, (0, 0, 100, 195))]  # product shot
    assert det.items(img) == []


def test_match_confidence_rewards_standing_out(tmp_path, monkeypatch):
    import json

    from silhouette_vision import match

    params = {"marqo": {"coef": [2.0, 30.0, 10.0], "intercept": -3.0}}
    (tmp_path / "match_confidence.json").write_text(json.dumps(params))
    monkeypatch.setattr(match, "path", lambda key: tmp_path)
    m = match.MatchConfidence()
    crowded = m.probability(np.array([0.90, 0.89, 0.89, 0.88, 0.88]))
    standout = m.probability(np.array([0.90, 0.80, 0.78, 0.77, 0.76]))
    assert standout > crowded
    assert m.tier(0.85).startswith("Very likely")
    assert m.tier(0.6).startswith("Possibly")
    assert m.tier(0.2).startswith("No confident")


def test_title_model_prefers_specific_names_and_folds_accents():
    from silhouette_vision.named_models import NamedModel, title_model

    models = [NamedModel("Louis Vuitton", "bag", "Speedy", ()),
              NamedModel("Louis Vuitton", "bag", "Nano Speedy", ()),
              NamedModel("Louis Vuitton", "bag", "Pochette Félicie", ("Felicie",)),
              NamedModel("Chanel", "bag", "Boy", ())]
    assert title_model(models, "Louis Vuitton", "Nano Speedy monogram bag").name == "Nano Speedy"
    assert title_model(models, "Louis Vuitton", "Pochette Felicie chain wallet").name == "Pochette Félicie"
    assert title_model(models, "Louis Vuitton", "Speedy 30 hand bag").name == "Speedy"
    assert title_model(models, "Chanel", "Boyfriend jeans") is None  # whole words only
    assert title_model(models, "Chanel", "Speedy bag") is None  # other brand's model


def test_gated_scores_only_compare_models_of_the_photos_item_type():
    from silhouette_vision.named_models import NamedModel, gated_scores

    models = [NamedModel("Miu Miu", "bag", "Arcadie", ()), NamedModel("Dior", "shoes", "J'Adior", ())]
    kinds, kind_vecs = ["bag", "shoes"], np.eye(2)
    prompts = np.array([[0.2, 0.9], [0.1, 0.8]])  # the bag prompt scores higher on a shoe photo
    shoe_photo = np.array([[0.0, 1.0]])
    s = gated_scores(shoe_photo, prompts, models, kinds, kind_vecs)[0]
    assert s.argmax() == 1 and s[0] == -1.0
