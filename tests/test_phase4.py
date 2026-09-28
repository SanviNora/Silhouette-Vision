import numpy as np
import pandas as pd
import torch
from PIL import Image

from silhouette_vision.explain import compare_attributes, occlusion_map, overlay


def _attr(name, value, conf=0.9):
    return {"attribute": name, "value": value, "confidence": conf}


def test_compare_attributes_same_similar_different_and_farfetch_trust():
    query = [_attr("article_type", "Tops"), _attr("colour", "Navy Blue"), _attr("pattern", "Solid"),
             _attr("sleeve_length", "Sleeveless"), _attr("neck", "V-Neck", conf=0.3)]
    item = pd.Series({"pred_article_type": "Tops", "conf_article_type": 0.9,
                      "pred_colour": "Blue", "conf_colour": 0.8,
                      "pred_pattern": "Striped", "conf_pattern": 0.7,
                      "pred_sleeve_length": "No Sleeves", "conf_sleeve_length": 0.9,
                      "pred_neck": "V-Neck", "conf_neck": 0.9})
    rel = {c["label"]: c["relation"] for c in compare_attributes(query, item, "myntra")}
    assert rel == {"Type": "same", "Colour": "similar", "Pattern": "different", "Sleeves": "same"}
    # Neck is left out: the photo's prediction (30%) is not confident enough.


class _TopLeftEncoder:
    """Toy encoder whose embedding only 'looks at' the top-left quarter of the image."""

    def preprocess(self, image):
        return torch.ones(3, 16, 16)

    def embed_pixels(self, pixels):
        signal = pixels[:, :, :8, :8].mean((1, 2, 3))
        v = torch.stack([signal, torch.full_like(signal, 0.5)], 1)
        return torch.nn.functional.normalize(v, dim=1).numpy()


def test_occlusion_map_finds_the_region_the_model_uses():
    heat = occlusion_map(_TopLeftEncoder(), Image.new("RGB", (32, 32)), np.array([1.0, 0.0]), grid=4)
    r, c = np.unravel_index(heat.argmax(), heat.shape)
    assert r < 2 and c < 2
    assert heat[3, 3] == 0  # no window covering the far corner touches the top-left quarter


def test_overlay_keeps_image_size():
    img = Image.new("RGB", (40, 60), (200, 200, 200))
    assert overlay(img, np.random.rand(8, 8)).size == (40, 60)
