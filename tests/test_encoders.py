"""Loads the real app model (Marqo-FashionSigLIP, ~10 s on CPU) so wrapper changes can't break the app."""

import numpy as np
import pytest
from PIL import Image

from silhouette_vision.encoders import load_encoder


@pytest.fixture(scope="module")
def marqo():
    return load_encoder("marqo_fashion_siglip", device="cpu")


def test_marqo_shapes_and_norms(marqo):
    img = Image.new("RGB", (300, 400), "white")
    image_emb = marqo.embed_images([img, img])
    text_emb = marqo.embed_texts(["a red dress", "black leather boots"])
    assert marqo.dim == 768
    assert image_emb.shape == (2, 768) and text_emb.shape == (2, 768)
    assert np.allclose(np.linalg.norm(image_emb, axis=1), 1, atol=1e-3)
    assert np.allclose(np.linalg.norm(text_emb, axis=1), 1, atol=1e-3)


def test_marqo_text_matches_image_content(marqo):
    red = Image.new("RGB", (224, 224), (200, 20, 20))
    scores = marqo.embed_images([red]) @ marqo.embed_texts(["a red image", "a blue image"]).T
    assert scores[0, 0] > scores[0, 1]
