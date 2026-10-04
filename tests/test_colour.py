import numpy as np
from PIL import Image

from silhouette_vision.colour import palette


def test_palette_ignores_white_background_and_orders_by_share():
    img = np.full((100, 100, 3), 255, np.uint8)  # studio white
    img[10:70, 10:90] = (90, 55, 35)  # brown body: 4,800 px
    img[75:95, 10:90] = (200, 160, 100)  # tan strap: 1,600 px
    colours = palette(Image.fromarray(img))
    assert len(colours) == 2
    (c1, s1), (c2, s2) = colours
    assert np.abs(np.array(c1) - (90, 55, 35)).max() <= 8 and np.abs(np.array(c2) - (200, 160, 100)).max() <= 8
    assert abs(s1 - 0.75) < 0.02 and abs(s1 + s2 - 1) < 1e-6
