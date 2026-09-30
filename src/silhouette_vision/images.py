"""Image loading shared by embedding, the app, and preprocessing."""

from pathlib import Path

from PIL import Image


def load_rgb(path: str | Path, max_side: int | None = None) -> Image.Image:
    """Open an image as RGB, compositing transparency onto white.

    Visuelle PNGs are RGBA with transparent backgrounds; a plain .convert("RGB") would turn
    those areas black. `max_side` uses JPEG draft mode so large Myntra photos decode fast.
    """
    img = Image.open(path)
    if max_side and img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background
    else:
        img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def secondhand_view(img: Image.Image) -> Image.Image:
    """Second-Hand Fashion photo -> the garment, upright.

    Garments lie sideways (collar pointing right) on a sorting table and fill 20-50% of the
    frame. Rotating upright and cropping the centre raised type 10-NN agreement from 50.7% to
    65.6% and zero-shot type accuracy from 47.3% to 62.0% on 2,000 garments; a garment-mask
    crop and a detector crop did worse or no better (scripts/exp_secondhand_view.py).
    """
    img = img.convert("RGB").rotate(90, expand=True)
    w, h = img.size
    return img.crop((int(0.08 * w), int(0.18 * h), int(0.92 * w), int(0.82 * h)))
