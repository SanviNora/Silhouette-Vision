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
