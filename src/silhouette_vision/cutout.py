"""Garment cut-out for Second-Hand Fashion photos (October 2026).

Second-Hand garments are photographed lying on a sorting table with a white tile grid. BiRefNet
(MIT; dichotomous image segmentation) separates the garment from the table; the garment is put on
white, trimmed to its outline with a small margin and centred on a square, like a studio product
shot. Unlike the earlier colour-mask crop it handles white garments on white tiles. The lite model
is 2.4x faster but kept measuring tapes and cables in 6 of 100 sample photos, so the full model is
used. Masks covering under 3% or over 90% of the photo are treated as failures (caller falls back).
"""

import numpy as np
import torch
from PIL import Image

MODEL = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"  # pinned: the model ships its own code


class GarmentCutout:
    def __init__(self, device: str | None = None):
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        from silhouette_vision.encoders import default_device

        self.device = device or default_device()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True).to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.inference_mode()
    def mask(self, img: Image.Image) -> np.ndarray:
        """Foreground probability per pixel (H x W, 0-1) at the photo's size."""
        x = self.transform(img).unsqueeze(0).to(self.device, self.dtype)
        pred = self.model(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
        return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0

    def __call__(self, img: Image.Image, pad: float = 0.04) -> Image.Image | None:
        """The garment on white, trimmed and centred on a square; None if the mask looks wrong."""
        img = img.convert("RGB")
        a = self.mask(img)
        keep = a > 0.5
        area = keep.mean()
        if not 0.03 <= area <= 0.9:
            return None
        ys, xs = np.nonzero(keep)
        m = int(pad * max(img.size))
        box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, img.width), min(ys.max() + m, img.height))
        alpha = Image.fromarray((a * 255).astype(np.uint8))
        out = Image.composite(img, Image.new("RGB", img.size, "white"), alpha).crop(box)
        side = max(out.size)
        square = Image.new("RGB", (side, side), "white")
        square.paste(out, ((side - out.width) // 2, (side - out.height) // 2))
        return square
