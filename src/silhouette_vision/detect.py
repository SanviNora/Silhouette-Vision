"""Garment detection, so a search can focus on one item in a busy or multi-item photo.

YOLOS fine-tuned on Fashionpedia (valentinafevu/yolos-fashionpedia, MIT license): 46 classes, of
which the first 27 are whole garments/accessories and the rest are parts (sleeve, collar,
neckline, pocket, ...). Only whole items are returned. ~0.3 s per photo on CPU.
"""

from dataclasses import dataclass

import torch
from PIL import Image

MODEL_ID = "valentinafevu/yolos-fashionpedia"
GARMENTS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants",
    "shorts", "skirt", "coat", "dress", "jumpsuit", "cape", "glasses", "hat",
    "headband, head covering, hair accessory", "tie", "glove", "watch", "belt", "tights, stockings",
    "sock", "shoe", "bag, wallet", "scarf", "umbrella",
}
FRIENDLY = {"shirt, blouse": "shirt / blouse", "top, t-shirt, sweatshirt": "top / t-shirt",
            "headband, head covering, hair accessory": "hair accessory", "bag, wallet": "bag",
            "tights, stockings": "tights"}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

    @property
    def name(self) -> str:
        return FRIENDLY.get(self.label, self.label)

    def crop(self, image: Image.Image, pad: float = 0.05) -> Image.Image:
        x0, y0, x1, y1 = self.box
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        return image.crop((max(0, x0 - px), max(0, y0 - py),
                           min(image.width, x1 + px), min(image.height, y1 + py)))


class GarmentDetector:
    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
        self.threshold, self.device = threshold, device

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[Detection]:
        """Whole garments, best first; overlapping boxes of the same label are merged by NMS."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.processor.post_process_object_detection(
            self.model(**inputs), threshold=self.threshold, target_sizes=[image.size[::-1]])[0]
        id2label = self.model.config.id2label
        dets = [Detection(id2label[int(lbl)], float(s), tuple(float(v) for v in b))
                for lbl, s, b in zip(out["labels"], out["scores"], out["boxes"], strict=True)
                if id2label[int(lbl)] in GARMENTS]
        dets.sort(key=lambda d: -d.score)
        kept: list[Detection] = []
        for d in dets:
            if all(k.label != d.label or _iou(k.box, d.box) < 0.5 for k in kept):
                kept.append(d)
        return kept


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


_LOOKBENCH = {
    "blouse": {"shirt, blouse"}, "shirt": {"shirt, blouse"},
    "t-shirt": {"top, t-shirt, sweatshirt"}, "sweatshirt": {"top, t-shirt, sweatshirt"},
    "hoodie": {"top, t-shirt, sweatshirt"}, "bag": {"bag, wallet"}, "handbag": {"bag, wallet"},
    "tote bag": {"bag, wallet"}, "clutch bag": {"bag, wallet"}, "boot": {"shoe"}, "sneaker": {"shoe"},
    "shoes": {"shoe"}, "jeans": {"pants"}, "headband": {"headband, head covering, hair accessory"},
    "tights": {"tights, stockings"},
}


def lookbench_to_fashionpedia(category: str) -> set[str]:
    """LookBench category -> Fashionpedia label(s); most names (coat, dress, skirt...) are shared."""
    c = category.strip().lower()
    return _LOOKBENCH.get(c, {c})
