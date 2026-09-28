"""Named-model recognition: "this is a Louis Vuitton Pochette Félicie" (scripts/07_named_models.py).

Exact-product search can only find what the catalog stocks, and our catalog is a 2019 snapshot:
it has 767 Louis Vuitton items but no Pochette Félicie. Model names, however, are known to
Marqo-FashionSigLIP from its training text, so we match the photo zero-shot against prompts for
the iconic models listed in configs/iconic_models.yml. A calibrator on [top score, gap to 2nd]
turns this into P(the named model is right), and doubles as the "none of these" rejection for
photos of other products.
"""

import json
import re
from dataclasses import dataclass

import numpy as np

from silhouette_vision.config import load_config, path

TEMPLATES = ["a photo of the {brand} {name} {kind}", "{brand} {name}"]
# Item type is decided first and only models of that type compete: without this, red T-bar
# pumps were named "Miu Miu Arcadie" (a bag) at 57%.
KIND_GROUPS = {"bag": "bag", "wallet": "bag", "shoes": "shoes", "belt": "belt", "outerwear": "outerwear"}
KIND_PROMPTS = {"bag": "a photo of a handbag", "shoes": "a photo of shoes",
                "belt": "a photo of a belt", "outerwear": "a photo of a coat"}


@dataclass(frozen=True)
class NamedModel:
    brand: str
    kind: str
    name: str
    aliases: tuple[str, ...]
    brand_aliases: tuple[str, ...] = ()  # catalog spellings, e.g. "Christian Dior"

    @property
    def catalog_brands(self) -> set[str]:
        return {self.brand, *self.brand_aliases}

    @property
    def label(self) -> str:
        return f"{self.brand} {self.name}"

    def pattern(self) -> re.Pattern:
        """Title matcher over the name and its aliases (whole words, accents optional)."""
        words = [re.escape(_fold(a)) for a in (self.name, *self.aliases)]
        return re.compile(r"(?<![\w])(" + "|".join(words) + r")(?![\w])", re.IGNORECASE)


def _fold(text: str) -> str:
    return (text.lower().replace("é", "e").replace("è", "e").replace("à", "a")
            .replace("ç", "c").replace("’", "'"))


def load_models() -> list[NamedModel]:
    out = []
    for brand_entry, kinds in load_config("iconic_models").items():
        brand, *brand_aliases = [s.strip() for s in brand_entry.split("|")]
        for kind, names in kinds.items():
            for entry in names:
                name, *aliases = [s.strip() for s in str(entry).split("|")]
                out.append(NamedModel(brand, kind, name, tuple(aliases), tuple(brand_aliases)))
    return out


def title_model(models: list[NamedModel], brand: str, title: str) -> NamedModel | None:
    """The one listed model a product title names (None if none or ambiguous)."""
    hits = [m for m in models if brand in m.catalog_brands and m.pattern().search(_fold(title))]
    # "Nano Speedy" also matches "Speedy": keep the most specific (longest) name.
    hits = [m for m in hits if not any(o is not m and _fold(m.name) in _fold(o.name) for o in hits)]
    return hits[0] if len(hits) == 1 else None


def prompt_matrix(encode_text, models: list[NamedModel]) -> np.ndarray:
    """One L2-normalised text embedding per model, averaged over the prompt templates."""
    vecs = np.stack([np.mean([encode_text(t.format(brand=m.brand, name=m.name, kind=m.kind))
                              for t in TEMPLATES], axis=0) for m in models])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def kind_matrix(encode_text) -> tuple[list[str], np.ndarray]:
    kinds = list(KIND_PROMPTS)
    return kinds, np.stack([encode_text(KIND_PROMPTS[k]) for k in kinds])


def gated_scores(image_vecs: np.ndarray, prompts: np.ndarray, models: list[NamedModel],
                 kinds: list[str], kind_vecs: np.ndarray) -> np.ndarray:
    """Model scores per image, with models of other item types than the image's set to -1."""
    kind = np.array(kinds)[(image_vecs @ kind_vecs.T).argmax(1)]
    model_kind = np.array([KIND_GROUPS[m.kind] for m in models])
    return np.where(model_kind[None, :] == kind[:, None], image_vecs @ prompts.T, -1.0)


def features(scores: np.ndarray) -> np.ndarray:
    """[top score, gap to 2nd] from rows of model scores."""
    s = -np.sort(-scores, axis=-1)
    return np.column_stack([s[..., 0], s[..., 0] - s[..., 1]])


class ModelRecogniser:
    def __init__(self, encode_text):
        self.models = load_models()
        self.prompts = prompt_matrix(encode_text, self.models)
        self.kinds, self.kind_vecs = kind_matrix(encode_text)
        self.params = json.loads((path("models") / "model_recognition.json").read_text())

    def recognise(self, image_vec: np.ndarray, top: int = 3) -> list[tuple[NamedModel, float]]:
        """The best-scoring models, the first with P(correct) and the rest with raw scores."""
        scores = gated_scores(image_vec[None, :], self.prompts, self.models,
                              self.kinds, self.kind_vecs)[0]
        order = np.argsort(-scores)[:top]
        x = features(scores[None, :])[0]
        z = float(np.dot(self.params["coef"], x) + self.params["intercept"])
        prob = 1.0 / (1.0 + np.exp(-z))
        return [(self.models[order[0]], prob)] + [(self.models[i], float(scores[i])) for i in order[1:]]


def catalog_listings(catalog, model: NamedModel) -> np.ndarray:
    """Catalog row positions whose brand and title name this model."""
    same_brand = np.flatnonzero(catalog.brand.isin(model.catalog_brands).values)
    pattern = model.pattern()
    return np.array([i for i in same_brand if pattern.search(_fold(catalog.title.values[i]))], dtype=int)
