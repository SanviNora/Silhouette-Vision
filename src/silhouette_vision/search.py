"""Visual and text search over the catalog.

Exact search on L2-normalized embeddings: one matrix-vector product gives the cosine similarity to
every item. At ~230k items this takes tens of milliseconds in NumPy, so no ANN index (FAISS,
HNSW) is needed; FAISS was also dropped because its bundled OpenMP runtime clashes with PyTorch's.
Filters are applied before ranking, so a filtered query still returns k results.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

from silhouette_vision.catalog import load_catalog
from silhouette_vision.embed import load_embeddings
from silhouette_vision.encoders import load_encoder


@dataclass
class Filters:
    sources: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    genders: list[str] = field(default_factory=list)
    preowned: bool | None = None  # None = either

    def mask(self, catalog: pd.DataFrame) -> np.ndarray | None:
        m = np.ones(len(catalog), dtype=bool)
        if self.sources:
            m &= catalog.source.isin(self.sources).values
        if self.categories:
            m &= catalog.category.isin(self.categories).values
        if self.genders:
            m &= catalog.gender.isin(self.genders).values
        if self.preowned is not None:
            m &= (catalog.is_preowned.fillna(False) == self.preowned).values
        return None if m.all() else m


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest scores (restricted to `mask`), best first."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


class SearchEngine:
    def __init__(self, model: str = "marqo_fashion_siglip", device: str | None = None):
        self.catalog = load_catalog()
        self.embeddings = load_embeddings(model, self.catalog)
        self.encoder = load_encoder(model, device=device)

    # --- query vectors -------------------------------------------------------------------
    def image_vector(self, image: Image.Image) -> np.ndarray:
        return self.encoder.embed_images([image])[0]

    def text_vector(self, text: str) -> np.ndarray:
        return self.encoder.embed_texts([text])[0]

    def item_vector(self, item_id: str) -> np.ndarray:
        return self.embeddings[self._row(item_id)]

    @staticmethod
    def combine(image_vec: np.ndarray, text_vec: np.ndarray, text_weight: float = 0.3) -> np.ndarray:
        """Blend an image query with a text refinement ("like this, but in red")."""
        v = (1 - text_weight) * image_vec + text_weight * text_vec
        return v / np.linalg.norm(v)

    # --- search --------------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 12, filters: Filters | None = None,
               exclude: list[str] | None = None) -> pd.DataFrame:
        mask = filters.mask(self.catalog) if filters else None
        if exclude:
            mask = np.ones(len(self.catalog), bool) if mask is None else mask.copy()
            mask[[self._row(i) for i in exclude]] = False
        scores = self.embeddings @ query.astype(np.float32)
        rows = top_k(scores, k, mask)
        result = self.catalog.iloc[rows].copy()
        result.insert(0, "score", scores[rows])
        return result.reset_index(drop=True)

    def _row(self, item_id: str) -> int:
        if not hasattr(self, "_row_of"):
            self._row_of = pd.Series(np.arange(len(self.catalog)), index=self.catalog.item_id)
        return int(self._row_of[item_id])
