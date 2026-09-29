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

from silhouette_vision import colour
from silhouette_vision.catalog import load_catalog
from silhouette_vision.config import path
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
    def __init__(self, model: str = "marqo_fashion_siglip", device: str | None = None,
                 precise: bool = True):
        self.catalog = load_catalog()
        self.embeddings = load_embeddings(model, self.catalog)
        self.encoder = load_encoder(model, device=device)
        self.colour_hists = self._load_colour_hists()
        self.precise = self._load_precise() if precise else None
        self._precise_encoder = None

    # "Precise match": Marqo + GR-Lite blended 50/50 (best on held-out LookBench queries:
    # exact R@1 55.4 vs 50.7 for Marqo alone, +7.6 on street photos). GR-Lite embeddings exist
    # only for Myntra (the public gallery), so precise search is restricted to Myntra.
    PRECISE_MODEL, PRECISE_SOURCE, PRECISE_WEIGHT = "gr_lite", "myntra", 0.5

    def _load_precise(self) -> tuple[np.ndarray, np.ndarray] | None:
        name = f"{self.PRECISE_MODEL}__{self.PRECISE_SOURCE}"
        rows = np.flatnonzero(self.catalog.source.values == self.PRECISE_SOURCE)
        try:
            emb = load_embeddings(name, self.catalog.iloc[rows].reset_index(drop=True))
        except (FileNotFoundError, ValueError):
            return None
        return rows, emb

    @property
    def has_precise(self) -> bool:
        return self.precise is not None

    def precise_vector(self, image: Image.Image) -> np.ndarray:
        if self._precise_encoder is None:
            self._precise_encoder = load_encoder(self.PRECISE_MODEL)
        return self._precise_encoder.embed_images([image])[0]

    def _load_colour_hists(self) -> np.ndarray | None:
        """Colour histograms from scripts/03_colour_features.py; colour re-ranking is off without them."""
        folder = path("embeddings") / "colour"
        if not (folder / "hists.npy").exists():
            return None
        ids = pd.read_parquet(folder / "item_ids.parquet").item_id.values
        if len(ids) != len(self.catalog) or not (ids == self.catalog.item_id.values).all():
            return None
        return np.load(folder / "hists.npy").astype(np.float32)

    @property
    def has_colour(self) -> bool:
        return self.colour_hists is not None

    # --- query vectors -------------------------------------------------------------------
    def image_vector(self, image: Image.Image) -> np.ndarray:
        return self.encoder.embed_images([image])[0]

    def text_vector(self, text: str) -> np.ndarray:
        return self.encoder.embed_texts([text])[0]

    def item_vector(self, item_id: str) -> np.ndarray:
        return self.embeddings[self._row(item_id)]

    @staticmethod
    def image_colour(image: Image.Image) -> np.ndarray:
        return colour.colour_hist(image)

    def item_colour(self, item_id: str) -> np.ndarray | None:
        return None if self.colour_hists is None else self.colour_hists[self._row(item_id)]

    @staticmethod
    def combine(image_vec: np.ndarray, text_vec: np.ndarray, text_weight: float = 0.3) -> np.ndarray:
        """Blend an image query with a text refinement ("like this, but in red")."""
        v = (1 - text_weight) * image_vec + text_weight * text_vec
        return v / np.linalg.norm(v)

    # --- search --------------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 12, filters: Filters | None = None,
               exclude: list[str] | None = None, query_colour: np.ndarray | None = None,
               colour_weight: float = colour.DEFAULT_WEIGHT, shortlist: int = 50,
               precise_query: np.ndarray | None = None) -> pd.DataFrame:
        """Top-k by cosine similarity. With `query_colour`, the top `shortlist` items are
        re-scored as cosine + colour_weight * colour intersection (see colour.py). With
        `precise_query` (a GR-Lite vector), search is limited to Myntra and scores are the
        Marqo/GR-Lite blend."""
        mask = filters.mask(self.catalog) if filters else None
        if exclude:
            mask = np.ones(len(self.catalog), bool) if mask is None else mask.copy()
            mask[[self._row(i) for i in exclude]] = False
        scores = self.embeddings @ query.astype(np.float32)
        if precise_query is not None and self.precise is not None:
            rows, emb = self.precise
            only = np.zeros(len(self.catalog), bool)
            only[rows] = True
            mask = only if mask is None else mask & only
            w = self.PRECISE_WEIGHT
            scores = scores.copy()
            scores[rows] = (1 - w) * scores[rows] + w * (emb @ precise_query.astype(np.float32))
        use_colour = query_colour is not None and self.colour_hists is not None and colour_weight > 0
        rows = top_k(scores, max(k, shortlist) if use_colour else k, mask)
        final = scores[rows]
        if use_colour:
            final = final + colour_weight * colour.intersection(query_colour, self.colour_hists[rows])
            order = np.argsort(-final)[:k]
            rows, final = rows[order], final[order]
        result = self.catalog.iloc[rows].copy()
        result.insert(0, "score", final)
        result.insert(1, "similarity", scores[rows])
        return result.reset_index(drop=True)

    def best_match(self, query: np.ndarray, filters: Filters | None = None,
                   precise_query: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
        """Top result by pure similarity (no colour re-ranking) and its top-5 scores, for the
        exact-match confidence, which was calibrated on pure similarity."""
        top = self.search(query, k=5, filters=filters, precise_query=precise_query)
        return top.iloc[0], top.similarity.values

    def _row(self, item_id: str) -> int:
        if not hasattr(self, "_row_of"):
            self._row_of = pd.Series(np.arange(len(self.catalog)), index=self.catalog.item_id)
        return int(self._row_of[item_id])
