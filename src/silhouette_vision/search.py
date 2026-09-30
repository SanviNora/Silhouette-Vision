"""Visual and text search over the catalog.

Exact search on L2-normalized embeddings: one matrix-vector product gives the cosine similarity to
every item. At 51k items this takes a few milliseconds in NumPy, so no ANN index (FAISS, HNSW) is
needed; FAISS was also dropped because its bundled OpenMP runtime clashes with PyTorch's.
Filters are applied before ranking, so a filtered query still returns k results.
The Marqo + GR-Lite blend ("Precise match", +4.7 R@1 on LookBench) was removed from the app: it
needs a second 1.2 GB model that does not fit the free host (docs/phase2_results.md).
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
    def __init__(self, model: str = "marqo_fashion_siglip", device: str | None = None):
        self.catalog = load_catalog()
        # Held at 16 bit (as stored) and scored in 32-bit chunks: ~80 MB less on the 2.7 GB free
        # host, identical rankings (16-bit storage measured lossless on LookBench, Phase 2).
        self.embeddings = load_embeddings(model, self.catalog).astype(np.float16)
        self.encoder = load_encoder(model, device=device)
        self.colour_hists = self._load_colour_hists()

    def _load_colour_hists(self) -> np.ndarray | None:
        """Colour histograms from scripts/03_colour_features.py; colour re-ranking is off without them."""
        folder = path("embeddings") / "colour"
        if not (folder / "hists.npy").exists():
            return None
        ids = pd.read_parquet(folder / "item_ids.parquet").item_id.values
        if len(ids) != len(self.catalog) or not (ids == self.catalog.item_id.values).all():
            return None
        return np.load(folder / "hists.npy").astype(np.float16)

    @property
    def has_colour(self) -> bool:
        return self.colour_hists is not None

    # --- query vectors -------------------------------------------------------------------
    def image_vector(self, image: Image.Image) -> np.ndarray:
        return self.encoder.embed_images([image])[0]

    def text_vector(self, text: str) -> np.ndarray:
        return self.encoder.embed_texts([text])[0]

    def item_vector(self, item_id: str) -> np.ndarray:
        return self.embeddings[self._row(item_id)].astype(np.float32)

    def similarities(self, query: np.ndarray, rows: np.ndarray | None = None, chunk: int = 8192) -> np.ndarray:
        """Cosine of `query` to every product (or to `rows`), computed in 32-bit chunks."""
        emb = self.embeddings if rows is None else self.embeddings[rows]
        q = query.astype(np.float32)
        return np.concatenate([emb[i:i + chunk].astype(np.float32) @ q for i in range(0, len(emb), chunk)])

    @staticmethod
    def image_colour(image: Image.Image) -> np.ndarray:
        return colour.colour_hist(image)

    def item_colour(self, item_id: str) -> np.ndarray | None:
        return None if self.colour_hists is None else self.colour_hists[self._row(item_id)].astype(np.float32)

    @staticmethod
    def combine(image_vec: np.ndarray, text_vec: np.ndarray, text_weight: float = 0.3) -> np.ndarray:
        """Blend an image query with a text refinement ("like this, but in red")."""
        v = (1 - text_weight) * image_vec + text_weight * text_vec
        return v / np.linalg.norm(v)

    # Hybrid text search: cosine + KEYWORD_WEIGHT * TF-IDF match on brand, title and type.
    # ZooClaw benchmark (zero-shot queries, whole catalog, weight chosen on the other half):
    # R@10 62.0 -> 83.2 for long LLM-written queries, 61.2 -> 92.2 for short ones (optimistic:
    # those are written from product titles). Embeddings alone miss brand and product names.
    KEYWORD_WEIGHT = 0.25

    def keyword_scores(self, text: str) -> np.ndarray:
        if not hasattr(self, "_keywords"):
            from sklearn.feature_extraction.text import TfidfVectorizer

            docs = (self.catalog.brand.fillna("") + " " + self.catalog.title.fillna("") + " "
                    + self.catalog.item_type.fillna("")).str.lower()
            vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
            self._keywords = (vec, vec.fit_transform(docs).T.tocsr())
        vec, docs_t = self._keywords
        return np.asarray((vec.transform([text.lower()]) @ docs_t).todense()).ravel().astype(np.float32)

    # --- search --------------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 12, filters: Filters | None = None,
               exclude: list[str] | None = None, query_colour: np.ndarray | None = None,
               colour_weight: float = colour.DEFAULT_WEIGHT, shortlist: int = 50,
               keywords: str | None = None) -> pd.DataFrame:
        """Top-k by cosine similarity (plus keyword match when `keywords` is given, for text
        search; the "similarity" column stays the cosine). With `query_colour`, the top
        `shortlist` items are re-scored as cosine + colour_weight * colour intersection
        (see colour.py)."""
        mask = filters.mask(self.catalog) if filters else None
        if exclude:
            mask = np.ones(len(self.catalog), bool) if mask is None else mask.copy()
            mask[[self._row(i) for i in exclude]] = False
        scores = self.similarities(query)
        rank = scores + self.KEYWORD_WEIGHT * self.keyword_scores(keywords) if keywords else scores
        use_colour = query_colour is not None and self.colour_hists is not None and colour_weight > 0
        rows = top_k(rank, max(k, shortlist) if use_colour else k, mask)
        final = rank[rows]
        if use_colour:
            final = final + colour_weight * colour.intersection(query_colour, self.colour_hists[rows].astype(np.float32))
            order = np.argsort(-final)[:k]
            rows, final = rows[order], final[order]
        result = self.catalog.iloc[rows].copy()
        result.insert(0, "score", final)
        result.insert(1, "similarity", scores[rows])
        return result.reset_index(drop=True)

    def best_match(self, query: np.ndarray, filters: Filters | None = None) -> tuple[pd.Series, np.ndarray]:
        """Top result by pure similarity (no colour re-ranking) and its top-5 scores, for the
        exact-match confidence, which was calibrated on pure similarity."""
        top = self.search(query, k=5, filters=filters)
        return top.iloc[0], top.similarity.values

    def _row(self, item_id: str) -> int:
        if not hasattr(self, "_row_of"):
            self._row_of = pd.Series(np.arange(len(self.catalog)), index=self.catalog.item_id)
        return int(self._row_of[item_id])
