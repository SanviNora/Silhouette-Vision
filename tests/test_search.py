import numpy as np
import pandas as pd

from silhouette_vision.search import Filters, SearchEngine


def make_engine(n=50, dim=16, seed=0):
    """A SearchEngine over random normalized vectors, without loading real data or models."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    engine = SearchEngine.__new__(SearchEngine)
    engine.catalog = pd.DataFrame({
        "item_id": [f"it_{i}" for i in range(n)],
        "source": ["myntra", "farfetch"] * (n // 2),
        "category": ["bag", "shoes", "top", "dress", "other"] * (n // 5),
        "gender": ["Women"] * n,
        "is_preowned": [i % 7 == 0 for i in range(n)],
    })
    engine.embeddings = emb
    return engine


def test_item_is_its_own_nearest_neighbour():
    engine = make_engine()
    for item in ["it_0", "it_17", "it_42"]:
        top = engine.search(engine.item_vector(item), k=1)
        assert top.item_id.iloc[0] == item
        assert abs(top.score.iloc[0] - 1.0) < 1e-5


def test_exclude_removes_query_item():
    engine = make_engine()
    res = engine.search(engine.item_vector("it_3"), k=5, exclude=["it_3"])
    assert "it_3" not in set(res.item_id)
    assert len(res) == 5


def test_filters_apply_inside_search():
    engine = make_engine()
    f = Filters(sources=["farfetch"], categories=["bag", "shoes"])
    res = engine.search(engine.item_vector("it_0"), k=10, filters=f)
    assert len(res) == 10
    assert set(res.source) == {"farfetch"}
    assert set(res.category) <= {"bag", "shoes"}


def test_preowned_filter():
    engine = make_engine()
    res = engine.search(engine.item_vector("it_1"), k=50, filters=Filters(preowned=True))
    assert res.is_preowned.all()
    assert len(res) == engine.catalog.is_preowned.sum()


def test_scores_sorted_descending():
    engine = make_engine()
    res = engine.search(engine.item_vector("it_5"), k=20)
    assert (np.diff(res.score.values) <= 1e-6).all()


def test_combine_is_normalized_and_moves_toward_text():
    rng = np.random.default_rng(1)
    img, txt = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 16)))
    mixed = SearchEngine.combine(img, txt, text_weight=0.4)
    assert abs(np.linalg.norm(mixed) - 1) < 1e-6
    assert mixed @ txt > img @ txt
