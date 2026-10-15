import numpy as np
import pandas as pd

from silhouette_vision import embed


def test_update_keeps_old_vectors_by_id_and_embeds_only_new(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "embeddings_dir", lambda name: tmp_path / name)
    old_dir = tmp_path / "m"
    old_dir.mkdir()
    np.save(old_dir / "embeddings.npy", np.array([[1, 0], [0, 1], [1, 1]], np.float16))
    pd.DataFrame({"item_id": ["a", "b", "gone"]}).to_parquet(old_dir / "item_ids.parquet")
    (old_dir / "chunks").mkdir()  # stale cache in the old row order

    embedded = []

    def fake_embed(encoder, catalog, **kw):
        embedded.extend(catalog.item_id)
        return np.full((len(catalog), 2), 7, np.float16)

    monkeypatch.setattr(embed, "embed_catalog", fake_embed)
    encoder = type("E", (), {"name": "m"})()
    new_catalog = pd.DataFrame({"item_id": ["b", "new", "a"]})
    out = embed.update_embeddings(encoder, new_catalog)

    assert embedded == ["new"]
    assert out.tolist() == [[0, 1], [7, 7], [1, 0]]
    assert encoder.name == "m" and not (old_dir / "chunks").exists()
    assert pd.read_parquet(old_dir / "item_ids.parquet").item_id.tolist() == ["b", "new", "a"]
