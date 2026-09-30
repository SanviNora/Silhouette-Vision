import numpy as np
import pandas as pd

from silhouette_vision.demand import lookalikes, wape


def test_wape_weights_errors_by_total_sales():
    y = np.array([[10.0, 0.0], [0.0, 10.0]])
    assert wape(y, y) == 0
    assert wape(y, np.zeros_like(y)) == 100
    assert wape(y, np.array([[5.0, 0.0], [0.0, 15.0]])) == 50  # (5 + 5) / 20


def test_lookalikes_are_ordered_and_can_exclude_self():
    emb = np.eye(4)[[0, 0, 1, 2]] + np.array([[0, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    nn = lookalikes(emb, emb, 2)
    assert nn[0, 0] == 0 and nn[0, 1] == 1  # itself first, then its near-twin
    nn = lookalikes(emb, emb, 1, exclude_self=True)
    assert nn[0, 0] == 1 and nn[1, 0] == 0


def test_store_percentile_is_relative_to_the_season():
    # The brand widened distribution in 2019: raw counts drift, within-season ranks don't.
    products = pd.DataFrame({"season": ["AW18"] * 3 + ["AW19"] * 3, "n_stores": [5, 10, 20, 10, 20, 40]})
    pct = products.groupby("season").n_stores.rank(pct=True).values
    assert np.allclose(pct[:3], pct[3:])


def test_bundle_bootstrap_never_wipes_a_non_bundle_folder(tmp_path, monkeypatch):
    import pytest

    from silhouette_vision import bootstrap

    calls = []

    def fake_download(repo, repo_type, local_dir):  # like snapshot_download: creates the folder
        calls.append(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "VERSION").write_text(bootstrap.BUNDLE_VERSION)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    repo = tmp_path / "my_project"
    repo.mkdir()
    (repo / "precious.txt").write_text("keep me")
    with pytest.raises(RuntimeError):
        bootstrap.ensure_bundle(repo, "user/data")
    assert (repo / "precious.txt").exists() and not calls

    old = tmp_path / "silhouette_bundle"  # an outdated bundle (no VERSION): replaced
    old.mkdir()
    (old / "stale.parquet").write_text("old")
    bootstrap.ensure_bundle(old, "user/data")
    assert calls == [old] and not (old / "stale.parquet").exists()
