"""Embed catalog images in resumable chunks.

Output for encoder `name`, in artifacts/embeddings/<name>/:
  chunks/chunk_00000.npy ...   float16 [chunk_size, dim]  (written as they finish; reruns skip them)
  embeddings.npy               float16 [n_items, dim], row i = catalog row i
  item_ids.parquet             item_id per row, to check alignment with the catalog
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from silhouette_vision.config import DATA_ROOT, path
from silhouette_vision.encoders import Encoder
from silhouette_vision.images import load_rgb


def _batches(paths: list[str], transform, batch_size: int, threads: int):
    """Tensors of decoded images, batch by batch; the next batch is decoded (by a thread pool)
    while the current one is on the GPU. Threads rather than DataLoader worker processes, as
    in the LookBench runs: nothing to pickle or re-import in each worker on macOS."""
    def decode(chunk):
        return torch.stack(list(pool.map(lambda p: transform(load_rgb(DATA_ROOT / p)), chunk)))

    with ThreadPoolExecutor(threads) as pool, ThreadPoolExecutor(1) as ahead:
        pending = None
        for s in range(0, len(paths), batch_size):
            future = ahead.submit(decode, paths[s:s + batch_size])
            if pending is not None:
                yield pending.result()
            pending = future
        if pending is not None:
            yield pending.result()


def embeddings_dir(name: str) -> Path:
    return path("embeddings") / name


def embed_catalog(
    encoder: Encoder,
    catalog: pd.DataFrame,
    chunk_size: int = 4096,
    batch_size: int = 64,
    threads: int = 6,
) -> np.ndarray:
    out_dir = embeddings_dir(encoder.name)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    assert chunk_size % batch_size == 0, "batches must not straddle chunk boundaries"
    paths = catalog.image_path.tolist()
    n_chunks = (len(paths) + chunk_size - 1) // chunk_size
    chunk_file = lambda c: chunk_dir / f"chunk_{c:05d}.npy"
    todo = [c for c in range(n_chunks) if not chunk_file(c).exists()]
    todo_paths = [p for c in todo for p in paths[c * chunk_size : (c + 1) * chunk_size]]

    start, done, parts = time.time(), 0, []
    chunks = iter(todo)
    current = next(chunks, None)
    for batch in _batches(todo_paths, encoder.transform, batch_size, threads):
        parts.append(encoder.embed_pixels(batch))
        expected = len(paths[current * chunk_size : (current + 1) * chunk_size])
        if sum(len(p) for p in parts) == expected:
            np.save(chunk_file(current), np.concatenate(parts).astype(np.float16))
            done += expected
            parts = []
            rate = done / (time.time() - start)
            print(f"chunk {current + 1}/{n_chunks}  {rate:.0f} img/s  "
                  f"~{(len(todo_paths) - done) / rate / 60:.0f} min left", flush=True)
            current = next(chunks, None)

    emb = np.concatenate([np.load(chunk_dir / f"chunk_{c:05d}.npy") for c in range(n_chunks)])
    assert len(emb) == len(catalog), (len(emb), len(catalog))
    np.save(out_dir / "embeddings.npy", emb)
    catalog[["item_id"]].to_parquet(out_dir / "item_ids.parquet", index=False)
    return emb


def load_embeddings(name: str, catalog: pd.DataFrame | None = None) -> np.ndarray:
    """Load float32 embeddings; if a catalog is given, verify row alignment."""
    out_dir = embeddings_dir(name)
    emb = np.load(out_dir / "embeddings.npy").astype(np.float32)
    if catalog is not None:
        ids = pd.read_parquet(out_dir / "item_ids.parquet").item_id
        if len(ids) != len(catalog) or not (ids.values == catalog.item_id.values).all():
            raise ValueError(f"{name} embeddings are out of sync with the catalog; re-run embedding")
    return emb
