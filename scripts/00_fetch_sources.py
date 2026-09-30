"""Stage 0: download the recent catalog sources from Hugging Face into data/raw/.

  zooclaw     srpone/zooclaw-fashion-eval (CC BY-NC 4.0): 12,000 products (2026), 2,086 brands.
              -> data/raw/zooclaw/{corpus,queries,ground_truth}.parquet, images/<corpus_id>.<ext>
  secondhand  chibifire/zenodo-second-hand-fashion-v3 (CC BY 4.0; Nauman et al., RISE, Wargön
              Innovation, Myrorna; Zenodo 10.5281/zenodo.13788681): ~32k donated garments
              photographed 2022-2023. Only the front photo and the labels are read (column
              projection over HTTP), about a third of the 26.6 GB dataset.
              -> data/raw/secondhand/labels.parquet, images/<garment_id>.jpg

LookBench (2025, Apache-2.0) is already in data/raw/lookbench. Resumable: existing images are skipped.
Usage: python scripts/00_fetch_sources.py [zooclaw] [secondhand]
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from silhouette_vision.config import ROOT

RAW = ROOT / "data/raw"


def _bytes(value):
    return value["bytes"] if isinstance(value, dict) else value


def fetch_zooclaw():
    repo, out = "srpone/zooclaw-fashion-eval", RAW / "zooclaw"
    (out / "images").mkdir(parents=True, exist_ok=True)
    for name in ["corpus", "queries", "ground_truth"]:
        src = hf_hub_download(repo, f"data/{name}-00000-of-00001.parquet", repo_type="dataset")
        table = pq.read_table(src)
        if name == "corpus":
            meta = table.drop(["image"]).to_pandas()
            for cid, img in zip(meta.corpus_id, table.column("image").to_pylist(), strict=True):
                ext = Path(img.get("path") or "x.jpg").suffix.lower() or ".jpg"
                dst = out / "images" / f"{cid}{ext}"
                if not dst.exists():
                    dst.write_bytes(img["bytes"])
            meta.to_parquet(out / "corpus.parquet", index=False)
            print(f"zooclaw: {len(meta):,} products, {meta.brand.nunique():,} brands")
        else:
            table.to_pandas().to_parquet(out / f"{name}.parquet", index=False)


def fetch_secondhand():
    repo, out = "chibifire/zenodo-second-hand-fashion-v3", RAW / "secondhand"
    (out / "images").mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings if s.rfilename.endswith(".parquet"))
    labels = []
    for n, f in enumerate(files, 1):
        pf = pq.ParquetFile(fs.open(f"datasets/{repo}/{f}"))
        keep = [c for c in pf.schema_arrow.names if not c.startswith("image") and c != "damage_image"]
        for rg in range(pf.num_row_groups):
            meta = pf.read_row_group(rg, columns=keep).to_pandas().assign(split=f.split("/")[1].split("-")[0])
            labels.append(meta)
            if all((out / "images" / f"{g}.jpg").exists() for g in meta.garment_id):
                continue  # resuming: this part's photos are already on disk
            t = pf.read_row_group(rg, columns=["image_front"])
            for gid, img in zip(meta.garment_id, t.column("image_front").to_pylist(), strict=True):
                dst = out / "images" / f"{gid}.jpg"
                if img is not None and not dst.exists():
                    dst.write_bytes(_bytes(img))
        print(f"  secondhand {n}/{len(files)} files", flush=True)
    labels = pd.concat(labels, ignore_index=True)
    labels.to_parquet(out / "labels.parquet", index=False)
    print(f"secondhand: {len(labels):,} garments, {labels.brand.nunique():,} brands")


if __name__ == "__main__":
    which = sys.argv[1:] or ["zooclaw", "secondhand"]
    for name in which:
        {"zooclaw": fetch_zooclaw, "secondhand": fetch_secondhand}[name]()
