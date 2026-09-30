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


ABO_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
ABO_FOOTWEAR = {"SHOES", "BOOT", "SANDAL"}


def fetch_abo():
    """Footwear from Amazon Berkeley Objects (CC BY 4.0, Amazon listings c. 2019-2021): the
    recent sources carry almost no shoes. One photo per product (listings repeat a shoe across
    marketplaces and sizes), originals shrunk to 1024 px."""
    import gzip
    import io
    import json
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from PIL import Image

    out = RAW / "abo"
    (out / "images").mkdir(parents=True, exist_ok=True)
    if not (out / "listings").exists():
        data = requests.get(f"{ABO_URL}/archives/abo-listings.tar", timeout=300).content
        tarfile.open(fileobj=io.BytesIO(data)).extractall(out, filter="data")
    if not (out / "images.csv.gz").exists():
        (out / "images.csv.gz").write_bytes(requests.get(f"{ABO_URL}/images/metadata/images.csv.gz", timeout=300).content)
    rows = []
    for f in sorted((out / "listings/metadata").glob("*.json.gz")):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            kind = d["product_type"][0]["value"] if d.get("product_type") else ""
            if kind not in ABO_FOOTWEAR or not d.get("main_image_id"):
                continue
            names = d.get("item_name", [])
            english = [n["value"] for n in names if n.get("language_tag", "").startswith("en")]
            rows.append({"item_id": d["item_id"], "type": kind, "main_image_id": d["main_image_id"],
                         "name": english[0] if english else None,
                         "brand": (d.get("brand") or [{}])[0].get("value")})
    listings = (pd.DataFrame(rows).sort_values("name", na_position="last")
                .drop_duplicates("main_image_id").dropna(subset=["name"]))
    paths = pd.read_csv(out / "images.csv.gz").set_index("image_id").path

    def get(image_id):
        dst = out / "images" / f"{image_id}.jpg"
        if dst.exists():
            return
        import time

        for attempt in range(4):  # the S3 endpoint occasionally resets connections
            try:
                r = requests.get(f"{ABO_URL}/images/original/{paths[image_id]}", timeout=60)
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                break
            except (requests.RequestException, OSError):
                time.sleep(2 ** attempt)
        else:
            return
        img.thumbnail((1024, 1024))
        img.save(dst, quality=92)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, listings.main_image_id))
    listings = listings[listings.main_image_id.map(lambda i: (out / "images" / f"{i}.jpg").exists())]
    listings.to_parquet(out / "footwear.parquet", index=False)
    print(f"abo: {len(listings):,} footwear products, {listings.brand.nunique()} brands")


if __name__ == "__main__":
    which = sys.argv[1:] or ["zooclaw", "secondhand", "abo"]
    for name in which:
        {"zooclaw": fetch_zooclaw, "secondhand": fetch_secondhand, "abo": fetch_abo}[name]()
