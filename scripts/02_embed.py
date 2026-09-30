"""Stage 2: embed every catalog image with one encoder (resumable).

Usage: python scripts/02_embed.py --model marqo_fashion_siglip [--source myntra] [--limit 2000]

With --source, only that source is embedded and saved as "<model>__<source>" (catalog order kept).
"""

import argparse

from silhouette_vision.catalog import load_catalog
from silhouette_vision.embed import embed_catalog
from silhouette_vision.encoders import load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="marqo_fashion_siglip")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N items (smoke test)")
    ap.add_argument("--source", choices=["myntra", "farfetch"], default=None)
    args = ap.parse_args()

    catalog = load_catalog()
    if args.source:
        catalog = catalog[catalog.source == args.source].reset_index(drop=True)
    if args.limit:
        catalog = catalog.head(args.limit)
    encoder = load_encoder(args.model)
    if args.source:
        encoder.name = f"{encoder.name}__{args.source}"
    if args.limit:
        encoder.name = f"{encoder.name}_smoke"
    emb = embed_catalog(encoder, catalog, batch_size=args.batch_size, threads=args.workers)
    print(f"done: {emb.shape} -> artifacts/embeddings/{encoder.name}/embeddings.npy")


if __name__ == "__main__":
    main()
