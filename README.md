# Silhouette Vision

Fashion product intelligence and visual search: upload a product image and receive visually
similar items, predicted product attributes, and explanations of why they match.

*Silhouette* is both a core fashion term and what the model actually looks at: the shape of a garment.

## Questions answered

- Which products look visually similar, and which attributes drive that similarity?
- Can customers search using an image (or image + text)?
- Can new products inherit demand information from visually similar products?
- Which product attributes are associated with demand or price?

## Data principles

- Portfolio simulation; not affiliated with any luxury house.
- Data: Fashion Product Images, Farfetch Listings, LookBench and Visuelle 2.0 (core), H&M (backup).
  See [docs/data_guide.md](docs/data_guide.md).
- Source licenses and provenance are recorded in `configs/data_sources.yml` before use.
- Demand and price findings are reported as associations, not causal effects.
- Visuelle 2.0 is used under CC BY-NC-SA 4.0 (Skenderi et al., CVPRW 2022).

## Stack

Python, PyTorch + GR-Lite / Marqo-FashionSigLIP (OpenCLIP, Transformers), NumPy exact search, scikit-learn, LightGBM, UMAP + HDBSCAN,
DuckDB + Parquet, FastAPI, Streamlit, pytest.

## Repository map

```text
src/silhouette_vision/  package code
scripts/             numbered pipeline stages
app/                 FastAPI service and Streamlit UI
configs/             source, taxonomy, and pipeline configuration
data/                raw, interim, processed, synthetic (ignored)
artifacts/           embeddings, indexes, models, reports (ignored)
tests/               automated checks
docs/                project documentation
notebooks/           exploratory analysis
```

## Pipeline

See [docs/pipeline.md](docs/pipeline.md) for the full end-to-end design, evaluation plan, and build order.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/01_build_catalog.py          # needs the raw data, see docs/data_guide.md
python scripts/02_embed.py --model marqo_fashion_siglip
streamlit run app/ui.py
```

## Current milestone

**Phase 1 complete:** visual + text search over 229,522 products; ~40 ms per query on an M4.
Results and failure analysis: [docs/phase1_results.md](docs/phase1_results.md).
Next: Phase 2, quantitative evaluation on LookBench.
