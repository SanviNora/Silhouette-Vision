# Silhouette Vision

**Fashion visual search that tells you how sure it is.** Upload a photo or type a description to
find similar products from 51,047 recent listings (2019–2026). The app predicts attributes, says
whether the top result is *the exact product*, names iconic luxury models, and explains why each
result matches. A second tool forecasts 12-week demand for a brand-new product from look-alike
past launches.

*Silhouette* is both a core fashion term and what the model actually looks at: the shape of a garment.

**Live demo:** [silhouette-vision.streamlit.app](https://silhouette-vision.streamlit.app/) (may take a minute to wake up)

## What it does

| Feature | How | Measured result |
|---|---|---|
| Search by photo | Marqo-FashionSigLIP embeddings, exact search over 51k products | 2.5× generic CLIP on LookBench exact Recall@1; ties the newest open fashion model (ZooClaw-SigLIP2) at half the size |
| Search by words | **Hybrid**: embedding + TF-IDF keyword match on brand, title, type | ZooClaw benchmark (zero-shot, whole catalog): R@10 65.1 → 83.6 |
| Garment picker | YOLOS-Fashionpedia detection; the user picks the item | Street-photo R@1 30.4 → 37.3 (auto-pick: 15.5) |
| Colour-aware ranking | CIELAB histogram re-ranking of the top 50 | Colour agreement 50.8 → 54.8% |
| Attributes ("What we see") | Heads on frozen embeddings, trained on Myntra + the catalog's own labels | Held-out recent photos: type 80.7%, colour 87.5%, pattern 79.0%, footwear 99.4% |
| Exact-match likelihood | Calibrated on how much the best result stands out | In-catalog photos: "≥ 80%" right 88% of the time; absent products claimed exact 0.1% |
| Named luxury models | Zero-shot over 161 iconic models, type-gated, calibrated | 92% top-1; 96% precision at P ≥ 0.8 |
| Why it matches | Shared attributes (✓ ≈ ✗) + occlusion heatmap | Heat on the garment 67% (random 21%); validated against 3 other methods |
| Style map | UMAP + k-means per category, centred per source, named by distinctive style words | 48 styles; which styles are donated second-hand vs sold new |
| New-product demand | Gradient boosting + look-alike launches (Visuelle 2.0) | 34.6% weekly WAPE vs 43.7% seasonal average, on unseen seasons |

## Findings worth reading
- **Check provenance, don't assume it:** a benchmark's 58k "2025" distractor images turned out to be
  copies of a 2017 dataset (tested: 27.6% near-duplicate hit rate, the 29% a random slice would
  give), so they were kept out of the catalog. ([Phase 1](docs/phase1_results.md))
- **Re-measure published claims:** a newer model's authors reported our model at 43.9 R@10; on
  identical data it scored 67.3, and the two tied on photo search. ([Phase 2](docs/phase2_results.md))
- **Cheap beats big:** hybrid keyword + embedding search added 18 points of text-search R@10,
  three times what the bigger model offered. ([Phase 2](docs/phase2_results.md))
- **Clusters can measure the camera, not the clothes:** raw style clusters split flat lays from studio
  shots (97% source purity) until each source's mean was removed. ([Phase 3](docs/phase3_results.md))
- **"It loads" isn't "it's correct":** GR-Lite's reference code silently drops its rotary position
  embeddings; restoring them added 4 points of Recall@1. ([Phase 2](docs/phase2_results.md))
- **Similarity is not a probability:** results at cosine ≥ 0.95 were the exact product only 64%
  of the time; the gap to the runner-up separates exact matches. ([Phase 4](docs/phase4_results.md))
- **The biggest demand signal is the planners' own bet** (number of stores), and it drifted in 2019:
  the raw feature gave +19% bias, its within-season rank −1%. ([Phase 5](docs/phase5_results.md))

## Data and licensing

| Dataset | Use | License |
|---|---|---|
| ZooClaw-Fashion (2026, 12k products, 2,086 brands) | Search catalog; text-search benchmark | CC BY-NC 4.0 |
| Second-Hand Fashion (2022–24, 32k donated garments) | Search catalog; attribute labels | CC BY 4.0 |
| LookBench (2025) | Search catalog (studio gallery); photo-search benchmark | Apache-2.0 |
| Amazon Berkeley Objects (c. 2019–21, 6k shoes) | Search catalog (footwear) | CC BY 4.0 |
| Visuelle 2.0 (5,355 launches, 110 stores) | Cold-start demand | CC BY-NC-SA 4.0 |
| Fashion Product Images (Myntra) | Attribute training labels (not searched: median product year 2012) | MIT |
| Farfetch listings (2019) | Earlier luxury analyses, kept local only | Scraped, no redistribution |

Portfolio project; not affiliated with any brand or retailer; non-commercial. Demand and style
findings are associations, not causal effects. Details: [docs/data_guide.md](docs/data_guide.md).

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
python scripts/00_fetch_sources.py            # ZooClaw, Second-Hand, ABO footwear (LookBench, Visuelle: docs/data_guide.md)
python scripts/01_build_catalog.py            # search catalog + thumbnails (--legacy: Myntra/Farfetch)
python scripts/02_embed.py --model marqo_fashion_siglip
python scripts/03_colour_features.py          # "Match colour"
python scripts/04_attributes.py               # attribute heads (trained on the legacy Myntra labels + recent labels)
python scripts/05_style_clusters.py           # style map
python scripts/06_match_confidence.py         # exact-match calibration + catalog checks
python scripts/07_named_models.py             # named luxury models
python scripts/08_visuelle_prepare.py && python scripts/09_demand.py   # demand
streamlit run app/ui.py
```

**Hosted app:** `scripts/10_build_bundle.py` builds the 1.1 GB data bundle and
`scripts/11_publish.py` uploads it to [Sanvii/silhouette-vision-data](https://huggingface.co/datasets/Sanvii/silhouette-vision-data);
Streamlit Community Cloud runs `app/ui.py` and downloads it on first start ([guide](deploy/STREAMLIT_CLOUD.md)).

Tests: `pytest -q` (35 tests; `-m "not slow"` skips the two that download the real model).

## Repository map

```text
src/silhouette_vision/  library: catalog, encoders, embed, search, colour, attributes, recent_labels,
                        detect, match, named_models, explain, demand, bootstrap
scripts/                numbered pipeline stages (00-11) and experiments (eval_*, exp_*)
app/ui.py               Streamlit app (theme in .streamlit/config.toml)
configs/                data sources and licenses, taxonomy, pipeline, iconic luxury models
deploy/                 Streamlit Cloud guide, data bundle card
docs/                   pipeline, data guide, results per phase, interview notes
tests/                  unit tests
```

## Documentation
- [Pipeline design](docs/pipeline.md) · [Data guide](docs/data_guide.md) · [Data research](docs/data_research.md)
- Results: [1 search](docs/phase1_results.md) · [2 measuring quality](docs/phase2_results.md) ·
  [3 product understanding](docs/phase3_results.md) · [4 explaining results](docs/phase4_results.md) ·
  [5 cold-start demand](docs/phase5_results.md)
- [Decision log and interview notes](docs/interview_notes.md)
