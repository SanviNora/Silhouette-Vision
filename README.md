# Silhouette Vision

**Fashion visual search that tells you how sure it is.** Upload a photo and get similar products,
predicted attributes, whether the top result is *the exact product*, the name of an iconic
luxury model, and why each result matches. A second tool forecasts 12-week demand for a
brand-new product from look-alike past launches.

*Silhouette* is both a core fashion term and what the model actually looks at: the shape of a garment.

**Live demo:** [silhouette-vision.streamlit.app](https://silhouette-vision.streamlit.app/) (public version: Myntra catalog + demand forecast; may take a minute to wake up) · **Code, methods, evaluation:** this repo

## What it does

| Feature | How | Measured result |
|---|---|---|
| Search by photo, text, or both | Marqo-FashionSigLIP embeddings, exact NumPy search over 229k products | ~40 ms per query; 2.5× generic CLIP on LookBench exact Recall@1 |
| "Precise match" | Blend of Marqo and GR-Lite (DINOv3) scores | +4.7 exact R@1 (+7.6 on street photos), held out |
| Colour-aware ranking | CIELAB histogram re-ranking of the top 50 | Colour precision 42% → 47% |
| Garment picker | YOLOS-Fashionpedia detection, user picks the item | Street-photo R@1 30.4 → 37.3 (auto-pick: 15.5) |
| Attributes ("What we see") | Linear / MLP heads on frozen embeddings | Article type 90.5% accuracy, 81.4 macro-F1 |
| Exact-match likelihood | Calibrated on how much the best result stands out | "≥ 90%" is right 91% of the time (held out) |
| Named luxury models | Zero-shot over 161 iconic models, type-gated, calibrated | 92% top-1; 96% precision at P ≥ 0.8 |
| Why it matches | Shared attributes + occlusion heatmap | Heat on the garment 67% (random 21%); validated against 3 other methods |
| Luxury style map | UMAP + k-means on Farfetch, named by distinctive style words | 113 styles with price, markdown and resale stats |
| New-product demand | Gradient boosting + look-alike launches (Visuelle 2.0) | 34.6% weekly WAPE vs 43.7% seasonal average, on unseen seasons |

## Findings worth reading
- **"It loads" isn't "it's correct":** GR-Lite's reference code silently drops its rotary position
  embeddings; restoring them added 4 points of Recall@1. ([Phase 2](docs/phase2_results.md))
- **Similarity is not a probability:** results at cosine ≥ 0.95 were the exact product only 64%
  of the time; the gap to the runner-up is what separates exact matches. ([Phase 4](docs/phase4_results.md))
- **Fast saliency maps can mislead:** an exact attention-pooling decomposition put the heat on
  background "register" patches, less often on the garment than chance. ([Phase 4](docs/phase4_results.md))
- **The biggest demand signal is the planners' own bet** (number of stores), and it drifted in
  2019: the raw feature gave a +19% forecast bias, the within-season rank −1%. The photo adds a
  small but significant gain (−1.0 WAPE, 95% CI 0.5–1.5). ([Phase 5](docs/phase5_results.md))
- **Measure before automating:** auto-cropping the most confident garment halved street-photo
  recall; letting the user choose improved it. ([Phase 3](docs/phase3_results.md))

## Data and licensing

| Dataset | Use | License / handling |
|---|---|---|
| Fashion Product Images (Myntra, 44k) | Labels for attributes; public demo catalog | MIT |
| Farfetch listings (July 2019, 187k) | Luxury search, pricing and style analysis | Scraped, no license: **local analysis only; never redistributed** (the public demo shows aggregates only) |
| LookBench (2025) | Model selection and retrieval benchmark | Apache-2.0 |
| Visuelle 2.0 (5,355 launches, 110 stores) | Cold-start demand | CC BY-NC-SA 4.0, Skenderi et al., CVPRW 2022 |

Portfolio project; not affiliated with any brand or retailer. Demand findings are associations,
not causal effects. Details: [docs/data_guide.md](docs/data_guide.md).

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
# Raw data: see docs/data_guide.md. Then build each stage (or `make <stage>`):
python scripts/01_build_catalog.py
python scripts/02_embed.py --model marqo_fashion_siglip
python scripts/03_colour_features.py        # "Match colour"
python scripts/04_attributes.py             # attributes + catalog enrichment
python scripts/05_style_clusters.py         # style map
python scripts/06_match_confidence.py       # exact-match calibration
python scripts/07_named_models.py           # named luxury models
python scripts/08_visuelle_prepare.py && python scripts/09_demand.py   # demand
streamlit run app/ui.py
```

**Public demo:** `python scripts/10_build_public.py` builds a 0.7 GB bundle (Myntra + Visuelle
only); `SILHOUETTE_DATA_ROOT=deploy_bundle SILHOUETTE_PUBLIC=1 streamlit run app/ui.py` runs it;
`python scripts/11_publish.py` uploads it to the Hugging Face dataset
[Sanvii/silhouette-vision-data](https://huggingface.co/datasets/Sanvii/silhouette-vision-data),
which the hosted app (Streamlit Community Cloud, [guide](deploy/STREAMLIT_CLOUD.md)) downloads on
first start. `deploy/` also holds a Docker setup for a Hugging Face Space (needs PRO).

Tests: `pytest -q` (34 tests; `-m "not slow"` skips the two that download the real model).

## Repository map

```text
src/silhouette_vision/  library: catalog, encoders, search, colour, attributes, detect, match,
                        named_models, explain, demand
scripts/                numbered pipeline stages (01-11) and experiments (eval_*, exp_*)
app/ui.py               Streamlit app
configs/                data sources and licenses, taxonomy, pipeline, iconic luxury models
deploy/                 hosting: Streamlit Cloud guide, Docker Space files, data card
docs/                   pipeline, data guide, results per phase, interview notes
tests/                  unit tests (CI: lint + tests)
```

## Documentation
- [Pipeline design](docs/pipeline.md) · [Data guide](docs/data_guide.md) · [Data research](docs/data_research.md)
- Results: [1 search MVP](docs/phase1_results.md) · [2 measuring quality](docs/phase2_results.md) ·
  [3 product understanding](docs/phase3_results.md) · [4 explaining results](docs/phase4_results.md) ·
  [5 cold-start demand](docs/phase5_results.md)
- [Decision log and interview notes](docs/interview_notes.md)
