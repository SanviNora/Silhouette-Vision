# Silhouette Vision — End-to-End Pipeline

Upload a product image → get visually similar items, predicted attributes, a plain-language
explanation of *why* they match, and a cold-start demand estimate borrowed from look-alike products.

```text
 ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
 │ 1. Ingest    │──▶│ 2. Clean &   │──▶│ 3. Unified    │──▶│ 4. Embed     │
 │ raw sources  │   │ validate     │   │ catalog +     │   │ image + text │
 └──────────────┘   └──────────────┘   │ splits        │   │ (CLIP)       │
                                       └───────────────┘   └──────┬───────┘
        ┌──────────────────────┬──────────────────────┬───────────┴──────────┐
        ▼                      ▼                      ▼                      ▼
 ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
 │ 5. Retrieval │   │ 6. Attribute     │   │ 7. Clustering    │   │ 9. Demand &      │
 │ index (FAISS)│   │ classifiers      │   │ (UMAP + HDBSCAN) │   │ cold-start       │
 └──────┬───────┘   └────────┬─────────┘   └──────────────────┘   └────────┬─────────┘
        └──────────┬─────────┘                                            │
                   ▼                                                      │
         ┌──────────────────┐     ┌──────────────────┐                    │
         │ 8. Explanations  │────▶│ 11. App          │◀───────────────────┘
         └──────────────────┘     │ (FastAPI+Streamlit)
                                  └──────────────────┘
            10. Attribute → demand & price drivers    ·  12. Evaluation, tests, model cards
```

Each stage is a script in `scripts/` that reads from the previous stage's output and writes a
versioned artifact. Everything is driven by `configs/pipeline.yml`, so the whole pipeline can be
re-run with one command (`make all` or `python -m silhouette_vision.pipeline`).

---

## Stage 0 — Scope, data contract, environment

| Decision | Choice | Why |
|---|---|---|
| Backbones | **GR-Lite** (image→image) + **Marqo-FashionSigLIP / ZooClaw-FashionSigLIP2** (text + image); FashionCLIP 2.0 and generic CLIP as baselines | Best open models on LookBench (see `docs/data_research.md`); comparing against generic CLIP is itself a result |
| Vector search | **Exact NumPy search** (one dot product over L2-normalized embeddings) | ~230k items → tens of ms per query; FAISS dropped (its OpenMP runtime crashes alongside PyTorch on macOS) |
| Storage | Parquet + DuckDB for tables, `.npy` for embeddings | Same stack as Couture Atelier Intelligence |
| Compute | Apple Silicon **MPS** for embedding; CPU for everything else | Embedding ~44k images per model is a one-off job of well under an hour on MPS |
| App | Streamlit (UI), hosted as a Hugging Face Docker Space | FastAPI was planned as optional and not needed; the library in `src/` is the reusable layer |

Deliverables: `configs/data_sources.yml` (license + provenance per source), `docs/data_contract.md`
(unified schema), and `docs/data_guide.md` (what is inside every dataset).

---

## Stage 1 — Ingestion (`scripts/01_ingest.py`)

Approved sources (details in `docs/data_guide.md`):

| Source | Tier | Role in the project |
|---|---|---|
| **Fashion Product Images** (full, 2400×1600; *not* the 60×80 "small" version) | Core | Clean labels → **attribute classifiers**, retrieval benchmark, public demo gallery |
| **Farfetch Listings** (July 2019 scrape) | Core | **Real luxury catalog**: brand, price, markdown, stock |
| **LookBench** (v20251201) | Core | Model selection; photo → product evaluation |
| **Visuelle 2.0** | Core | **New-product demand** from images; customer baskets |
| **H&M Personalized Fashion Recommendations** | Backup | Cold-start at scale; text descriptions |

Practical notes:
- Visuelle 2.0 and LookBench are downloaded. The Kaggle sources need `~/.kaggle/kaggle.json`.
- The full H&M set is ~30 GB, mostly images. Pull `articles.csv` + `transactions_train.csv`
  first, and **only the images you need** (72,962 articles have ≥ 20 sales).
- Record file checksums, download date, and row counts in `data/raw/_manifest.json`.

---

## Stage 2 — Cleaning & validation (`scripts/02_validate.py`)

- **Image integrity:** open every file with PIL; drop corrupt/truncated files and images < 64 px.
- **Duplicates:** perceptual hash (`imagehash.phash`); near-duplicates across splits are the #1
  source of inflated retrieval metrics — remove or group them.
- **Label hygiene:** normalize casing/synonyms (`Navy Blue` vs `Navy`), collapse rare classes
  (< 50 examples → `Other`), drop rows with missing images.
- **Schema checks** with `pandera` (same as the sibling project) and a pytest suite.
- **Output:** `data/interim/*.parquet` + a validation report (`artifacts/reports/validation.md`).

---

## Stage 3 — Unified catalog & splits (`scripts/03_build_catalog.py`)

**3a. Unified schema.** Map the catalog sources into one table:

```text
item_id | source | image_path | title | description | brand | category_l1 | category_l2 |
article_type | colour | fabric | pattern | gender | season | usage | price | price_is_normalized |
launch_date
```

Write a taxonomy crosswalk (`configs/taxonomy.yml`) mapping Myntra `articleType`, Farfetch
categories, Visuelle `category` and H&M `product_type_name` onto shared categories.

**3b. Luxury layer = real Farfetch data.** Brand, retail price, sale price and stock come from the
Farfetch scrape, so no synthetic luxury catalog is needed. If a synthetic field is ever added, it
gets an `is_synthetic` flag and a note in `docs/data_contract.md`.

**3c. Splits — leakage-aware.**
- Colour variants of one product (H&M `product_code`) never cross splits.
- For demand/cold-start, split **by season**: Visuelle's official split trains on earlier seasons
  and tests on 749 SS19/AW19 products. Use it as-is so results compare with published papers.
- Save split assignments to `data/processed/splits.parquet` so every stage uses the same ones.

---

## Stage 4 — Embeddings (`scripts/04_embed.py`)

1. **Image embeddings:** GR-Lite (1024-d), Marqo-FashionSigLIP (768-d), ZooClaw-FashionSigLIP2
   (768-d); baselines FashionCLIP 2.0 and generic CLIP. L2-normalized, batched on MPS.
   Composite transparent PNGs (Visuelle) onto white first.
2. **Text embeddings:** embed title + description with a SigLIP model's text encoder, so image and
   text live in one space. GR-Lite is image-only and is used for image → image search.
3. **Fused embedding:** `normalize(α·img + (1−α)·txt)`; tune α on the validation split (start α = 0.7).
4. Save as `artifacts/embeddings/{model}_{modality}.npy` + `ids.parquet` (row order = item_id order).
5. Cache aggressively; embedding is the only expensive step, and everything downstream re-uses it.

*Stretch:* fine-tune the projection head (or LoRA on the vision tower) with a contrastive loss on
same-`product_code` pairs, and report the gain over zero-shot.

---

## Stage 5 — Visual similarity & retrieval (`src/silhouette_vision/search.py`)

- Exact cosine search over image embeddings; text and image+text queries share the same space.
- **Query modes:** image → image, text → image ("black leather crossbody bag"), image + text
  refinement ("like this but in red": `normalize(q_img + β·(t_red − t_original_colour))`).
- **Filters:** applied before ranking (source, category, gender, new/pre-owned), so filtered queries still return k results.
- **Optional re-ranking:** combine cosine score with attribute agreement from Stage 6.

**Evaluation** (on held-out queries):

| Metric | Relevance definition |
|---|---|
| Fine Recall@1/5/10 on **LookBench** | Same `item_ID` — comparable to the public leaderboard |
| Precision@K on Myntra / Farfetch | Same `article_type` **and** `colour` — loose |
| Latency p50/p95 | Per query, CPU |

Baselines to beat: random, colour histogram + category, generic CLIP, DINOv3.

---

## Stage 6 — Attribute classification (`scripts/06_attributes.py`)

Targets: Myntra `articleType`, `baseColour`, `gender`, `season`, `usage`, plus the style-JSON
attributes `Pattern`, `Fabric`, `Sleeve Length`, `Fit`, `Neck`, `Material`; Visuelle `fabric`;
zero-shot for Farfetch (no labels) and for anything else unlabelled (e.g. silhouette).

Three approaches, compared in one table:
1. **Zero-shot CLIP**: prompts like `"a photo of a {colour} {article_type}"` — no training.
2. **Linear probe**: logistic regression on frozen embeddings (fast, strong, interpretable).
3. **Small MLP head** on embeddings (multi-task, one head per attribute).

Metrics: macro-F1 (classes are imbalanced), per-class F1, confusion matrices, calibration
(ECE, with temperature scaling). Output: predicted attributes + confidences for every item, so
**catalog enrichment** (filling missing tags) is a concrete deliverable.

---

## Stage 7 — Clustering (`scripts/07_cluster.py`)

- UMAP (to ~15-d for clustering, 2-d for plots) → HDBSCAN; K-means as baseline.
- **Name clusters automatically:** top attributes by lift within cluster + nearest CLIP text prompts
  from a style vocabulary ("minimalist", "logo-heavy", "tailored", "boho", …).
- Evaluate: silhouette, cluster purity vs. `article_type`, and a visual contact sheet per cluster.
- Output: `cluster_id` per item + an interactive 2-D map for the app ("style map").

---

## Stage 8 — Explainable similarity (`src/silhouette_vision/explain.py`)

For every (query, result) pair, return three grounded explanations:

1. **Attribute overlap** — "Both are *black*, *leather*, *crossbody bags*; differs in *hardware*."
   (from Stage 6 predictions).
2. **Concept-level similarity** — project both embeddings onto text directions (colour, silhouette,
   material, pattern) and report which concepts contribute most to the cosine score.
3. **Visual evidence** — occlusion or patch-similarity heatmap showing *which regions* of the two
   images match.

Sanity-check explanations: remove the top-cited attribute's region and confirm the similarity score
drops (a faithfulness test, not just a pretty picture).

---

## Stage 9 — Demand inheritance & cold-start (`scripts/09_cold_start.py`)

**Primary data: Visuelle 2.0.** Target = weekly units for the first 12 weeks after release,
per product (summed over stores) and per product-store. Setup = official season split
(train 2017–2019 seasons, test 749 SS19/AW19 products); "new" products have no sales history at
prediction time. Compare MAE/WAPE with published results (GTM-Transformer, etc.).

**Backup / scale check: H&M.** Target = units in the first *N* weeks after first sale (N = 4, 8);
time-based split on first-sale date.

| Model | Features |
|---|---|
| Category mean | category average |
| Metadata GBM (LightGBM) | category, colour, fabric, price, season, number of stores, Google Trends |
| **Visual kNN** | similarity-weighted mean demand of the *k* nearest *older* items |
| Metadata + embedding GBM | GBM features + PCA(embedding) + kNN demand feature |

Metrics: WAPE, Spearman rank correlation (merchandisers care about ranking), top-decile hit rate.
The key story: *how much does "looks like past winners" add over metadata alone?*

**Cold-start recommendation:** for a new item, recommend it to customers who bought its visual
neighbours (Visuelle `customer_data.csv`: 665k customers, 3.2M purchases). Evaluate with MAP@12
against a popularity baseline.

---

## Stage 10 — What drives demand and price (`scripts/10_drivers.py`)

- SHAP on the Stage 9 GBM → which attributes (colour, category, price band, cluster) move demand.
- Partial-dependence plots per attribute; compare clusters by average early demand.
- Farfetch: which brands and visual attributes go with higher retail price and with markdowns
  (`isOnSale`). Resale value is out of scope (no usable open resale data); list it as future work.
- State clearly that these are **associations**, not causal effects.

---

## Stage 11 — Application (`app/`)

```text
app/
  api.py        FastAPI: POST /search (image and/or text), GET /item/{id}, POST /attributes
  ui.py         Streamlit front-end calling the API
```

User flow:
1. Upload an image (or type a query, or both).
2. See top-12 similar items with scores, filterable by category/price tier.
3. Click a result → attribute overlap, concept bars, and side-by-side heatmap.
4. Predicted attributes for the uploaded image (catalog-enrichment view).
5. "If we launched this": estimated early demand range from visual neighbours.
6. Style map tab: 2-D UMAP with the query plotted among the clusters.

Load the index and model once at startup; target < 300 ms per query on CPU.
Deploy on Hugging Face Spaces or Streamlit Community Cloud with a ~10k-item **Myntra (MIT)** demo
index; the Farfetch gallery stays in the local/private demo because its images are scraped.

---

## Stage 12 — Evaluation, testing, documentation

- `tests/`: schema checks, split leakage checks (no product in two splits), embedding shape
  and norm checks, index round-trip (an item's nearest neighbour is itself), API smoke test.
- `artifacts/reports/results.md`: one table per stage, baseline vs. model.
- Model card + data card (limitations: Farfetch is a 2019 snapshot with no sales;
  Visuelle is Italian fast-fashion womenswear; coarse colour labels; gender label caveats).
- `docs/interview_case_study.md`: the 2-minute story and the numbers.

---

## Suggested build order

| Phase | Scope | Outcome |
|---|---|---|
| **1. MVP** | Myntra (full) + Farfetch → Marqo-FashionSigLIP embeddings → exact search → Streamlit upload | Working visual search demo |
| **2. Measure** | LookBench evaluation, retrieval metrics, baselines (FashionCLIP, CLIP, DINOv3) | Credible numbers |
| **3. Enrich** | Attribute classifiers + clustering + style map | Catalog-enrichment story |
| **4. Explain** | Attribute overlap, concept bars, heatmaps | Explainable results |
| **5. Demand** | Visuelle 2.0 cold-start models + SHAP (H&M as scale check) | Business-impact story |
| **6. Polish** | Farfetch price drivers, FastAPI, deployment, docs, tests | Portfolio-ready app |

Phase 1 alone is a demoable project; each later phase adds one clear talking point.
