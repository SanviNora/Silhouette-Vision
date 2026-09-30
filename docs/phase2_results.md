# Phase 2 Results — Measuring Quality

## Update 2026-09-29: measured again on the recent catalog (51,047 products)
**Which model? Marqo stays.** ZooClaw-FashionSigLIP2 (2026, Apache-2.0; its authors' table puts
it far ahead) was measured on identical data:

| Test | Marqo-FashionSigLIP | ZooClaw-FashionSigLIP2 |
|---|---|---|
| Photo → exact product, LookBench studio + street (1,992 queries, 19k gallery) | R@1 49.4 | 50.2 |
| Text → product, ZooClaw zero-shot short queries (12k products) | R@10 67.3 | 73.2 |
| Text → product, long queries | R@10 76.9 | 78.3 |
| Model size / speed | 0.8 GB, ~45 img/s | 1.5 GB, ~10 img/s (384 px) |

Photo search (the app's main job) is a tie; blending both adds only 1–2 R@1 and needs both in
memory. The published table understates Marqo: it reports R@10 43.9 on short queries where we
measure 67.3, a reminder to re-measure rather than trust a model author's table.

**Hybrid text search (shipped).** Cosine + 0.25 × TF-IDF match on brand, title and type, weight
chosen on the other half of the queries (whole 51k catalog, zero-shot queries):

| Queries | Embedding only R@1 / R@10 | Hybrid R@1 / R@10 |
|---|---|---|
| Long, LLM-written (fairer) | 34.2 / 65.1 | **61.7 / 83.6** |
| Short (optimistic: written from titles) | 35.8 / 63.0 | **80.0 / 93.8** |

Embeddings miss brand and product names ("levi's jeans" returned Levi's t-shirts); keywords fix
that at no GPU cost. Side effect: short Second-Hand titles ("Striped t-shirt") match generic
queries almost perfectly, so they dominate those results.

**Rejected:** averaging each photo with its mirror image (R@1 −0.2 to 0, R@10 +0.4–0.9, twice
the embedding cost).

**Colour re-ranking re-checked** on Second-Hand colour labels: γ = 0.1 still the best trade-off
(colour P@10 50.8 → 54.8, type −1.2 points; held-out half).

**Catalog quality** (precision@10 of the 10 nearest products vs chance):

| Source | Label | P@10 | Chance | Lift |
|---|---|---|---|---|
| Second-Hand | item type (33) | 60.7% | 7.5% | 8.1× |
| Second-Hand | pattern | 68.4% | 13.3% | 5.1× |
| Second-Hand | colour | 51.2% | 13.3% | 3.9× |
| ZooClaw | brand (2,085) | 8.5% | 0.1% | 64× |
| ZooClaw | type | 93.6% | 35.1% | 2.7× |
| Amazon footwear | brand | 75.7% | 12.1% | 6.3× |
| LookBench | type (41) | 67.1% | 4.6% | 14.7× |


**Dates:** 2026-09-27/28 · **Hardware:** Apple M4, 16 GB · All numbers reproducible with the
scripts named in each section.

## TL;DR
- Our evaluation **reproduces the LookBench paper** within ~1 point (Marqo fine R@1 63.8% vs 62.8%
  published; GR-Lite 65.1% vs 65.7%).
- **Fashion-specific training matters:** Marqo beats generic CLIP 2.5× overall and 3.5× on street photos.
- **Blending Marqo + GR-Lite 50/50 is the best retriever:** exact R@1 55.4% vs 50.7% (Marqo alone)
  on held-out queries, **+7.6 points on street photos**.
- **Colour re-ranking** lifts colour precision 42% → 47% for a 0.9-point cost in type precision;
  shipped as an app toggle.
- Three generic re-ranking tricks were tested and **rejected** (no gain on held-out queries).

## 1. LookBench (scripts/eval_lookbench.py)
2,345 queries in 4 subsets; each gallery has its own items plus 58,275 shared distractors.
Three hit definitions:
- **exact**: the retrieved item is the same product (strictest, and what a shopper wants)
- **fine**: same category + same main attribute (the paper's "fine Recall@1")
- **coarse**: same category

| Model | Exact R@1 / R@5 / R@10 | Fine R@1 | Coarse R@1 | Speed on M4 |
|---|---|---|---|---|
| Generic CLIP ViT-B/16 | 20.5 / 33.3 / 37.5 | 25.1 | 38.8 | ~37 img/s |
| **Marqo-FashionSigLIP** | 51.6 / 74.3 / 79.7 | 63.8 | 83.7 | **~36–55 img/s** |
| **GR-Lite** (with RoPE) | 52.8 / 77.5 / 82.9 | 65.1 | 84.6 | ~4.5–5 img/s |

Per subset, exact R@1: Marqo vs GR-Lite
| Subset | Marqo | GR-Lite | Note |
|---|---|---|---|
| Studio product photos | 55.4 | 57.6 | |
| **Street photos** (the phone-photo case) | 41.8 | **45.1** | GR-Lite's clearest win |
| AI-generated studio | **69.4** | 58.5 | GR-Lite much worse (−10.9) |
| AI-generated street | **65.6** | 62.5 | |

### Reproducing the paper
Our first Marqo number (51.6%) was 11 points below the paper's 62.8%. Checks, in order:
1. Gallery sizes match the paper exactly, so the data is right.
2. Preprocessing: Marqo's official transform is already a square resize, as in the paper's code.
3. **Metric definition:** the paper's "fine Recall@1" counts a hit when the category and main
   attribute match, not the exact product. Computed that way from the same cached embeddings:
   63.8% (paper 62.8%). All four subsets are within ~2.6 points.

### GR-Lite needs RoPE
GR-Lite's published reference code omits DINOv3's rotary position embeddings. Because RoPE has no
learned weights, the model still loads without errors. On 200 studio queries:
with RoPE R@1/5/10 = 60.5/85.0/91.0, without = 56.5/76.0/79.5. We run it with RoPE.

## 2. Combining models (held-out evaluation)
Scores blended as `w · GR-Lite + (1 − w) · Marqo`, with w chosen on even-indexed queries (dev)
and reported on odd-indexed queries (test).

| Test half | Exact R@1 / R@5 / R@10 |
|---|---|
| Marqo alone | 50.7 / 75.1 / 80.5 |
| GR-Lite alone | 53.6 / 78.1 / 82.8 |
| **Blend, w = 0.5** | **55.4 / 80.8 / 85.6** |
| Blend on street photos | 47.6 / 70.8 / 75.1 (Marqo alone 40.0 / 63.5 / 68.2) |

The two models make different mistakes (GR-Lite is better on real photos, Marqo on AI-generated
images), so the blend beats both.

## 3. Catalog-level quality (scripts/eval_catalog.py)
For 5,000 sampled products per source, precision@10 of the nearest neighbours against labels,
compared with random chance.

| Label | Precision@10 | Chance | Lift |
|---|---|---|---|
| Myntra article type (141) | 88.2% | 5.7% | 15.5× |
| Myntra sleeve length | 89.8% | 46.4% | 1.9× |
| Myntra pattern | 82.5% | 28.4% | 2.9× |
| Myntra neck | 81.1% | 44.4% | 1.8× |
| **Myntra colour** | **42.1%** | 10.1% | 4.2× |
| Farfetch category | 89.9% | 12.5% | 7.2× |
| **Farfetch brand** (2,297) | **36.7%** | 0.3% | **105×** |

Colour is the weakest label, which confirms Phase 1's qualitative finding. Brand is recognized
very strongly from images alone.

## 4. Improvement attempts

### Rejected: generic re-ranking (scripts/exp_retrieval_tricks.py)
All tricks use model outputs only. LookBench labels its 58k distractors "noise data", so using
the given labels would make the benchmark trivially easy. Held-out change in exact R@1 vs baseline:

| Trick | real studio | AI studio | real street | AI street |
|---|---|---|---|---|
| Zero-shot category agreement | −0.2 | −4.2 | −0.8 | −5.0 |
| Database-side augmentation | −1.2 | −4.2 | +0.2 | −7.5 |
| Alpha query expansion | +0.2 | −3.1 | −0.6 | −2.5 |

These tricks help when each query has many relevant items to borrow from. Here each query has
1–4 true matches among ~60k near-duplicates, so averaging with neighbours mostly adds noise.

### Adopted: colour-aware re-ranking (scripts/exp_colour_rerank.py, src/silhouette_vision/colour.py)
Colour descriptor: CIELAB histogram of non-background pixels. The top-50 results are re-scored as
`cosine + γ · histogram intersection`, with γ chosen on the dev half.

| γ | Colour P@10 | Type P@10 |
|---|---|---|
| 0 | 42.3% | 87.8% |
| **0.1 (chosen)** | **47.2%** | 86.9% |
| 0.5 | 49.6% | 84.2% |

Shipped as the app's "Match colour" toggle (on by default). On the test photos it helped the
yellow lace top (cream/yellow tops rose to the top) and partly helped the burgundy Miu Miu
slingbacks (reds replaced by dark patent heels). Limitation: the background removal only
handles plain studio backgrounds, so a shoe on a patterned sofa carries the sofa's colours into
its descriptor.

## 5. Engineering fixes during Phase 2
| Issue | Fix |
|---|---|
| Evaluation decoded all 69k benchmark images at once and pushed the Mac into 24 GB of swap | Stream compressed bytes; decode each batch in threads just before embedding |
| DataLoader workers each received a copy of 2 GB of image bytes (macOS spawn) | Threads instead of worker processes (also 4× faster) |
| The overnight run lost progress when the battery died | Benchmark embeddings are now checkpointed every 4,096 images |
| An encoder refactor would have crashed the app on start-up | Fixed; two tests now load the real model |

## 6. Decisions and what's next
- **Default model stays Marqo:** fast (~40 ms/query), handles text search, and ~10× faster
  embedding keeps the whole catalog practical.
- **The Marqo + GR-Lite blend is the best retriever.** Offering it in the app requires GR-Lite
  embeddings for the catalog: ~13.5 h locally for all 229k items (at 4.7 img/s), ~2.5 h for
  Myntra only (the public demo gallery), or roughly 1 h on a Colab GPU.
- **Next weakness to address: busy photos.** Detecting and cropping the garment would fix
  multi-item photos, street backgrounds, and the colour descriptor's background problem at once.
