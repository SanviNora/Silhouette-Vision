# Phase 3 Results — Product Understanding

**Date:** 2026-09-28 · Features: Marqo-FashionSigLIP embeddings (frozen) · CPU only

## TL;DR
- **Attribute prediction:** a small head trained on frozen embeddings beats zero-shot on every
  attribute. Article type 90.5% accuracy (81.4 macro-F1 over 59 types), sleeves 95.8%, gender 94%.
- **Catalog enrichment:** attributes predicted for all 187k Farfetch luxury products (no labels
  of their own). Category agreement with Farfetch's titles: 89.6%. **Domain shift found and
  handled:** coats predicted as handbags (Myntra has no coat class), luxury materials called
  "synthetic", 95% of items "casual".
- **Style clusters:** 113 visual styles across 8 luxury categories, with price, markdown and
  pre-owned stats. Iconic bags (Chanel quilted, LV/Dior monogram) are never marked down and are
  mostly pre-owned.
- **Garment detection:** cropping to the item the user picks lifts street-photo retrieval from
  30.4% to 37.3% R@1, recovering 60% of the gap to a perfect crop. Auto-picking the item halves it.

## 3.1 Attribute prediction (scripts/04_attributes.py)
Labels: Myntra styles metadata + structured `articleAttributes`. Classes with <50 examples dropped.
**Group-aware split** 70/15/15: colour and gender variants of one product ("Puma Men Miami Black
Slipper" / "Puma Unisex Miami Purple Slipper") share a group, so near-twins never straddle
train and test. Method chosen on validation; test reported.

| Attribute | Classes | Zero-shot F1 | Linear probe F1 | MLP F1 | Chosen: test accuracy |
|---|---|---|---|---|---|
| Article type | 59 | 68.3 | **81.4** | 81.9 | 90.5% |
| Sleeve length | 5 | 70.1 | 82.8 | **83.1** | 95.8% |
| Gender | 5 | 54.1 | 79.2 | **86.3** | 94.0% |
| Usage | 5 | 46.9 | 67.0 | **69.6** | 92.8% |
| Neck | 11 | 49.0 | 61.6 | **64.2** | 89.7% |
| Material | 9 | 32.2 | 59.1 | **62.6** | 65.1% |
| Pattern | 10 | 41.9 | **57.8** | 61.7 | 86.3% |
| Fit | 3 | 37.1 | 56.7 | **60.6** | 82.6% |
| Season | 4 | 25.4 | 53.1 | **57.4** | 72.3% |
| Fabric | 11 | 9.5 | 43.3 | **44.0** | 81.1% |
| Colour | 34 | 37.8 | 42.9 | **41.3** | 72.2% |

(F1 = test macro-F1 %. Bold = the method chosen on validation.)

**Colour, investigated.** Adding the colour histogram to the classifier's input changed nothing
(41.3 → 41.7). The errors are adjacent shades: Grey→Black (76), Blue↔Navy Blue (136), Silver→White,
Maroon→Red. Counting the colour family as correct raises accuracy from 72.2% to 78.5%. The ceiling
is label ambiguity between neighbouring shades, not missing signal.

**What the app shows:** type, colour, pattern, sleeves, neck, fit, material. Hidden: gender (a
learned bias, e.g. women's wide-leg jeans → "Men" at 100%), usage and season (weakest F1 yet
predicted at ~100% confidence), fabric (44 F1).

## 3.2 Enriching the luxury catalog
Farfetch has no attribute labels, so predictions were checked against independent signals:

| Check | Result |
|---|---|
| Category implied by the predicted type vs category from Farfetch titles (176k items) | **89.6% agree** |
| Predicted colour vs a single colour word in the title (4,139 items) | 61.4% exact, 63.0% same family (Myntra test: 72.2%) |

**Domain shift found:**
- **Coats → "Handbags".** Myntra's only kept outerwear type is "Jackets" (258 items); Farfetch has
  14k outerwear items. Shiny black coats were read as bags. **Fix:** type predictions are
  constrained to types consistent with the item's title category, so coats now become "Jackets".
- **"Synthetic" luxury.** 59% of Farfetch bags/shoes were predicted synthetic or synthetic leather,
  and 95% of items "casual" (Myntra has no evening/occasion class). **Fix:** only type, colour,
  pattern, sleeves and neck are shown for Farfetch items.

## 3.3 Style clusters (scripts/05_style_clusters.py)
Per category: UMAP (cosine, 10-d) → k-means (k = items / 1500, clamped to 8–20); separate 2-D
UMAP for the map. Named by the style words most *distinctive* for each cluster in Marqo's
image-text space.

| Category | Items | Clusters | Silhouette (UMAP space) |
|---|---|---|---|
| Top | 36,155 | 20 | 0.39 |
| Shoes | 24,802 | 17 | 0.46 |
| Bag | 24,717 | 16 | 0.40 |
| Bottoms | 24,172 | 16 | 0.46 |
| Dress | 21,192 | 14 | 0.34 |
| Accessory | 17,938 | 12 | 0.52 |
| Outerwear | 13,982 | 9 | 0.52 |
| Jewellery | 13,076 | 9 | 0.43 |

**Iterations:**
1. HDBSCAN gave 2 clusters for 21k dresses but 35 for 25k bags, and left up to 32% of items
   unassigned. A merchandising map needs every item assigned at comparable granularity, so k-means
   on the same UMAP space was used instead.
2. Naming by raw text similarity labelled most clusters "logo print · statement". Those words
   are close to every fashion image in the text space (hubness). Scoring each word against the
   category average fixed it.

**Merchandising insight (bags):**

| Style | Median price | On sale | Pre-owned | Signature brands |
|---|---|---|---|---|
| quilted · gold-tone | S$6,349 | 0% | 97% | Chanel |
| quilted · chain strap | S$2,817 | 5% | 0% | Saint Laurent |
| monogram canvas · vintage | S$2,190 | 0% | 62% | Louis Vuitton, Gucci, Dior |
| logo print · sporty | S$608 | 26% | 1% | Chiara Ferragni, Adidas |

Iconic quilted and monogram bags are almost never discounted and trade mostly pre-owned, while
dresses sit at ~50% on sale in nearly every style.

**Limitations:** some second name words are noise (white sneakers named "monogram canvas · all
white"); silhouettes of 0.34–0.52 mean the styles are a continuum with soft boundaries, not
cleanly separated groups.

## 3.4 Garment detection (src/silhouette_vision/detect.py, scripts/exp_detection_crop.py)
Detector: YOLOS fine-tuned on Fashionpedia (46 classes, MIT license), ~0.3 s per photo on CPU.
Grounding DINO (open-vocabulary) was tried first: 3–7 s per photo on CPU, and it missed a
clearly visible black dress.

LookBench RealStreetLook, 981 street photos, Marqo, exact-item recall:

| Query | R@1 | R@5 | R@10 |
|---|---|---|---|
| Full photo | 30.4 | 46.3 | 50.6 |
| **Crop of the item the user picks** | **37.3** | **55.2** | **61.1** |
| Crop of the detector's top item (automatic) | 15.5 | 23.5 | 27.0 |
| Official crop (upper bound) | 41.9 | 62.7 | 67.5 |

The detector found the query's garment type in 51% of photos (any garment in 98%); otherwise the
full photo is used. Even so, the picked crop recovers 60% of the gap to a perfect crop.
**Auto-picking is harmful:** outfit photos contain several items, and the most confident box is
usually not the one the shopper wants. **App design:** detect all items and let the user choose;
default to the whole photo when more than one item is found.

## 3.5 "Precise match": the Marqo + GR-Lite blend in the app
GR-Lite embeddings were computed for the 41,906 Myntra products (the public gallery; ~2.8 h on
the M4). The app's "Precise match (Myntra)" toggle blends the two scores 50/50, the weight chosen
on held-out LookBench queries in Phase 2 (+4.7 exact R@1 overall, +7.6 on street photos).

On the test photos the blend changes the *kind* of similarity: GR-Lite weighs colour and texture
more, Marqo weighs silhouette more. For the burgundy Miu Miu slingbacks, Marqo alone returned
pointed slingbacks in mixed colours; the blend returned all maroon/burgundy/red patent shoes, but
in mixed shapes (flats, sandals, a wedge). For the black lace mini dress, the blend returned more
black mini dresses and fewer robes and nightwear. It costs ~160 ms extra per query and a second
model (1.2 GB), so it is optional and off by default.

## App changes
- **Garment picker:** thumbnails of each detected item; the search, colour matching and attributes
  use the chosen crop.
- **"What we see":** predicted attributes for the uploaded photo, with confidence.
- **Attribute tags on results** (trusted attributes only for Farfetch).
- **Precise match (Myntra)** toggle: Marqo + GR-Lite blend.
- **Style map tab:** interactive UMAP map per category, a style table (price, markdown,
  pre-owned, signature brands), and sample products per style.
