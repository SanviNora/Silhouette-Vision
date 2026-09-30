# Phase 4 Results — Explaining Results

## Update 2026-09-29: re-validated on the recent catalog (51,047 products)
**Exact-match likelihood** (calibrator unchanged: same Marqo model, fitted on LookBench):

| Check | Result |
|---|---|
| LookBench studio photos of products **in** the catalog (985 queries vs all 51k) | exact #1 48.9%; P ≥ 0.8 → 88% precision (22% of queries); P ≥ 0.9 → 93% |
| Street photos of products **not** in the catalog (954) | wrongly claimed exact: 4.3% at P ≥ 0.5, **0.1% at P ≥ 0.8** |
| Degraded catalog photos (crop, 70% resize, JPEG q60, brightness; 400) | original #1 in **95.8%** (86.2% on the old catalog) |

LookBench numbers items per subset (street item 3 is not studio item 3; shared ids share a
category only 37% of the time), so its street queries show products the catalog does not have:
a free test of false "exact" claims. The "Precise match" calibrator was removed with the feature.

**Named models:** 440 products from the listed luxury brands (Burberry, Fendi, Chloé, Gucci…);
28 of the 161 models have catalog listings (e.g. Celine Triomphe ×8, Hermès Kelly and Birkin,
Lady Dior). The others show "not stocked" with a web-search link. Recognition itself is
catalog-independent (the LV Pochette Félicie photo still scores 93%).

**Why it matches:** attribute comparison now uses one trust rule for all sources; types in one
family count as "≈ similar" (block-heel mule sandal vs "Heels", tunic vs top). Ethnic-wear types
that no recent source contains (Kurtis, Kurtas, Sarees, Dupatta) were removed from the type head:
an uploaded polka-dot top was called "Kurtis 54%", now "Tops 100%"; Myntra type macro-F1
81.9 → 83.1, Second-Hand type 79.6% → 80.7%.


## 4.1 Exact-match confidence (scripts/06_match_confidence.py, src/silhouette_vision/match.py)
**Question a shopper asks:** "Is this the exact product in my photo, or just something similar?"

**Raw similarity can't answer it.** Cosine similarity is not a probability. Calibrated on
LookBench (top-10 results of 2,345 queries with known products), similarity alone never exceeded
~64% precision, and even top results with similarity ≥ 0.95 were the exact product only 64% of
the time. A different photo of the same product and a photo of a similar product score alike.

**What works: whether the best result stands out.** Gap between #1 and #2 on LookBench:

| Gap ≥ | Share of queries | Top-1 is the exact product |
|---|---|---|
| 0.02 | 32% | 75% |
| 0.05 | 13% | 89% |
| 0.08 | 8% | 91% |

**Calibrator:** logistic regression on [top-1 similarity, gap to #2, gap to #5], fitted on
even-indexed queries, evaluated on odd-indexed ones:

| Say "exact" when P ≥ | Share of queries | Precision (held out) |
|---|---|---|
| 0.5 | 41% | 73% |
| 0.8 | 10% | 86% |
| 0.9 | 5% | 91% |

Held-out calibration error (ECE): 0.061 (Marqo), 0.058 (Precise blend, own calibrator).

**"Photo found online" simulation:** 400 catalog photos cropped to 85–95%, shrunk to 70%,
JPEG quality 60, brightness ±15%, then searched. The original came back **#1 in 86.2%**, with a
median confidence of 72%. The calibrator is deliberately conservative: it learned from LookBench,
where matches are *different photos* of a product, the harder and more common real case.

**In the app:** a "best match" panel above the results with three tiers: *Very likely the exact
product* (≥ 80%), *Possibly the exact product* (50–80%), and *No confident exact match - closest
alternatives* (< 50%). The raw similarity is shown separately and labelled as such. The confidence
uses pure similarity (no colour re-ranking), matching how it was calibrated.

**On the 11 test photos:** 10 of 11 were "No confident exact match" (23–49%). That is correct:
the catalog dates from 2011–2019 and these products aren't in it, even though raw similarities of
0.73–0.83 look like matches. A catalog photo re-saved as a small low-quality JPEG found its exact
product at 78% ("Possibly"). For recent internet photos to get exact hits, the catalog needs
those products; that's a data limit, not a model one.

## 4.2 Garment picker fixes (src/silhouette_vision/detect.py)
Reported from the app: a Louis Vuitton bag product shot offered a "skirt (51%)" item, and a
black dress photo offered "skirt (52%)" next to "dress (99%)".

Cause: the detector was trained on Fashionpedia (photos of people), so it's weak on product
shots and also reports garment *parts* (the lower half of a dress).

Fixes in `GarmentDetector.items`, checked on all 11 test photos:
- confidence ≥ 0.6 (false items were 0.51–0.55; real ones 0.62–0.99);
- drop skirt/top/pants detections lying inside a higher-scoring dress or jumpsuit;
- no crop offered when a single box covers > 75% of the photo (a product shot).

The detector's labels are no longer shown: it called a denim tote "skirt" at 96%, and our own
type classifier called a small crop of a chain bag "Bangle". Items appear as thumbnails
("Item 1", "Item 2"); after picking, "What we see" describes the item with confidences.

## 4.3 Named-model recognition (scripts/07_named_models.py, src/silhouette_vision/named_models.py)
Reported from the app: a Louis Vuitton Pochette Félicie photo (a current best-seller) got "no
confident exact match". Diagnosis: retrieval was fine (all top-10 were small LV monogram
pochettes), but the 2019 catalog has 767 LV items and **no Félicie**. Exact-product search can only
find what is stocked; the model name can be recognised regardless.

**Method:** zero-shot. The photo embedding is compared with text prompts ("a photo of the Louis
Vuitton Pochette Félicie bag", "Louis Vuitton Pochette Félicie", averaged) for 161 models across 21
brands in `configs/iconic_models.yml`. The item type (bag / shoes / belt / coat) is decided first
and only models of that type compete. A calibrator on [top score, gap to 2nd] gives P(correct).

**Evaluation:** 2,892 Farfetch photos whose titles name one listed model of their brand (93
distinct models) and 6,000 bags/shoes/accessories from unlisted brands, which should be rejected.
Calibrator fitted on even-indexed items, evaluated on odd-indexed ones.

| | |
|---|---|
| Top-1 / top-3 model accuracy | 92.0% / 98.0% |
| Nearest catalog photo's model name instead (LV subset) | 88.4% vs 93.7% zero-shot |

| Name a model when P ≥ | Precision | Listed-model photos named | False alarm, unlisted brands |
|---|---|---|---|
| 0.5 | 91.4% | 90.1% | 1.8% |
| 0.8 | 96.0% | 67.1% | 0.3% |
| 0.9 | 97.1% | 40.8% | 0.1% |

Per brand top-1: 83–100% for most brands; weakest Bottega Veneta 77% (Pouch/Knot vs Andiamo),
Valentino 87% (Rockstud vs Roman Stud). Celine has only 3 test photos (all wrong); too few to judge.

**Internet photos:** the Félicie screenshot is named at **93%**; the 10 other test photos (dresses,
jeans, sneakers, a denim tote...) score ≤ 6% except red T-bar pumps at 38% (J'Adior), all below
the 50% claim threshold. Degraded catalog photos (crop, resize, JPEG, brightness): Neverfull,
Speedy and Birkin each named correctly in 14/15, Baguette 11/15.

**Two bugs found while building it:** (1) listing "Rockstud" as both a bag and pumps split one
name in two (Valentino 64% → 87% after merging); (2) the catalog spells brands "Christian Dior"
and "Céline", so Dior had no test photos and would show no listings (fixed with brand aliases).

**In the app:** above the exact-match panel, "Recognised model: Louis Vuitton Pochette Félicie
(93%)" (≥ 80%; "Probably" at 50–80%), followed by our listings of that model ranked by visual
similarity, or "Not stocked in our catalog (a 2019 snapshot)", plus a web search link.

## 4.4 Why it matches (src/silhouette_vision/explain.py, scripts/exp_explain_heatmap.py)
Two explanations, shown under the best match ("Why this matches") and, in short form, on every
result card.

**Shared attributes.** The photo's predicted attributes side by side with the item's (type,
colour, pattern, sleeves, neck; trusted ones only on Farfetch), each marked same (✓), similar
(≈, a neighbouring shade such as Navy Blue vs Blue, the colour classifier's main error) or
different (✗). Only confident predictions (≥ 50%) on both sides are compared. Cards show
"Shares: Tops · Blue" / "Differs: Striped (yours: Solid)". This is a readable summary of what
the two have in common, *not* the model's reasoning; the heatmap is.

**Heatmap: where the match comes from.** Four methods, tested on 150 LookBench street photos
against each photo's true product. Pointing = the hottest cell lies inside the garment's box.
Deletion = similarity drop when the hottest 25% of the image is hidden (a faithful map must
beat hiding a random 25%).

| Method | Pointing | Deletion drop: hottest / random / coldest 25% | Hottest beats random | Time per map |
|---|---|---|---|---|
| **Occlusion, 8×8 grid (shipped)** | **67%** | **0.070** / −0.010 / −0.077 | **95%** | 1.2 s |
| Occlusion, 12×12 grid | 59% | 0.066 / −0.008 / −0.065 | 95% | 3.1 s |
| Attention-pooling attribution | 11% | 0.043 / −0.024 / −0.071 | 94% | 0.05 s |
| Gradient × input on patch tokens | 27% | −0.012 / −0.019 / −0.035 | 61% | 0.11 s |

Baselines for pointing: a random cell 21%, the centre cell 60–65%.

- **Occlusion** hides one window at a time and measures how much the similarity drops:
  slow (~50 forward passes) but faithful by construction, and it is the only method whose
  hottest regions both land on the garment and matter when removed.
- **Attention-pooling attribution** splits Marqo's pooled embedding exactly into per-patch
  terms and is 25× faster, but its hottest patch lands on the garment *less often than chance*.
  The heat sits on scattered background patches: the ViT "register" effect, where a few
  low-information patches are repurposed to store global image information. The map is exact
  about the arithmetic but misleading about *where in the photo* the evidence is.
- **Gradient × input** fails deletion (its top regions are no more important than random ones).
- Honest caveat: on street photos the garment is usually central, so "always point at the
  centre" is a strong baseline (60%). Occlusion beats it only modestly on pointing; its real
  evidence is the deletion test.

**In the app:** attributes appear instantly; the heatmaps (the photo, and the match) sit
behind a toggle ("a few seconds") so searching stays fast. On the test photos occlusion
highlights the LV logos and flap of the Félicie, the waistband and leg line of the wide-leg
jeans, and the body of the denim tote.

## Phase 4 status
Complete: exact-match confidence (4.1), garment picker fixes (4.2), named-model recognition
(4.3), why it matches (4.4). Next: Phase 5, cold-start demand forecasting (Visuelle 2.0).
