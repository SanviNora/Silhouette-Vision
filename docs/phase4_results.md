# Phase 4 Results — Explaining Results (in progress)

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

## Next in Phase 4
"Why it matches": attribute overlap between the query and each result, plus which image regions
drive the similarity.
