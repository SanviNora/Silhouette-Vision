# Phase 5 Results — Cold-Start Demand Forecasting

**Question:** can a brand-new product, with no sales history, borrow a demand forecast from past
products that look like it?
**Data:** Visuelle 2.0: 5,355 launches of Nuna Lie (Italian fast-fashion womenswear), 110 stores,
12 weeks of sales per product and store, six seasons 2017–2019 (CC BY-NC-SA).

## TL;DR
- Final model (gradient boosting on store, launch timing, distribution breadth, price, tags and
  image features): **34.6% WAPE** on weekly product sales for 1,900 products never seen in
  training, vs 43.7% for a seasonal average and 39.8% for pure look-alike averaging. 12-week
  totals: 14.4% WAPE, bias −1%; for 80% of products the actual total is 0.82–1.23× the forecast.
- **The photo helps, modestly and significantly:** −1.0 WAPE points (95% CI 0.5–1.5).
- **The strongest signal is distribution breadth** (number of stores; rank correlation 0.64 with
  sales per store), which carries the planners' own expectations. Without it, everything a
  product's photo and tags say about demand ranks sales per store at only ~0.2.
- **Concept drift found and fixed:** in 2019 the brand widened distribution while per-store sales
  of widely stocked products fell. With the raw store count, the model over-forecast by 19%;
  using the count's percentile within the season fixed it (bias −1%).

## 5.1 Setup (scripts/08_visuelle_prepare.py, scripts/09_demand.py)
**Strict cold-start split:** train on SS17, AW17, SS18, AW18; test on **all** SS19 + AW19
products (1,900 products, 44,327 product-store rows). Settings chosen by training on SS17–SS18
and validating on AW18.

**Why not the official split?** `stfore_train/test` splits *product-store rows*: 503 of its 749
test products also sell in other stores in the training file, so a model can learn their
popularity. Fine for the benchmark's purpose, but not a new-product test.

**Known at planning time, used:** store, launch week/month, category, colour, fabric, price,
number of stores, photo. **Not used:** `restock` (stock delivered during the season) and weekly
discounts, which are only known afterwards.

**Data notes:** one image (AI19/04442.png) is truncated and loaded partially; season launch
windows overlap by a few months (a product launches in different stores at different dates).

## 5.2 Results (test: SS19 + AW19)
Primary metric: WAPE of weekly sales per product (summed over stores). Store-week values are
~1 unit, almost pure noise, so their WAPE (~80%) barely separates models.

| Model | Weekly product WAPE | 12-week total WAPE | Bias | Rank of sales per store |
|---|---|---|---|---|
| Seasonal average | 43.7 | 32.9 | +5.8% | −0.08 |
| Tag average (category/colour/fabric) | 41.2 | 30.1 | −17.3% | 0.09 |
| Look-alikes (k = 100 nearest photos) | 39.8 | 29.4 | −16.1% | 0.04 |
| Boosting: basics (store, timing, breadth, price) | 35.4 | 14.6 | −1.8% | 0.70 |
| + tags | 35.6 | 14.7 | −0.8% | 0.66 |
| **+ tags + image (final)** | **34.6** | **14.4** | **−1.1%** | 0.68 |
| Final, but raw store count | 44.1 | 27.0 | +18.9% | 0.69 |

Per season (final): SS19 36.2, AW19 33.2.

**What the product itself says** (same models without the number of stores):

| | Weekly product WAPE | Rank of sales per store |
|---|---|---|
| Tags | 41.0 | 0.23 |
| Image | 39.2 | 0.20 |
| Tags + image | 39.7 | 0.23 |

**Image effect (paired bootstrap over products, 1,000 resamples):** −1.03 WAPE points [0.52,
1.52] with the store count, −1.34 [0.62, 2.10] without.

## 5.3 What didn't work
| Idea | Result (validation AW18) |
|---|---|
| Google Trends, 52 weeks before launch (category/colour/fabric; as in published Visuelle models) | Rank correlation −0.14 to +0.04 with sales; −0.3 WAPE: not kept |
| Learned image score (ridge on the embedding) instead of look-alike averaging | 0.26 vs 0.24 rank correlation: no real gain, not kept |
| Poisson loss instead of log target | Worse (39.9 vs 37.6 WAPE, bias +12%) |
| Zero-shot text for suggested tags | Category 31% right; vote of the 15 nearest past products 77% |

## 5.4 Concept drift
| | SS17 | AW17 | SS18 | AW18 | SS19 | AW19 |
|---|---|---|---|---|---|---|
| Mean stores per product | 14.4 | 18.5 | 21.1 | 18.9 | 23.3 | 23.4 |
| Median units/store, products in 40+ stores | 19.8 | 19.3 | 17.6 | 16.9 | 15.9 | 15.6 |

Trees learned "wide distribution → high per-store sales" and cannot extrapolate to a season where
wider distribution is the norm. The within-season percentile keeps the planners' relative signal
without the drifting level. **Caveat:** the drift was discovered on the test set. The fix is a
principled change, not a tuned one, and it also improves validation (37.8 → 34.8 WAPE).

## 5.5 In the app: "New product forecast" tab
Upload a photo → garment picker → a type check (bags, shoes and jewellery are refused: the brand
sells only women's clothing) → suggested category/colour/fabric (editable) → planning inputs
(stores, price level, launch date) → 12-week total with an 80% range, per-store units, a weekly
chart, and the 8 look-alike past launches with their actual sales.

The range comes from the test products. Shoppers' photos (people, backgrounds) look less like the
brand's flat product shots (closest-product similarity 0.67–0.75 vs ≥ 0.81 for the brand's own
products); the app says so, and the forecast should then be read as rough. On the test photos,
forecasts for 18 stores range 184–214 units: the store count carries most of the prediction.

## Limitations
- One brand, one country, 2017–2019; per-store sales are small (median 12 units in 12 weeks).
- Sales are censored by stock: a sell-out looks like low demand. `restock` could flag it, but
  it is only known afterwards.
- No zero-sales rows exist in the data ("never sold" is unobserved).
