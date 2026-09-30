# Phase 1 Results — Visual Search MVP

## Update 2026-09-29: recent catalog (2022–2026)
The search catalog was rebuilt from recent, shareable sources only; Myntra (median product year
2012) and Farfetch (2019, no redistribution) became a local *legacy* catalog used to train the
attribute heads and to reproduce the earlier analyses. One app, identical locally and online.

| Source | Products | Years | License |
|---|---|---|---|
| ZooClaw-Fashion | 11,999 (2,086 brands, H&M to Versace) | 2026 | CC BY-NC 4.0 |
| Second-Hand Fashion (RISE, Wargön, Myrorna) | 31,936 donated garments, 382 brands | 2022–2024 | CC BY 4.0 |
| LookBench studio gallery | 1,048 | 2025 | Apache-2.0 |
| Amazon Berkeley Objects, footwear | 6,064 shoes, boots, sandals (56 brands) | c. 2019–2021 | CC BY 4.0 |
| **Total** | **51,047** | | |

**Checks and decisions**
- **LookBench's 58,275 "noise" images are not recent.** 27.6% of a random Fashion200k (2017)
  sample has a near-identical copy (cosine ≥ 0.97) among them, which is what a random 58k-of-201k
  slice of Fashion200k would give (29%); the real 2025 gallery matches 0%. Left out.
- **Second-Hand photos** show garments lying sideways on a sorting table. On 2,000 garments
  (scripts/exp_secondhand_view.py):

  | Preparation | Type from 10 nearest neighbours | Zero-shot type |
  |---|---|---|
  | Raw photo | 50.7% | 47.3% |
  | Rotated upright | 64.0% | 57.6% |
  | **Rotated + centre crop (used)** | **65.6%** | **62.0%** |
  | Rotated + garment-mask crop | 60.5% | 57.3% |
  | Rotated + detector crop | 65.2% | 60.1% |

  The simple crop won; the "smarter" crops failed on white garments against white tiles.
- **Data cleaning:** "Missing" / "Not Applicable" brands blanked (3,535); 31 sliver-shaped photos
  (aspect > 3:1) dropped; ZooClaw titles de-duplicated against the brand and stripped of sizes.
- **Embedding precision** (LookBench, 62k-item gallery): 16-bit storage keeps R@1/R@10 identical
  (55.39 vs 55.49 studio; 41.79 vs 41.79 street); 8-bit costs 0.2–0.4 R@1 for 4× less memory.
  Embeddings are stored at 16 bit. With 45k products memory is no longer a constraint (~140 MB).

**Test photos, before → after** (mean top-5 similarity 0.705 on the old public Myntra catalog →
0.723 → **0.737** with footwear): clothing clearly improved (black mini dress → black slip/mini
dresses; jeans → wide-leg jeans; polka dots → polka-dot tops). The recent sources had only 7
shoes, so sneakers matched New Balance *sweatshirts* by logo. **Fix:** ABO footwear. Mules now
return cream block-heel mules (top similarity 0.810, above the old Farfetch catalog's 0.793),
sneakers return white chunky sneakers, burgundy slingbacks return pointed heels.
ABO listings repeat one shoe across marketplaces and sizes: deduplicated by photo (16,818 → 6,068).


**Date:** 2026-09-27 · **Model:** Marqo-FashionSigLIP (768-d) · **Catalog:** 229,522 products
(41,906 Myntra + 187,616 Farfetch) · **Hardware:** Apple M4, 16 GB

## What was built
- Unified catalog across Myntra and Farfetch with 9 shared categories (`scripts/01_build_catalog.py`).
- Embeddings for every product image, in resumable chunks (`scripts/02_embed.py`), ~350 MB as float16.
- Exact cosine search with filters applied before ranking (`src/silhouette_vision/search.py`).
- Streamlit app: search by photo, by text, or photo refined by text; filters; "More like this" (`app/ui.py`).

## Speed
| Step | Time |
|---|---|
| Embed the whole catalog | ~75 min on the M4 GPU (32–55 img/s depending on memory pressure) |
| Embed one query photo | **26 ms** (M4 GPU) |
| Search 229,522 products | **15 ms** (NumPy, exact) |

## Qualitative test: 11 real-world photos
Photos chosen to stress the system: luxury logos, street photos, a runway shot, a close-up on
feet, and a two-item outfit. None of the exact items are in the catalog (it dates from 2011–2019),
so the test is whether results are visually and stylistically similar.

| # | Query | Top-8 result quality | Notes |
|---|---|---|---|
| 1 | Louis Vuitton Félicie pochette (studio) | ★★★ | **8/8 Louis Vuitton monogram** bags (pre-owned Farfetch). Logo patterns are captured very strongly. |
| 2 | Black lace-trim mini dress (street, on person) | ★★★ | Black mini/slip dresses with lace trim, the key detail |
| 3 | Cream square-toe mules (close-up on feet) | ★★★ | Cream and nude block-heel mules; even woven texture matched. Background and feet ignored |
| 4 | White cowl-neck maxi dress (street) | ★★★ | White/ivory long slip dresses; 1 black ruched mermaid dress (same shape, wrong colour) |
| 5 | New Balance 1080 running shoes | ★★☆ | White running sneakers, 3 New Balance; matched on shape and colour more than brand |
| 6 | Chanel logo dress (runway) | ★☆☆ | **Weakest.** Sleeveless printed shift dresses, 2 Chanel. Busy print dominates; the white base is lost |
| 7 | Miu Miu burgundy patent slingbacks (on a sofa) | ★★☆ | Shape and material right (pointed patent slingbacks, Prada / Manolo Blahnik); **colour drifts to red** |
| 8 | Yellow lace-trim button top (on model) | ★★☆ | Lace-trim camisoles with buttons; **pale yellow read as white/cream** |
| 9 | Dark wide-leg jeans (on model) | ★★★ | **8/8 wide-leg jeans**, top 4 dark rinse |
| 10 | Polka-dot halter top + shorts (two items) | ★★☆ | Polka pattern dominates; results mix dresses, tops and shorts |
| 11 | Denim-effect tote with brown trim | ★★★ | Canvas/denim totes with brown leather trim |

**Similarity scores track difficulty:** easy queries score 0.79–0.83, hard ones (runway, two
items, unusual colour) 0.72–0.74. That's a usable confidence signal for the app.

## What the model captures well, and what it doesn't
- **Strong:** logo and pattern (LV monogram, polka dots), silhouette (wide-leg, maxi, slip dress), material and texture (lace, patent, woven).
- **Weaker:** subtle colour (burgundy → red, pale yellow → cream), busy runway images, and photos with more than one item.
- **Catalog balance:** Farfetch is 82% of the catalog and dominates most results; Myntra shows up for mass-market queries (sneakers, casual totes), which is appropriate.

## Fixes built into the app, tested
| Problem | Fix | Result |
|---|---|---|
| Colour drift (Q7) | Photo + text: "burgundy patent leather", weight 0.3 | Results 5–8 became burgundy/plum patent heels; top 4 still red. It helps, but a single mixing weight is blunt |
| Two items in one photo (Q10) | Category filter | "tops" → polka-dot tops; "bottoms" → polka-dot shorts/trousers. Clean fix |

## Next (Phase 2 onwards)
1. **Quantitative evaluation on LookBench** (Recall@1/5/10) against the public leaderboard, and decide whether GR-Lite earns its cost.
2. **Automatic item cropping** (detect each garment, then search per item) to fix multi-item and street photos.
3. **Colour-aware re-ranking or explanations**, since colour is the most visible weakness.
