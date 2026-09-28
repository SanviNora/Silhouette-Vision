# Silhouette Vision — Data & Model Research (Sept 2026)

**Approved 2026-09-27:** core = A, B, C, D; H&M (E) = backup; F and G dropped.

Checked against source pages, Kaggle/Hugging Face APIs and papers on 2026-09-27.
Anything marked *verify on download* could not be confirmed without logging in.

## Recommended data stack

| # | Dataset | Role | Size | License / access | Phase |
|---|---|---|---|---|---|
| A | Fashion Product Images (full) | Attribute labels, retrieval benchmark, open demo gallery | 44k products, 15.7 GB | MIT, Kaggle | 1 |
| B | Farfetch Listings (2019 scrape) | **Real luxury catalog**: brands, prices, markdowns, stock | ~3.9 GB, 300 px images | Unknown (scraped) → portfolio use only, no redistribution | 1 |
| C | LookBench | Model selection + street-photo query evaluation | ~2.4k queries, ~60k gallery per subset | Apache-2.0, Hugging Face | 1–2 |
| D | Visuelle 2.0 | **Cold-start demand** with images | 5,355 products, 110 stores, weekly sales | CC BY-NC-SA 4.0, Google form | 5 |
| E | H&M Personalized Fashion | Large-scale cold-start + recommendations, text descriptions | 105k articles, 31.8M transactions | Kaggle competition rules (non-commercial) | 5 |
| F | Fashionpedia | Fine-grained attributes (material, neckline, silhouette) for explanations | 48,825 images, 294 attributes | CC BY 4.0 | 3–4 (optional) |
| G | Branded Bottoms Resale | Resale value by brand/condition | 5k+ listings with images, 2.6 GB | CC BY 4.0, Kaggle (May 2025) | 6 (optional) |

### A. Fashion Product Images — `paramaggarwal/fashion-product-images-dataset`
- 44,446 products (Myntra, India). `styles.csv`: `id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName`. The full version adds `styles/<id>.json` (description, brand, price, *verify on download*).
- **Use the full version (15.7 GB, 2400×1600 images), not "small".** The small version's images are ~60×80 px (*verify on download*), too small for 336–384 px models. Download, resize to 512 px, and delete the originals.
- MIT license, so this is the one gallery that can safely be shown in a public demo.
- Limits: mass-market; no demand data.

### B. Farfetch Listings — `alvations/farfetch-listings`
- Scraped from farfetch.com in July 2019 with the open `fmarket` scraper. The scrape write-up reports ~180k products; the count in the Kaggle file is to be confirmed on download.
- Columns (confirmed from the scraper's sample output): `brand.name, gender, shortDescription, priceInfo.initialPrice, priceInfo.finalPrice, priceInfo.isOnSale, priceInfo.discountLabel, merchandiseLabel ("New Season"), stockTotal, merchantId, images.cutOut, images.model, availableSizes`.
- 300 px cut-out (white-background) product images, plus on-model image URLs.
- Why it matters: this is **real luxury data** (Golden Goose, Moschino, and many more), so the luxury layer becomes mostly real rather than synthetic. It supports price-tier modeling, markdown (`isOnSale`) analysis, and brand-level visual clustering.
- License listed as "Unknown" (scraped): use for analysis and a private demo only; don't redistribute the images or deploy them publicly.

### C. LookBench — `srpone/look-bench` (Hugging Face, v2025-12-01)
- Subsets: RealStudioFlat (1,011 queries / 62,226 gallery, easy), AIGen-Studio (192 / 59,254), RealStreetLook (1,000 / 61,553, hard: street photos → products), AIGen-StreetLook (160 / 58,846).
- Fields: `image, category, main_attribute, other_attributes, bbox, item_ID, difficulty`.
- Why: it evaluates exactly what the app does (an uploaded photo → a product), and it has a public leaderboard to compare our numbers against.

### D. Visuelle 2.0 — humaticslab.github.io/forecasting/visuelle
- Nuna Lie (Italian fast fashion). 5,355 products, 110 stores, 6 seasons (Nov 2016 – Dec 2019).
- Weekly sales per store, inventory, restock flags, normalized price, discounts, release dates.
- One image per product (median 575×722, white background); tags: 27 categories, 10 colours, 59 fabrics.
- Exogenous data: Google Trends per attribute, weather. Also 667,086 anonymized customers with baskets.
- Built for **new-product forecasting from images**, with published baselines (GTM-Transformer, MDiFF, etc.) to compare against.
- Access: submit the Google form and a download link is sent. CC BY-NC-SA 4.0. **Submit the form early.**

### E. H&M Personalized Fashion Recommendations (Kaggle competition)
- `articles.csv` (105,542 rows): `article_id, product_code, prod_name, product_type_name, product_group_name, graphical_appearance_name, colour_group_name, perceived_colour_value_name, perceived_colour_master_name, department_name, index_group_name, section_name, garment_group_name, detail_desc`.
- `transactions_train.csv` (31.8M rows, Sep 2018 – Sep 2020): `t_dat, customer_id, article_id, price (normalized), sales_channel_id`.
- `customers.csv` (1.37M rows) and one image per article (the full download is roughly 30+ GB, mostly images).
- Role: large-scale cold-start at scale (launch date = first sale), cold-start recommendation (MAP@12), and rich text descriptions.
- Access: join the competition and accept its rules; the data is for non-commercial use. Download only the images you need.

### F. Fashionpedia — fashionpedia.github.io
- 48,825 images with segmentation masks; ontology of 27 categories, 19 garment parts, 294 fine-grained attributes (silhouette, neckline, length, material, pattern, opening type).
- These are the attributes explanations need ("both have a *notched lapel* and *wool* texture"). Optional.

### G. Branded Bottoms Resale — `bathingtape/branded-bottoms-resale-dataset`
- 5k+ resale listings with images: Acne Studios, Rick Owens, Saint Laurent, Kapital, Carhartt, Levi's. Condition, price, seller, specs. CC BY 4.0.
- Small and trousers-only, but it is the only open resale dataset with images found. Pair it with Farfetch retail prices by brand to compare resale and retail prices by brand.

## Checked and not recommended

| Dataset | Why not |
|---|---|
| Vestiaire Collective 900k (Kaggle, justinpakzad) | **Removed**: Kaggle page returns 404. Only available through paid scrapers or services, which may breach the site's terms. |
| SSENSE / Net-a-Porter (Kaggle, justinpakzad) | Text only (brand, description, price, gender), under 3 MB. No images. |
| Fashion-Gen (SSENSE, 325k images) | High quality, but access was by request and current availability is uncertain. |
| DeepFashion / DeepFashion2 | Research-only license and request form; older. LookBench covers street-to-shop queries. |
| Amazon Reviews 2023 (Clothing) | 7.2M items, but it has reviews rather than sales and very noisy catalog data; weak for luxury. |
| FashionIQ | Image URLs point to Amazon and many are dead. |
| Farfetch SIGIR-2022 outfits (400k products) | Given only to challenge participants; not publicly released. |

## Embedding models (updated from the original plan)

Fine Recall@1 on LookBench (paper Table, arXiv 2601.14706):

| Model | License | Overall | RealStreetLook | RealStudioFlat |
|---|---|---|---|---|
| GR-Lite (DINOv3-L fine-tuned, 1024-d, 336 px, image-only) | Apache-2.0 | **65.7%** | 60.1% | 68.7% |
| Marqo-FashionCLIP | Apache-2.0 | 63.2% | 56.9% | 66.4% |
| Marqo-FashionSigLIP (image + text) | Apache-2.0 | 62.8% | 55.2% | 66.2% |
| SigLIP2-B/16 (generic) | Apache-2.0 | 59.4% | 51.9% | 62.8% |
| DINOv3-ViT-L (generic) | — | 44.0% | 37.4% | 54.7% |
| CLIP-B/16 (generic) | MIT | 31.9% | 22.4% | 45.3% |

ZooClaw-FashionSigLIP2 (SigLIP2-B/16 at 384 px, 768-d, Apache-2.0, released Jul 2026) beats
Marqo-FashionSigLIP on text→image retrieval (ZooClaw-Fashion short queries R@1 0.423 vs 0.371; H&M R@10 0.136 vs 0.114).

**Plan:**
- **Image → image search:** GR-Lite.
- **Text search, zero-shot attributes, explanations:** ZooClaw-FashionSigLIP2 and Marqo-FashionSigLIP. These need a shared image–text space, which GR-Lite doesn't have.
- **Baselines:** FashionCLIP 2.0 (the original plan), generic CLIP, DINOv3.
- **Caveat:** LookBench, GR-Lite and ZooClaw all come from the same group, so we re-benchmark every model on our own splits before choosing.

## Actions needed from you
1. Kaggle account + API token (`~/.kaggle/kaggle.json`).
2. Join the H&M competition on Kaggle and accept its rules (needed only for Phase 5).
3. Submit the Visuelle 2.0 Google form now, because the link comes back by email.
4. Disk: roughly 25 GB for Phase 1 downloads (261 GB free).
