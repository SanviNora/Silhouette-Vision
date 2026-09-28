# Silhouette Vision — Data Guide

What is inside every dataset, in plain language, with the real numbers. Use this to explain the
data in interviews.

| Status | Dataset |
|---|---|
| ✅ Downloaded and profiled from our copy | Visuelle 2.0, LookBench, Farfetch Listings, Fashion Product Images |
| 🟡 Tables profiled from a public mirror (backup, not downloaded) | H&M |

Profiled 2026-09-27.

---

## The one-minute version

| Dataset | What it is, in one sentence | Why it's in the project |
|---|---|---|
| **Fashion Product Images** | 44k product photos from Myntra (Indian e-commerce) with hand-entered labels, prices, brands and detailed attributes | Clean labels to train and test attribute prediction; MIT license, so it's the public demo gallery |
| **Farfetch Listings** | A July 2019 snapshot of Farfetch's women's catalog (Singapore site): 187k products, 2,336 brands, prices, markdowns, stock | Real luxury products to search over and analyze by price, including 10.6k pre-owned listings |
| **LookBench** | A 2025 benchmark of photo → product matching, including street photos | Picks the best embedding model with numbers comparable to a public leaderboard |
| **Visuelle 2.0** | Three years of weekly store sales for 5,355 new products from an Italian fast-fashion brand, each with a photo | Tests "can a new product borrow demand from look-alike past products?" |
| **H&M** *(backup)* | Two years of H&M online transactions (31.8M) over 105k products with photos and descriptions | Scale check for the demand results; fallback if Visuelle falls short |

How they fit together: **LookBench** chooses the model → the model embeds **Myntra + Farfetch**
(search, attributes, clusters) → the same model embeds **Visuelle** products to predict demand for
new items from their visual neighbours → **H&M** repeats that at 20× the scale if needed.

---

## 1. Visuelle 2.0 ✅

**Source:** HumaticsLab, University of Verona, with the Italian fast-fashion brand **Nuna Lie**
(womenswear, 100+ stores in Italy). Paper: Skenderi et al., *"The multi-modal universe of
fast-fashion: the Visuelle 2.0 benchmark"*, CVPR 2022 Workshops.
**License:** CC BY-NC-SA 4.0 (non-commercial, cite the paper, share derivatives alike).
**Our copy:** `data/raw/visuelle2/` (2.23 GB zip, checksum in `data/raw/_manifest.json`).

### Headline numbers
| | |
|---|---|
| Products | **5,355** (each has a photo) |
| Stores | **110** |
| Product × store rows | **106,850** (a product is sold in a median of **14 stores**) |
| Seasons | 6: SS17 (901 products), AW17 (852), SS18 (758), AW18 (944), SS19 (846), AW19 (1,054) |
| Release dates | 28 Nov 2016 → 30 Dec 2019 |
| Sales horizon | **12 weeks** after each product's release in each store |
| Customers | **665,285** anonymized loyalty customers, 3.18M purchase lines |

### Files
| File | Rows | What each row is | Key columns |
|---|---|---|---|
| `sales.csv` | 106,850 | One product in one store | `external_code` (product id), `retail` (store id), `season`, `category`, `color`, `fabric`, `image_path`, `release_date`, `restock`, `0`…`11` (units sold in weeks 1–12) |
| `price_discount_series.csv` | 106,850 | Same product × store | `0`…`11` = discount in each week (0.2 = 20% off), `price` = normalized price |
| `restocks.csv` | 949,766 | One restock event | `external_code`, `retail`, `week`, `year`, `qty` (covers 2016–2022) |
| `customer_data.csv` | 3,184,162 | One item bought by one customer | `customer`, `retail`, `external_code`, `data` (timestamp), `qty` |
| `vis2_gtrends_data.csv` | 220 weeks | Weekly Google Trends (Italy), Oct 2015 → Dec 2019 | 96 series: one per category, colour and fabric |
| `vis2_weather_data.csv` | 89,071 | Daily weather in one town | temperature, humidity, rain, wind; 61 localities |
| `stfore_train.csv` / `stfore_test.csv` | 96,166 / 10,684 | **Official split** (sales divided by 53) | Test = **749 products** from SS19 + AW19 |
| `category_labels.pt`, `color_labels.pt`, `fabric_labels.pt` | — | Label → integer mappings used by the authors' code | PyTorch files |
| `shop_weather_pairs.pt` | — | Which weather town belongs to which store | PyTorch file |
| `images/{AI17…PE19}/*.png` | 5,355 | One product photo | **AI** = *Autunno-Inverno* (Fall/Winter), **PE** = *Primavera-Estate* (Spring/Summer) |

**Column meanings to remember**
- `restock` in `sales.csv` is the **total stock delivered to that store**. The authors' code uses it as the cap on what could be sold.
- `price` is **normalized** (0.011–0.571) for confidentiality. It works for comparisons, but it isn't euros.
- `release_date` is **per store**: the same product can launch on different dates in different stores.

### Labels
- **27 categories** (all womenswear): long sleeve 1,173 · culottes 915 · sleeveless 468 · solid colour top 453 · short sleeves 310 · doll dress 305 · patterned top 193 · long dress 189 · kimono dress 167 · trapeze dress 129 · … · long duster 17.
- **10 colours:** black 2,111 · brown 571 · white 568 · blue 546 · grey 409 · violet 407 · green 313 · red 229 · yellow 181 · orange 20.
- **59 fabrics:** georgette 730 · "nice" 562 · cotton 431 · matte jersey 270 · tulle 235 · scuba crepe 220 · acrylic 202 · faux leather 200 · lurex 179 · …

### What the sales look like
- Per product per store: **mean 15 units, median 12** over 12 weeks (max 268).
- Per product across all stores: **median 152, mean 299, top 10% above 740, max 6,120**. The distribution is heavily right-skewed, so we model log(units).
- **Sales decay after launch:** week 1 = 195k units across the chain, week 6 ≈ 144k, week 12 = 69k.
- **75%** of product-store rows get a discount at some point in the 12 weeks.

### Images
- White-background **flat product shots**, no model. PNG with a **transparent background (RGBA)**. Sizes vary (sample median ≈ 519×706 px).
- We composite them onto white before embedding; otherwise the transparent areas turn black.

### Quirks (good to mention; it shows you looked)
- **Coarse colour labels.** A hot-pink crochet halter is labelled `red`, because only 10 colours exist. Image embeddings capture what the labels miss, which is part of why vision helps.
- **Only rows with sales are included.** No product-store row has zero sales over its 12 weeks, so "never sold" isn't observed.
- **The customer file is broader than the sales file:** 129 stores, 5,577 products, and dates up to June 2021. Filter it to the 5,355 products and to dates before the test season.
- **Google Trends leakage rule** (from the authors' code): only use the **52 weeks before** release.

### How we use it
Cold-start demand (Stage 9): predict the 12-week sales curve of a **new** product from its image,
tags and price, using the official season split so results compare with published papers.
Customer baskets drive cold-start recommendation (who would buy this new item?).

---

## 2. LookBench ✅

**Source:** SerendipityOne / ZooClaw team. Paper: *"LookBench: A Live and Holistic Open Benchmark
for Fashion Image Retrieval"* (arXiv 2601.14706, Jan 2026). **License:** Apache-2.0.
**Our copy:** `data/raw/lookbench/v20251201/` (2.1 GB, released 11 Dec 2025).

### What it tests
Given a **query photo**, can the model find the **exact same item** in a large gallery?

| Subset | Queries | Own gallery | + shared distractors | Difficulty | What the query looks like |
|---|---|---|---|---|---|
| RealStudioFlat | 1,011 | 3,951 | 58,275 | Easy | Real product photo on a plain background |
| AIGen-Studio | 193 | 979 | 58,275 | Medium | AI-generated studio product image |
| RealStreetLook | 981 | 3,278 | 58,275 | **Hard** | **Real street photo of a person wearing the item**, the closest match to a customer's phone upload |
| AIGen-StreetLook | 160 | 571 | 58,275 | Hard | AI-generated street-style photo |

In total: 2,345 queries, 8,779 gallery items and 58,275 "noise" images added so the search is hard.

### Columns
`image` (cropped item), `raw_image` (full photo before cropping), `category` (blouse, dress, coat,
pants, bag, shoe, …), `main_attribute` (the defining detail, e.g. *leather, sequin, snakeskin,
crocodile*), `other_attributes` (e.g. *"applique, printed, bead"*), `bbox` (crop box), `item_ID`
(the match key), `task`, `difficulty`.

### Why it's credible
- Images were collected **recently**, so older models can't have seen them during training.
- It has a public leaderboard. Top open model: **GR-Lite, 65.7%** fine Recall@1. Generic CLIP-B/16 scores 31.9%.
- Caveat: the same team publishes GR-Lite and ZooClaw-FashionSigLIP2, so we don't take their numbers on trust; we re-run everything.

### How we use it
Stage 2 (Measure): run every candidate model through all four subsets, report Recall@1/5/10, and
choose the backbone. RealStreetLook is the "customer uploads a phone photo" test.

---

## 3. Fashion Product Images (Myntra) ✅

**Source:** Param Aggarwal, Kaggle, 2019. Scraped from **Myntra** (India's largest fashion
e-commerce site). **License:** MIT. **Our copy:** `data/raw/fashion_product_images/fashion-dataset/`
(15 GB). The Kaggle zip (24.8 GB) contains the whole dataset **twice**; we extracted one copy.

### Headline numbers
| | |
|---|---|
| Products | **44,446** JSON records; `styles.csv` parses to 44,424 rows (22 lines break because product names contain commas); **44,441 images** (5 products have none) |
| Images | Studio photos, portrait, **1080×1440 or 1800×2400 px** |
| Gender | Men 22,104 · Women 18,357 · Unisex 2,126 · Boys 830 · Girls 655 |
| masterCategory (7) | Apparel 21,361 · Accessories 11,244 · Footwear 9,197 · Personal Care 2,139 · Free Items 105 · Sporting Goods 25 · Home 1 |
| subCategory | 45 (Topwear 15,383 · Shoes 7,323 · Bags 3,053 · Bottomwear 2,685 · Watches 2,542 · …) |
| articleType | **141** (T-shirts 7,065 · Shirts 3,212 · Casual Shoes 2,845 · Watches 2,542 · Kurtas 1,844 · Handbags 1,759 · Heels 1,323 · …); **73 types have fewer than 50 items** |
| baseColour | 46 (Black 9,699 · White 5,497 · Blue 4,906 · Brown 3,440 · …) |
| season / usage | Summer 21k · Fall 11k · Winter 8.5k · Spring 2.7k / Casual 34k · Sports 4k · Ethnic 3.2k · Formal 2.3k |
| Brands | **424** (Nike 2,203 · Puma 2,098 · Adidas 1,923 · United Colors of Benetton 1,563 · Fabindia 751 · …) |
| Price | **INR**, median ₹1,199 (10% below ₹395, 10% above ₹3,500, max ₹28,950); 16.6% discounted |
| Catalog dates | Added between Feb 2011 and Nov 2016 |

### Files
- `styles.csv`: `id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName` (e.g. *"Turtle Check Men Navy Blue Shirt"*).
- `images/<id>.jpg`: one studio photo per product. `images.csv` maps each file to its original Myntra URL.
- `styles/<id>.json`: the **full product record**, with everything in `styles.csv` plus:
  - `price`, `discountedPrice` (INR), `brandName`, `ageGroup`, `colour1`/`colour2` (secondary colours), `catalogAddDate`, `myntraRating` (almost always 0 or 1, so unusable)
  - `productDescriptors.description`: an HTML product description (99.9% of items)
  - `articleAttributes`: **structured attributes** (82.8% of items have at least one)

### The structured attributes (why they matter)
These are labelled "why it matches" features, so we don't need Fashionpedia:

| Attribute | Items | Values (top) |
|---|---|---|
| Pattern | 16,502 | Solid, Printed, Striped, Checked, Self Design, Lace, Embellished (16 values) |
| Fabric | 15,592 | Cotton, Blended, Polyester, Synthetic, Nylon, Viscose Rayon (24) |
| Sleeve Length | 12,836 | Short, Long, Sleeveless, Three-Quarter (5) |
| Occasion | 11,002 | Casual, Everyday, Western, Formal, Sports, Party (16) |
| Fit | 10,086 | Regular, Slim, Skinny, Loose (12) |
| Neck | 9,345 | Round, Polo Collar, V-Neck, Scoop, Mandarin, Boat (23) |
| Material (shoes, bags) | 5,120 | Synthetic Leather, Synthetic, Leather, Canvas, Mesh (21) |
| Type | 10,207 | Sandals, Driving Shoes, Shoulder Bag, Sneakers, … (125) |

Keys are category-specific: shirts get Collar and Fit, shoes get Ankle Height and Closure, bags get Material.

### Quirks
- The **"small" Kaggle version is 60×80 px** (checked), too small for modern models, which is why we use this full version.
- Very imbalanced: half the article types are rare, so we collapse types with fewer than 50 items into `Other` and report macro-F1.
- Includes non-fashion items (perfume, "Free Items", one "Home" product), which we filter out.
- Mass-market (Nike, Puma, Benetton), not luxury, and no demand data. Prices are in rupees.

### How we use it
Attribute classifiers (Stage 6), including the detailed ones (pattern, fabric, neck, sleeve, fit);
the labelled source for similarity explanations (Stage 8); clustering (Stage 7); a search benchmark
with loose relevance (same type + same colour); and the **public** demo gallery.

---

## 4. Farfetch Listings ✅

**Source:** Kaggle (`alvations/farfetch-listings`), scraped from **Farfetch's Singapore site** in
**July 2019** with the open-source `fmarket` scraper. Farfetch is a global marketplace where luxury
boutiques and brands list their stock. **License:** listed as "Unknown" (scraped).
**Our copy:** `data/raw/farfetch/` (3.8 GB zip).

### Headline numbers
| | |
|---|---|
| Rows / unique products | 188,817 rows / **187,616 products** (1,201 ids appear twice, almost all exact duplicates) |
| Images | **187,620 cut-outs + 187,620 on-model photos**, all **300×400 px** JPG |
| Gender | **women 176,978 · unisex 11,839**; no menswear (the scrape covered the women's catalog) |
| Brands | **2,336**. Top: Chanel Pre-Owned 2,605 · Prada 2,522 · Gucci 2,501 · Saint Laurent 2,490 · Dolce & Gabbana 2,340 · Marni 1,925 · Valentino 1,743 · Fendi 1,738 · Burberry 1,671 · Jimmy Choo 1,546 · … · Bottega Veneta 1,081 |
| Boutiques (`merchantId`) | 1,002 |
| Currency | **SGD (Singapore dollars)** for every row, although the formatted price shows only "$" |
| Price (final) | median **SGD 612**; 10% below 180, 10% above 2,273, 1% above 7,778, max 89,724 |
| On sale | **36.7%** of products; median markdown **40%** (most common labels: 50% · 30% · 40% · 20% off) |
| Stock | median **4 units**; 25.6% have exactly **1 unit** left; 10% have more than 24 |
| Labels | New Season 41,469 · Positively Conscious (sustainability) 12,324 · Permanent Collection 1,275 · Seasonal Pick 741 · Exclusive 632 |
| **Pre-owned** | **10,611 listings from 67 "Pre-Owned" brands**: Chanel 2,605 · Louis Vuitton 770 · Hermès 747 · Yves Saint Laurent 707 · Comme des Garçons 540 · Dior 495 · Fendi 464 · … |

Most expensive brands by median price (brands with 200+ items): Liska (furs) SGD 4,657 ·
Hermès Pre-Owned 3,269 · Louis Vuitton Pre-Owned 2,595 · Carolina Herrera 2,440 · Chanel
Pre-Owned 2,376 · Oscar de la Renta 2,244 · Tom Ford 1,926 · Brunello Cucinelli 1,830.

### Columns
| Column | Meaning |
|---|---|
| `id` | Farfetch product id; image files start with it (`<id>_<imageid>_300.jpg`) |
| `brand.id`, `brand.name` | Designer; "X Pre-Owned" marks second-hand listings |
| `gender` | women / unisex |
| `shortDescription` | Short title, e.g. *"fox fur-trimmed knit cashmere sweater"*, *"knuckle crocodile effect clutch"*, *"Nike Air sneakers"*. 118,706 distinct titles |
| `priceInfo.initialPrice` / `finalPrice` | Full price and current price (SGD) |
| `priceInfo.isOnSale`, `discountLabel` | Markdown flag and label ("50% Off") |
| `merchandiseLabel` | New Season, Positively Conscious, Exclusive, … (empty for 70%) |
| `stockTotal` | Units available across boutiques |
| `merchantId` | Boutique listing the item |
| `availableSizes` | Size list (missing for 26k, mostly one-size items) |
| `images.cutOut`, `images.model` | Image URLs; the files are in `cutout-img/` and `model-img/` |
| `hasSimilarProducts`, `isCustomizable` | Site flags (360 customizable items) |
| `priceInfo.installmentsLabel` | Always empty; drop it |

### No category column
Categories have to come from the title. A first keyword pass gives: tops 36.6k · bottoms 24.4k ·
bags 24.0k · shoes 23.5k · dresses 19.3k · outerwear 13.3k · accessories 13.2k · jewellery 12.3k ·
unmatched 22.1k. The final taxonomy will use keywords plus zero-shot image classification for the
unmatched items.

### Quirks
- **SGD, not USD.** Convert with the July 2019 rate (≈ 0.73 USD per SGD) if showing USD, and say so.
- A **single snapshot**: prices and stock, but no sales history. "Stock = 1" or a deep markdown are hints about demand, not sales.
- Pre-owned prices are **asking prices**, not sold prices, and they're different items from the new ones.
- Scraped data: analysis and a **private** demo only; the images are not redistributed.

### How we use it
The luxury search gallery; "which brands and visual styles go with higher prices or markdowns"
(Stage 10); brand-level style clusters. Optional: compare **pre-owned vs new price by brand and
category** (e.g. Gucci vs Gucci Pre-Owned). This is a light resale-value angle that needs no extra dataset.

---

## 5. H&M Personalized Fashion Recommendations 🟡 (backup)

**Source:** H&M Group, Kaggle competition (2022). **License:** competition rules, non-commercial.
**Size:** ~30+ GB (mostly images); tables are ~1 GB.

### Headline numbers (from a public mirror of the tables)
| | |
|---|---|
| Articles | **105,542** (47,224 `product_code`s; an article = one colour variant of a product) |
| Transactions | **31,788,324**, from 20 Sep 2018 to 22 Sep 2020 |
| Customers | 1,362,281 who bought something (1.37M in `customers.csv`) |
| Channel | 70% online (`2`), 30% store (`1`) |
| Articles by first sale | 2018: 42,144 · 2019: 37,084 · 2020: 25,319, so plenty of "new launches" to test cold-start |
| Sales per article | median 65, top 10% above 793; **72,962 articles have ≥ 20 sales** |

### Files
- `articles.csv` (25 columns): `article_id, product_code, prod_name, product_type_name` (131 types), `product_group_name` (19), `graphical_appearance_name` (30: Solid, All over pattern, Melange, Stripe, Denim, …), `colour_group_name` (50), `perceived_colour_master_name` (20), `department_name` (250), `index_group_name` (Ladieswear, Baby/Children, Divided, Menswear, Sport), `section_name` (56), `garment_group_name` (21), `detail_desc` (text, e.g. *"Jersey top with narrow shoulder straps."*, missing for 416 articles).
- `transactions_train.csv`: `t_dat, customer_id, article_id, price, sales_channel_id`. One row = one item bought.
- `customers.csv`: `customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code`.
- `images/0xx/<article_id>.jpg`: one photo per article (a few are missing).

### Quirks
- **Price is normalized** (median 0.025); it's comparable between items but isn't currency.
- Colour variants share a `product_code`, so a random split would leak near-identical images into the test set. We split by `product_code` and by time.

### When we'd use it
If Visuelle results need a scale check, or if something blocks Visuelle.

---

## Interview cheat sheet

**"Why not just use one big dataset?"**
No public dataset has all three: luxury products, clean labels **and** sales. Each source covers
one piece. Myntra has labels, Farfetch has luxury prices, Visuelle has sales with images, and
LookBench is an unbiased test.

**"Is any of this real luxury sales data?"**
No, and I say so. Luxury houses don't publish sales. Farfetch gives real luxury *catalog and
pricing* data; demand comes from Visuelle (fast fashion). The method (borrowing demand from
visually similar past products) transfers; the numbers don't.

**"Visuelle has only 5k products; isn't that small?"**
It's small in products but rich in detail: 106k product-store series, weekly sales, prices,
discounts, stock and trends. It's also the standard benchmark for this exact task, so results
compare with published papers. H&M is the scale check.

**"How did you avoid leakage?"**
Colour variants never cross splits; demand is split by season (the official split); Google Trends
uses only the 52 weeks before launch; and customer history is cut off before the test season.

**"Why these embedding models?"**
Chosen by numbers, not name: GR-Lite leads LookBench among open models (65.7% vs 31.9% for generic
CLIP). Because the benchmark's authors also publish GR-Lite, I re-ran every model myself.

**"Any licensing issues?"**
Myntra is MIT, LookBench is Apache-2.0, Visuelle is CC BY-NC-SA (non-commercial, cited), and H&M is
under Kaggle competition rules. Farfetch is scraped, so it stays out of the public demo.

**"Farfetch prices are in which currency?"**
Singapore dollars: the scrape came from Farfetch's Singapore site. I convert at the July 2019 rate when I show USD.

---

## Sources
- Visuelle 2.0: https://humaticslab.github.io/forecasting/visuelle · https://arxiv.org/abs/2204.06972 · https://github.com/HumaticsLAB/visuelle2.0-code
- LookBench: https://huggingface.co/datasets/srpone/look-bench · https://arxiv.org/abs/2601.14706
- Fashion Product Images: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- Farfetch Listings: https://www.kaggle.com/datasets/alvations/farfetch-listings · https://github.com/zpencerguy/fmarket
- H&M: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
