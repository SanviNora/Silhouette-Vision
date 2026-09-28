# Silhouette Vision — Interview Notes

A running log of the decisions we make, why we made them, and the questions an interviewer is
likely to ask about each. It's updated as the project is built; once the build is finished it
becomes the basis for mock-interview practice.

Data-specific Q&A lives in [data_guide.md](data_guide.md#interview-cheat-sheet).

---

## Decision log

| Date | Decision | Why | Alternatives rejected |
|---|---|---|---|
| 2026-09-27 | Four core datasets (Myntra, Farfetch, LookBench, Visuelle 2.0) + H&M backup | No single public dataset has luxury products, clean labels **and** sales | Fashionpedia (street photos, not product shots), Branded Bottoms (5k trousers only), Vestiaire (removed from Kaggle) |
| 2026-09-27 | Full Myntra images, not "small" | The small version is 60×80 px; the models expect 224–384 px | — |
| 2026-09-27 | Use Myntra's style JSONs, not just `styles.csv` | They add price, brand and structured attributes (Pattern, Fabric, Neck, Sleeve, Fit, Material), which serve as labels for explanations | Buying or adding another attribute dataset |
| 2026-09-27 | Farfetch categories from title keywords (`configs/taxonomy.yml`) | Farfetch has no category column; ordered rules handle cases like "shirt dress" → dress | Zero-shot image classification for everything (kept for the leftovers) |
| 2026-09-27 | Load GR-Lite weights into `transformers`' DINOv3 class with a strict key check | Its own loader breaks on transformers 5; a strict load proves every weight is used | Pinning transformers 4.x for the whole project |
| 2026-09-27 | Test GR-Lite with and without RoPE on LookBench | The published reference code omits DINOv3's RoPE; RoPE has no learned weights, so a missing RoPE wouldn't show up as a missing key | Trusting the reference code blindly |
| 2026-09-28 | **Phase 1 runs on Marqo-FashionSigLIP only**; GR-Lite must win on LookBench (Phase 2) before we spend compute on it | Measured on an M4: Marqo 62 img/s vs GR-Lite 7 img/s (~1 h vs ~9 h for 229k images); on CPU 21 vs 3 img/s (~50 ms vs ~350 ms per query). Marqo covers image search, text search, zero-shot attributes and explanations; GR-Lite only image→image, for ~3 points of LookBench Recall@1 | Running GR-Lite overnight; Colab now (kept as the Phase 2 fallback) |
| 2026-09-28 | Store embeddings as float16 on disk, cast to float32 for search | Halves the size (229k × 768 → ~350 MB); the dot-product ranking is essentially unchanged | float32 on disk |
| 2026-09-28 | One DataLoader across all chunks, saving results chunk by chunk | macOS starts DataLoader workers by re-importing torch; restarting them per chunk halved throughput (30 → 55 img/s) | Per-chunk loaders |
| 2026-09-28 | **Dropped FAISS; exact search is one NumPy dot product** | FAISS's bundled OpenMP runtime crashed alongside PyTorch's on macOS (the suggested env-var workaround is documented as able to "silently produce incorrect results"). At 230k × 768 an exact search is tens of ms, so an index buys nothing | `KMP_DUPLICATE_LIB_OK=TRUE`; relinking FAISS to torch's libomp |
| 2026-09-28 | **Public app shows Myntra (MIT) + Visuelle (CC BY-NC-SA, credited); Farfetch appears as analysis + images loaded from Farfetch's own URLs; everything runs locally** | Showing images on a public site is redistribution. Farfetch images are scraped (no license), so we don't re-host them; H&M's rules forbid redistribution | Re-hosting all images publicly; dropping Farfetch from the public version entirely |
| 2026-09-27 | Qualitative test on 11 real-world photos before any metrics | Catches failure modes metrics hide (colour drift, multi-item photos); gives demo material. Results in `phase1_results.md` | Going straight to LookBench numbers |

## Likely questions (answers to be filled in with our own results)

**Models**
- *What's the difference between a CLIP/SigLIP-style model and a DINO-style model?*
  CLIP/SigLIP learn from image–caption pairs, so images and text share one space (text search, zero-shot labels). DINO learns from images alone with self-supervision, and is often better at fine visual detail but can't read text.
- *Why two models instead of one?* → fill in with the LookBench results.
- *How did you choose between them?* → LookBench Recall@1, speed (images/s on an M4), and whether text search is needed.
- *SigLIP vs CLIP?* SigLIP uses a sigmoid loss on each image–text pair instead of a softmax over the whole batch. It trains well at smaller batch sizes and is usually stronger at the same size.

**Engineering**
- *How did you make embedding 229k images practical on a laptop?* 512 px thumbnails made once (Myntra originals are up to 1800×2400), resumable chunked embedding, a single long-lived DataLoader, and a small model. Also: the first attempt ran at 4–24 img/s because the Mac was swapping 18.8 GB. **Always check memory and swap before blaming the model.**
- *Why not just use the most accurate model?* Accuracy per unit of cost: GR-Lite is ~3 Recall@1 points better on LookBench but ~8× slower and image-only. For a CPU-hosted demo that also needs text search, the smaller model wins, unless our own benchmark shows a bigger gap.
- *Why is cosine similarity a dot product in your index?* The embeddings are L2-normalized, so the dot product equals cosine similarity.
- *Exact vs approximate nearest-neighbour search?* At about 230k vectors an exact search (one matrix-vector product) is fast enough, so we don't use an index at all. Approximate methods (HNSW, IVF-PQ in FAISS or a vector database) pay off at millions of vectors, trading a little recall for speed and memory.
- *Why no FAISS?* It crashed next to PyTorch on macOS (two OpenMP runtimes in one process). Rather than use an "unsafe" workaround, I measured and found we didn't need it. **Measure before adding infrastructure.**
- *How do filters work without hurting results?* They're applied before ranking (scores of excluded items set to −∞), so "bags only" still returns k bags. Filtering *after* a top-k search can return too few results.

**Evaluation**
- *What did your first tests show?* 11 real photos: logo/pattern and silhouette are captured very strongly (8/8 Louis Vuitton monogram bags for an LV query; 8/8 wide-leg jeans). Weaknesses: subtle colour (burgundy→red), busy runway shots, multi-item photos. Similarity scores fall from ~0.83 to ~0.72 on the hard cases, so they work as a confidence signal.
- *How would you fix the weaknesses?* Colour: text refinement (partial fix, tested) or colour-aware re-ranking. Multi-item: detect and crop each garment, then search per item; a category filter already fixes it manually (tested).
- *Why does the model get the logo but miss the colour?* Contrastive training on product captions rewards brand/pattern/shape words that appear in titles; colour is often a single word and less emphasized. Lighting and backgrounds (a burgundy shoe on a patterned sofa) also shift perceived colour.
- *How do you know the search is good?* → LookBench Recall@K against the public leaderboard; precision@K on Myntra labels.
- *What is Recall@1 here?* The share of query photos whose top result is the exact same product.

**Data**
- See the data guide's cheat sheet: leakage, licensing, SGD prices, and Visuelle's size.

**Deployment and licensing**
- *Why doesn't your public demo show the Farfetch catalog directly?* The images were scraped and have no license. Hosting them publicly would be redistributing copyrighted photos. The public app uses properly licensed galleries (Myntra MIT, Visuelle CC BY-NC-SA with credit), shows Farfetch as analysis, and loads Farfetch images from Farfetch's own URLs rather than copying them.
- *What does CC BY-NC-SA require?* Credit the authors (BY), non-commercial use only (NC), and share derivatives under the same license (SA).
