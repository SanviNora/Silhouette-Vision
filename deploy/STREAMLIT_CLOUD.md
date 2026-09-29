# Deploying on Streamlit Community Cloud (free)

The app runs from this GitHub repo; its data (Myntra + Visuelle thumbnails, embeddings, trained
heads) is downloaded on first start from the public Hugging Face dataset
`Sanvii/silhouette-vision-data` (built by `scripts/10_build_public.py`, uploaded by
`scripts/11_publish.py`).

1. Go to https://share.streamlit.io and sign in with GitHub (the SanviNora account).
2. **Create app** → "Deploy a public app from GitHub":
   - Repository: `SanviNora/Silhouette-Vision`, branch `main`
   - Main file path: `app/ui.py`
   - App URL: e.g. `silhouette-vision`
3. **Advanced settings**:
   - Python version: **3.13** (`app/requirements.txt` pins CPU PyTorch wheels built for 3.13)
   - Secrets:
     ```toml
     SILHOUETTE_PUBLIC = "1"
     SILHOUETTE_DEVICE = "cpu"
     SILHOUETTE_PRECISE = "0"
     SILHOUETTE_DATA_ROOT = "/tmp/silhouette_bundle"
     DATA_REPO = "Sanvii/silhouette-vision-data"
     ```
4. **Deploy.** The first build installs dependencies (~5–10 min); the first visit downloads the
   data (~1 min) and the Marqo model.

Notes
- Memory: ~2.2 GB with everything but "Precise match" (GR-Lite, +1.7 GB), which is switched off
  by `SILHOUETTE_PRECISE = "0"`.
- Free apps sleep after inactivity; the first visit afterwards takes a minute to wake.
- `SILHOUETTE_PUBLIC = "1"` keeps the Farfetch catalog out (its license forbids redistribution).
