# Deploying on Streamlit Community Cloud (free)

Live: https://silhouette-vision.streamlit.app

The app runs from this GitHub repo (`main`); its data (catalog, thumbnails, embeddings, trained
heads, demand model) is downloaded on first start from the Hugging Face dataset
`Sanvii/silhouette-vision-data` (built by `scripts/10_build_bundle.py`, uploaded by
`scripts/11_publish.py`).

1. https://share.streamlit.io → sign in with GitHub → **Create app** → deploy from GitHub:
   repository `SanviNora/Silhouette-Vision`, branch `main`, main file `app/ui.py`.
2. No settings needed. With no local data the app detects a fresh host
   (`silhouette_vision/config.py`), downloads the bundle into a temp folder and unpacks it.
   Any Python 3.12–3.14 works (`app/requirements.txt` picks the matching CPU PyTorch wheel).
   Optional secret: `DATA_REPO` to use another dataset repo.
3. **Deploy.** First build ~5–10 min; first visit downloads ~1.1 GB (~1–2 min).

Updating data: rebuild the bundle, bump `BUNDLE_VERSION` in `silhouette_vision/bootstrap.py`,
run `scripts/11_publish.py`, push. Running hosts see the new version and download it again.

Notes
- Memory: ~2 GB (Marqo-FashionSigLIP, YOLOS detector, 51k embeddings, demand model).
- Free apps sleep after inactivity; the first visit afterwards takes a minute to wake.
