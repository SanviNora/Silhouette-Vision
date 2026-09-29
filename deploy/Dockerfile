# Hugging Face Space (Docker SDK). Code comes from this repo; data (Myntra + Visuelle thumbnails,
# embeddings, trained heads) is downloaded at start-up from the dataset repo in DATA_REPO.
FROM python:3.14-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    SILHOUETTE_PUBLIC=1 SILHOUETTE_DEVICE=cpu \
    SILHOUETTE_DATA_ROOT=/home/user/bundle \
    DATA_REPO=SanviNora/silhouette-vision-data
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "start.sh"]
