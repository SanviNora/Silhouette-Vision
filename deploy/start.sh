#!/usr/bin/env bash
# Download the data bundle (cached across restarts when persistent storage is enabled), then serve.
set -euo pipefail
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["DATA_REPO"], repo_type="dataset", local_dir=os.environ["SILHOUETTE_DATA_ROOT"])
PY
exec streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
