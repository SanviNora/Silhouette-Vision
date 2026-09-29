#!/usr/bin/env bash
# Download the data bundle (cached across restarts when persistent storage is enabled), then serve.
set -euo pipefail
python -c 'import os; from pathlib import Path; from silhouette_vision.bootstrap import ensure_bundle; ensure_bundle(Path(os.environ["SILHOUETTE_DATA_ROOT"]), os.environ["DATA_REPO"])'
exec streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
