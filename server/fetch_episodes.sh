#!/usr/bin/env bash
# Downloads the R2R and RxR VLN-CE episode files. These are public.
# The Matterport3D *scenes* are NOT public and cannot be scripted -- see README.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"
mkdir -p data/datasets

# Self-contained: works from any env, including a bare conda base.
if ! python3 -c "import gdown" >/dev/null 2>&1; then
    echo "installing gdown into the active interpreter ..."
    python3 -m pip install --quiet gdown
fi
gd() { python3 -m gdown "$@"; }

if [ ! -d data/datasets/R2R_VLNCE_v1-3_preprocessed ]; then
    gd "https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr" -O /tmp/r2r.zip
    unzip -q /tmp/r2r.zip -d data/datasets && rm /tmp/r2r.zip
fi
if [ ! -d data/datasets/RxR_VLNCE_v0 ]; then
    gd "https://drive.google.com/uc?id=145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t" -O /tmp/rxr.zip
    unzip -q /tmp/rxr.zip -d data/datasets && rm /tmp/rxr.zip
fi
find data/datasets -maxdepth 2 -type d | sort
