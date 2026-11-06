#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ "$#" -eq 0 ]; then
  echo 'Usage: bash run_waffle.sh "navigation instruction" [client options]' >&2
  exit 2
fi
instruction="$1"
shift
export PYTHONDONTWRITEBYTECODE=1
# The profile selects tb3/flir_camera/image_raw/compressed and FLIR calibration.
exec flock -n -E 75 "$PWD/.inference-run.lock" \
  "${ROBOT_PYTHON:-.venv/bin/python}" -u client.py --robot waffle --namespace tb3 \
  --router "${ZENOH_ROUTER:-127.0.0.1:7447}" --server http://127.0.0.1:8642 \
  --futurenav-root "${FUTURENAV_ROOT:-$HOME/FutureNav}" --max-steps 100 --max-duration 600 --instruction "$instruction" "$@"
