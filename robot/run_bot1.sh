#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ "$#" -eq 0 ]; then
  echo 'Usage: ./run_bot1.sh "navigation instruction" [client options]' >&2
  exit 2
fi
instruction="$1"
shift
export PYTHONDONTWRITEBYTECODE=1
exec flock -n -E 75 "$PWD/.inference-run.lock" \
  "${ROBOT_PYTHON:-.venv/bin/python}" -u client.py --robot jetracer --namespace bot1 \
  --scan-topic - --no-yaw-feedback --turn-radius 0.31 \
  --router "${ZENOH_ROUTER:-127.0.0.1:7447}" --server http://127.0.0.1:8642 \
  --futurenav-root "${FUTURENAV_ROOT:-$HOME/FutureNav}" --max-steps 100 --max-duration 600 --instruction "$instruction" "$@"
