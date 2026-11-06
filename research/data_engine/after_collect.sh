#!/usr/bin/env bash
# Wait for all 6 collect.py shards (by their logs, not pgrep, which matches its own
# command line), then cache System 2 features. Aborts if a shard crashed.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=outputs/data_engine/logs
until [ "$(grep -l '^done shard' $L/collect_*.log 2>/dev/null | wc -l)" -ge 6 ]; do
    if grep -l Traceback $L/collect_*.log >/dev/null 2>&1; then echo "collect crashed: $(grep -l Traceback $L/collect_*.log)"; exit 1; fi
    sleep 60
done
echo "collection done: $(ls outputs/data_engine/expert_v1/*/*.npz | wc -l) episodes, $(date)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exec ~/miniforge3/envs/futurenav/bin/python \
    experiments/data_engine/cache_s2.py --max "${MAX:-1500}"
