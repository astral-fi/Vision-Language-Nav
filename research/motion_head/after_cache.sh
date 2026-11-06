#!/usr/bin/env bash
# Wait for cache_s2.py (by PID), then train the flow head and the regression
# baseline in parallel on the full cache. Logs: outputs/motion_head/logs/.
set -uo pipefail
cd "$(dirname "$0")/../.."
PID=${1:?cache_s2 pid}
L=outputs/motion_head/logs; mkdir -p $L
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
N=$(find outputs/data_engine/s2_v1 -name '*.npz' | wc -l)
echo "cache done: $N episodes, $(date)"
if [ "$N" -lt 1400 ]; then echo "cache incomplete, not training"; exit 1; fi
PY=~/miniforge3/envs/futurenav/bin/python
$PY experiments/motion_head/train.py --mode flow --tag flow_v1 > $L/flow_v1.log 2>&1 &
$PY experiments/motion_head/train.py --mode reg --tag reg_v1 > $L/reg_v1.log 2>&1 &
wait
echo "training done: $(date)"
