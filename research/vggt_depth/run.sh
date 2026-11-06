#!/usr/bin/env bash
# VGGT depth-quality check: collect expert frames + GT depth in Habitat, then replay them through FutureNav's
# streaming VGGT and compare depth_head output with GT.
#   [CAM_H=0.75] bash experiments/vggt_depth/run.sh [WAIT_PID]   # optionally wait for another GPU job (by PID) to exit first
# Needs ~12 GB free GPU memory for the evaluation (VGGT-1B + 56-frame KV cache).
set -uo pipefail
cd "$(dirname "$0")/../.."
CAM_H=${CAM_H:-1.25}
D=outputs/vggt_depth; [ "$CAM_H" != 1.25 ] && D=outputs/vggt_depth/h$CAM_H; mkdir -p $D
if [ -n "${1:-}" ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 8000 ]; do sleep 60; done
echo "[$(date '+%F %T')] collecting"
PATH=$HOME/miniforge3/envs/habitat/bin:$PATH MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
    env -u DISPLAY python experiments/vggt_depth/collect.py --cam-height $CAM_H --out $D/frames > $D/collect.log 2>&1 || { echo "collect failed, see $D/collect.log"; exit 1; }
grep "^episode\|episodes from" $D/collect.log
echo "[$(date '+%F %T')] evaluating"
PYTHONPATH=$PWD/src:$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniforge3/envs/futurenav/bin/python experiments/vggt_depth/eval_depth.py --cam-height $CAM_H --frames $D/frames --out $D/results.json > $D/eval.log 2>&1 || { echo "eval failed, see $D/eval.log"; exit 1; }
sed -n '/^{/,$p' $D/eval.log
echo "[$(date '+%F %T')] done"
