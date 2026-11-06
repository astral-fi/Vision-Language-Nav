#!/usr/bin/env bash
# Detached Stage-1 pipeline for the LoRA plan (runs on the GPU box, survives the
# client disconnecting). Launch:  setsid nohup bash experiments/s2_lora/run_pipeline.sh &
#   1. wait for any running select_control.sh (the GPU must be free)
#   2. parity test (gate: >= 95% first-action agreement) -> stop if it fails
#   3. precompute streaming VGGT features: val, then train
# Log: outputs/s2_lora/logs/pipeline.log (+ one log per stage).
set -uo pipefail
cd "$(dirname "$0")"
L=../../outputs/s2_lora/logs; mkdir -p $L
PY=~/miniforge3/envs/futurenav/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
log() { echo "[$(date '+%F %T')] $*" | tee -a $L/pipeline.log; }

while pgrep -f "bash experiments/motion_head/select_control.sh" >/dev/null; do sleep 30; done
log "GPU free; parity test"
$PY parity.py --episodes 20 --check-capture > $L/parity.log 2>&1 || { log "parity crashed (see parity.log)"; exit 1; }
RES=$(grep '^PARITY' $L/parity.log); log "$RES"; grep 'capture check' $L/parity.log | tee -a $L/pipeline.log
OK=$($PY -c "import re,sys;m=re.search(r'= ([0-9.]+) first','$RES');print(int(m and float(m.group(1))>=0.95))")
if [ "$OK" != 1 ]; then log "parity gate FAILED; stopping before the precompute"; exit 1; fi
for S in val train; do
    log "precompute $S"
    $PY precompute_vggt.py --split $S > $L/precompute_$S.log 2>&1 || { log "precompute $S crashed"; exit 1; }
    log "precompute $S done: $(ls ../../outputs/s2_lora/vggt | wc -l) files, $(du -sh ../../outputs/s2_lora/vggt | cut -f1)"
done
log "stage 1 complete"
