#!/usr/bin/env bash
# Takes over from run_pipeline.sh once its parity gate has passed, so a training
# smoke test runs before the 8-hour precompute. Detached:
#   setsid nohup bash experiments/s2_lora/run_stage2.sh &
#   1. wait for PARITY in pipeline.log; stop if < 95%; stop run_pipeline.sh
#   2. precompute val -> smoke test (train on val features, 64 samples)
#   3. precompute train -> full training (tag lora_v1)
set -uo pipefail
cd "$(dirname "$0")"
L=../../outputs/s2_lora/logs
PY=~/miniforge3/envs/futurenav/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
log() { echo "[$(date '+%F %T')] $*" | tee -a $L/pipeline.log; }

until grep -q '^PARITY' $L/parity.log 2>/dev/null; do
    pgrep -f "s2_lora/run_pipeline.sh" >/dev/null || { log "stage2: pipeline exited before parity"; exit 1; }
    sleep 20
done
sleep 5
RES=$(grep '^PARITY' $L/parity.log)
OK=$($PY -c "import re;m=re.search(r'= ([0-9.]+) first','$RES');print(int(m and float(m.group(1))>=0.95))")
pkill -f "s2_lora/run_pipeline.sh"; sleep 2; pkill -f "precompute_vggt.py"; sleep 5
[ "$OK" = 1 ] || { log "stage2: parity gate failed ($RES)"; exit 1; }
log "stage2: took over after parity"

log "precompute val"
$PY precompute_vggt.py --split val > $L/precompute_val.log 2>&1 || { log "precompute val crashed"; exit 1; }
log "smoke test"
$PY train_lora.py --tag smoke --train-split val --max-samples 64 --accum 4 --eval-every 8 --save-every 1000 \
    --val-samples 16 > $L/smoke.log 2>&1 || { log "smoke test crashed (see smoke.log)"; exit 1; }
log "smoke: $(grep -c '"step"' $L/smoke.log) steps; last: $(grep '"step"' $L/smoke.log | tail -1)"
log "precompute train"
$PY precompute_vggt.py --split train > $L/precompute_train.log 2>&1 || { log "precompute train crashed"; exit 1; }
log "precompute train done: $(ls ../../outputs/s2_lora/vggt | wc -l) files, $(du -sh ../../outputs/s2_lora/vggt | cut -f1)"
log "training lora_v1"
$PY train_lora.py --tag lora_v1 > $L/train_lora_v1.log 2>&1 || { log "training crashed"; exit 1; }
log "training done: $(tail -1 $L/train_lora_v1.log)"
