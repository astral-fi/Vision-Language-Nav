#!/usr/bin/env bash
# Queue, one GPU job at a time (each model server needs the GPU alone). Resumable: eval/run.py skips
# episodes that already have log/stats_<id>.json, so re-issuing this continues an interrupted run.
#   1. both small heads on the 400 stair-free subset (ids_sf400.txt)       -> <arm>_sf400
#   2. LoRA (mode C, adapter_final) on the subset episodes not in lora_v1_C_val300 (ids_lora_rest.txt, 321)
# FutureNav needs no new run: base500 + capture_pool already cover all 1,407 stair-free episodes.
# Merge with experiments/head_eval/merge.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=outputs/head_eval; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
[ -f experiments/head_eval/ids_lora_rest.txt ] || $PY experiments/head_eval/build_lora_rest.py

bash experiments/head_eval/run.sh full

NAME=lora_v1_C_sf400rest; PORT=8644
$PY deploy/chunk_server.py --checkpoint pretrain_models/FutureNav-4B-Base --adapter outputs/s2_lora/lora_v1/adapter_final \
    --mode C --port $PORT > $D/server_$NAME.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
if kill -0 $S 2>/dev/null; then
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=val_unseen EPISODE_IDS=experiments/head_eval/ids_lora_rest.txt \
        EXP_SAVE=trace TAG=$NAME bash eval/eval_r2r_remote.sh val_unseen 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
    kill $S; wait $S 2>/dev/null
    echo "[$(date '+%F %T')] $NAME done: $(grep -o 'SR=.*' $D/$NAME.log | tail -1)" | tee -a $D/summary.log
else
    echo "$NAME: server died, see $D/server_$NAME.log" | tee -a $D/summary.log
fi
echo "[$(date '+%F %T')] ALL DONE" | tee -a $D/summary.log
