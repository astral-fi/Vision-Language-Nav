#!/usr/bin/env bash
# PLAN step 9: LoRA chunk policy on the 100 holdout episodes (same set and derails as
# experiments/motion_head/bench100.sh), modes A / C / B, clean then perturbed, traces on.
# One server at a time. Results: outputs/eval/r2r/<tag>/train/, logs in outputs/s2_lora/drive/.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench
D=outputs/s2_lora/drive; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
AD=outputs/s2_lora/lora_v1/adapter_final
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

run() {   # name derail(0/1) mode
    local NAME=$1 DER=$2 MODE=$3 PORT=8644
    $PY deploy/chunk_server.py --checkpoint pretrain_models/FutureNav-4B-Base --adapter $AD \
        --mode $MODE --port $PORT > $D/server_$NAME.log 2>&1 &
    local S=$!
    until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
    local SPEC=""; [ "$DER" = 1 ] && SPEC=$B/holdout_derail.json
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$B/holdout_ids.txt EXP_SAVE=trace \
        DERAIL_SPEC=$SPEC TAG=$NAME bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
    curl -s http://127.0.0.1:$PORT/health > $D/$NAME.health.json
    kill $S; wait $S 2>/dev/null
    echo "[$(date '+%F %T')] $NAME done: $(grep -o 'SR=.*' $D/$NAME.log | tail -1)" >> $D/bench.log
}
echo "[$(date '+%F %T')] start" >> $D/bench.log
run lora_v1_A_clean  0 A
run lora_v1_C_clean  0 C
run lora_v1_B_clean  0 B
run lora_v1_A_derail 1 A
run lora_v1_C_derail 1 C
run lora_v1_B_derail 1 B
echo "[$(date '+%F %T')] all done" >> $D/bench.log
