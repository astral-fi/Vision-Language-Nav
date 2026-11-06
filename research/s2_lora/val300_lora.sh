#!/usr/bin/env bash
# PLAN step 10: LoRA mode C on val300 (val_unseen, buildings neither model trained on).
# Waits for bench_lora.sh. FutureNav reference: base500 on the same 300 ids = SR 0.547 SPL 0.490.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench
D=outputs/s2_lora/drive
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
while pgrep -f bench_lora.sh >/dev/null; do sleep 60; done
$PY deploy/chunk_server.py --checkpoint pretrain_models/FutureNav-4B-Base --adapter outputs/s2_lora/lora_v1/adapter_final \
    --mode C --port 8644 > $D/server_lora_v1_C_val300.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8644/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=val_unseen EPISODE_IDS=$B/val300_ids.txt EXP_SAVE=trace \
    TAG=lora_v1_C_val300 bash eval/eval_r2r_remote.sh val_unseen 0 http://127.0.0.1:8644 > $D/lora_v1_C_val300.log 2>&1
kill $S; wait $S 2>/dev/null
echo "[$(date '+%F %T')] lora_v1_C_val300 done: $(grep -o 'SR=.*' $D/lora_v1_C_val300.log | tail -1)" >> $D/bench.log
