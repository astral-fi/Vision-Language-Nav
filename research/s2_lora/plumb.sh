#!/usr/bin/env bash
# Plumbing test for deploy/chunk_server.py: mode C on 2 perturbed holdout episodes, traces on.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench
D=outputs/s2_lora/drive
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
$PY deploy/chunk_server.py --checkpoint pretrain_models/FutureNav-4B-Base \
    --adapter outputs/s2_lora/lora_v1/adapter_final --mode C --port 8644 > $D/server_plumb.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8644/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$D/plumb_ids.txt EXP_SAVE=trace \
    DERAIL_SPEC=$B/holdout_derail.json TAG=lora_v1_plumb bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:8644 > $D/plumb.log 2>&1
curl -s http://127.0.0.1:8644/health >> $D/plumb.log
kill $S; wait $S 2>/dev/null
echo "plumb done $(date)" >> $D/plumb.log
