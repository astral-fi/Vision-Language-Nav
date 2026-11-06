#!/usr/bin/env bash
# Re-run with per-step traces (EXP_SAVE=trace): flow head on all 100 clean holdout
# episodes, FutureNav on the episodes the head failed. For the failure analysis.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench
D=outputs/motion_head/drive
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
run() {   # name port ids server args...
    local NAME=$1 PORT=$2 IDS=$3; shift 3
    $PY "$@" --port $PORT > $D/server_$NAME.log 2>&1 &
    local S=$!
    until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$IDS EXP_SAVE=trace \
        TAG=$NAME bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
    kill $S; wait $S 2>/dev/null
    echo "$NAME done $(date): $(grep -o 'SR=.*' $D/$NAME.log | tail -1)"
}
run flow_v1_trace100     8643 $B/holdout_ids.txt \
    deploy/head_server.py --checkpoint pretrain_models/FutureNav-4B-Base --head outputs/motion_head/flow_v1/best.pt
run futurenav_trace_fail 8642 $B/head_fail_clean.txt \
    deploy/server.py --checkpoint pretrain_models/FutureNav-4B-Base --attn sdpa
