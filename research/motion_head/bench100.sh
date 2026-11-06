#!/usr/bin/env bash
# Full holdout benchmark (bench.py): flow head and FutureNav on the 100 holdout
# episodes, clean and with one scripted perturbation each. One server at a time
# (each needs the GPU alone). Results: outputs/eval/r2r/<tag>/train/.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench
D=outputs/motion_head/drive; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

run() {   # name port derail(0/1) server args...
    local NAME=$1 PORT=$2 DER=$3; shift 3
    $PY "$@" --port $PORT > $D/server_$NAME.log 2>&1 &
    local S=$!
    until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
    local SPEC=""; [ "$DER" = 1 ] && SPEC=$B/holdout_derail.json
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$B/holdout_ids.txt \
        DERAIL_SPEC=$SPEC TAG=$NAME bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
    kill $S; wait $S 2>/dev/null
    echo "$NAME done $(date): $(grep -o 'SR=.*' $D/$NAME.log | tail -1)"
}
HEAD="deploy/head_server.py --checkpoint pretrain_models/FutureNav-4B-Base --head outputs/motion_head/flow_v1/best.pt"
FN="deploy/server.py --checkpoint pretrain_models/FutureNav-4B-Base --attn sdpa"
[ "${SKIP_FIRST:-0}" = 1 ] || run flow_v1_holdout100   8643 0 $HEAD
run futurenav_holdout100 8642 0 $FN
run flow_v1_derail100    8643 1 $HEAD
run futurenav_derail100  8642 1 $FN
