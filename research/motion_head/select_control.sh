#!/usr/bin/env bash
# Controls for the medoid diagnostic: (1) default selection with another seed on the
# 22 FN-win episodes (run-to-run noise); (2) medoid selection on all 100 clean episodes.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench; D=outputs/motion_head/drive
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
run() {  # name ids select seed
    HEAD_SELECT=$3 ~/miniforge3/envs/futurenav/bin/python deploy/head_server.py --checkpoint pretrain_models/FutureNav-4B-Base \
        --head outputs/motion_head/flow_v1/best.pt --port 8643 --seed $4 > $D/server_$1.log 2>&1 &
    local S=$!
    until curl -sf http://127.0.0.1:8643/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$2 EXP_SAVE=trace \
        TAG=$1 bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:8643 > $D/$1.log 2>&1
    kill $S; wait $S 2>/dev/null
    echo "$1 done $(date): $(grep -o 'SR=.*' $D/$1.log | tail -1)"
}
run flow_v1_seed1_fnwin   $B/fn_win_clean.txt consistent 1
run flow_v1_medoid_trace100 $B/holdout_ids.txt medoid 0
