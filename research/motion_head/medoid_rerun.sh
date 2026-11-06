#!/usr/bin/env bash
# Diagnostic: flow head without plan consistency or blending (HEAD_SELECT=medoid)
# on the 22 clean holdout episodes FutureNav wins and the head loses.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$PWD/outputs/motion_head/bench; D=outputs/motion_head/drive
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HEAD_SELECT=medoid
~/miniforge3/envs/futurenav/bin/python deploy/head_server.py --checkpoint pretrain_models/FutureNav-4B-Base \
    --head outputs/motion_head/flow_v1/best.pt --port 8643 > $D/server_flow_v1_medoid_fnwin.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:8643/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$B/fn_win_clean.txt EXP_SAVE=trace \
    TAG=flow_v1_medoid_fnwin bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:8643 > $D/flow_v1_medoid_fnwin.log 2>&1
kill $S; wait $S 2>/dev/null
echo "done $(date): $(grep -o 'SR=.*' $D/flow_v1_medoid_fnwin.log | tail -1)"
