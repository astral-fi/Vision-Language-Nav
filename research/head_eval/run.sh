#!/usr/bin/env bash
# Closed-loop val_unseen eval of the two small heads on the stratified 400-episode stair-free subset (select400.py).
#   bash experiments/head_eval/run.sh smoke        # 5 episodes per arm
#   bash experiments/head_eval/run.sh full         # 400 stair-free episodes per arm, one server at a time
# Results: outputs/eval/r2r/<tag>/val_unseen/, logs: outputs/head_eval/. Env: ARMS="discrete lora_waypoint".
set -uo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-smoke}
D=outputs/head_eval; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
IDS=experiments/head_eval/ids_sf400.txt; SUF=sf400
if [ "$MODE" = smoke ]; then head -5 $IDS > $D/ids_smoke.txt; IDS=$D/ids_smoke.txt; SUF=smoke5; fi

port() { case $1 in discrete) echo 8651;; lora_waypoint) echo 8652;; esac; }
for ARM in ${ARMS:-discrete lora_waypoint}; do
    PORT=$(port $ARM); NAME=${ARM}_$SUF
    $PY deploy/head_eval_server.py --arm $ARM --port $PORT > $D/server_$NAME.log 2>&1 &
    S=$!
    until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
    if ! kill -0 $S 2>/dev/null; then echo "$NAME: server died, see $D/server_$NAME.log"; continue; fi
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=val_unseen EPISODE_IDS=$IDS EXP_SAVE=trace TAG=$NAME \
        bash eval/eval_r2r_remote.sh val_unseen 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
    curl -s http://127.0.0.1:$PORT/health > $D/health_$NAME.json
    kill $S; wait $S 2>/dev/null
    echo "[$(date '+%F %T')] $NAME done: $(grep -o "SR=.*" $D/$NAME.log | tail -1)" | tee -a $D/summary.log
done
