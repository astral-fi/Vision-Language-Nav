#!/usr/bin/env bash
# Dual system on the val300 benchmark: System 2 = LoRA waypoint head (head_eval_server.py --exec plan, queried only
# on request), System 1 = eval/s1_local.py (navmesh-LiDAR local planner + event triggers, run.py --s1).
# Compare with lora_waypoint_cont_val300 (same head, continuous executor, re-query every <=4 poses / 30 deg).
#   bash experiments/head_eval/s1.sh smoke              # 2 episodes
#   bash experiments/head_eval/s1.sh full               # 300 episodes
#   S1_CONFIG='{"max_steps": 24}' TAGSUF=_ms24 bash experiments/head_eval/s1.sh full
set -uo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-smoke}
D=outputs/head_eval; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
IDS=outputs/motion_head/bench/val300_ids.txt
NAME=lora_waypoint_s1_val300${TAGSUF:-}
if [ "$MODE" = smoke ]; then head -2 $IDS > $D/ids_s1_smoke.txt; IDS=$D/ids_s1_smoke.txt; NAME=lora_waypoint_s1_smoke${TAGSUF:-}; fi
PORT=8655
$PY deploy/head_eval_server.py --arm lora_waypoint --exec plan --port $PORT > $D/server_$NAME.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
if ! kill -0 $S 2>/dev/null; then echo "$NAME: server died, see $D/server_$NAME.log"; exit 1; fi
PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=val_unseen EPISODE_IDS=$IDS EXP_SAVE=trace,data TAG=$NAME \
    WAYPOINT_EXEC=1 S1=1 S1_CONFIG=${S1_CONFIG:-} bash eval/eval_r2r_remote.sh val_unseen 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
curl -s http://127.0.0.1:$PORT/health > $D/health_$NAME.json
kill $S; wait $S 2>/dev/null
echo "[$(date '+%F %T')] $NAME done: $(grep -A7 '^episodes' $D/$NAME.log | tr '\n' ' ')" | tee -a $D/summary.log
