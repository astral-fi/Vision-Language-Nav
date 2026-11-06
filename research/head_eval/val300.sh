#!/usr/bin/env bash
# LoRA waypoint head on the 300-episode val_unseen benchmark (outputs/motion_head/bench/val300_ids.txt; the same
# ids as lora_v1_C_val300 and the base500 FutureNav reference), two executors:
#   discrete    pose intervals -> nearest primitive (as in sf400). Only the val300 ids not already run in
#               lora_waypoint_sf400 are run here (runs are deterministic); merge_val300.py joins them.
#   continuous  each pose interval executed as is (head_eval_server.py --exec continuous + run.py --waypoint-exec)
#   bash experiments/head_eval/val300.sh smoke     # 2 episodes, continuous
#   bash experiments/head_eval/val300.sh full      # ARMS="continuous discrete" by default, one server at a time
set -uo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-smoke}
D=outputs/head_eval; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
V300=outputs/motion_head/bench/val300_ids.txt
REST=experiments/head_eval/ids_val300_not_sf400.txt
grep -vxFf experiments/head_eval/ids_sf400.txt $V300 > $REST
echo "val300: $(wc -l < $V300) ids, $(wc -l < $REST) not in sf400"

run_arm() {  # exec ids tag
    local EXEC=$1 IDS=$2 NAME=$3 PORT=$([ $1 = continuous ] && echo 8653 || echo 8654)
    $PY deploy/head_eval_server.py --arm lora_waypoint --exec $EXEC --port $PORT > $D/server_$NAME.log 2>&1 &
    local S=$!
    until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
    if ! kill -0 $S 2>/dev/null; then echo "$NAME: server died, see $D/server_$NAME.log"; return 1; fi
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=val_unseen EPISODE_IDS=$IDS EXP_SAVE=trace,data TAG=$NAME \
        WAYPOINT_EXEC=$([ $EXEC = continuous ] && echo 1) bash eval/eval_r2r_remote.sh val_unseen 0 http://127.0.0.1:$PORT \
        > $D/$NAME.log 2>&1
    curl -s http://127.0.0.1:$PORT/health > $D/health_$NAME.json
    kill $S; wait $S 2>/dev/null
    echo "[$(date '+%F %T')] $NAME done: $(grep -o 'SR=.*' $D/$NAME.log | tail -1) $(grep -A1 '^episodes' $D/$NAME.log | tr '\n' ' ')" | tee -a $D/summary.log
}

if [ "$MODE" = smoke ]; then
    head -2 $V300 > $D/ids_val300_smoke.txt
    run_arm continuous $D/ids_val300_smoke.txt lora_waypoint_cont_smoke2
    exit
fi
for ARM in ${ARMS:-continuous discrete}; do
    case $ARM in
        continuous) run_arm continuous $V300 lora_waypoint_cont_val300 ;;
        discrete)   run_arm discrete $REST lora_waypoint_val300rest ;;
    esac
done
