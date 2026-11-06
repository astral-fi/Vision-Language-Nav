#!/usr/bin/env bash
# Test 0: re-run the LoRA waypoint head with --log-probs on the 52 episodes it lost but LoRA won
# (from lost_episodes.json), to see how the 256-anchor probability is spread at the fork.
#   bash experiments/head_eval/test0.sh smoke     # 1 episode
#   bash experiments/head_eval/test0.sh full      # all 52
# Results: outputs/eval/r2r/lora_waypoint_t0_<mode>/val_unseen/trace (head.probs per query). Analysis: test0_analyze.py
set -uo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-smoke}
D=outputs/head_eval; mkdir -p $D
PY=~/miniforge3/envs/futurenav/bin/python
export PYTHONPATH=$PWD/src:$PWD:$PWD/deploy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
IDS=experiments/head_eval/ids_t0_lora_won.txt
$PY -c "
import json; L=[e['id'] for e in json.load(open('experiments/head_eval/lost_episodes.json')) if e['success']['lora']]
open('$IDS','w').write('\n'.join(L)+'\n'); print(len(L), 'episodes')"
[ "$MODE" = smoke ] && { head -1 $IDS > $D/ids_t0_smoke.txt; IDS=$D/ids_t0_smoke.txt; }
NAME=lora_waypoint_t0_$MODE; PORT=8652
$PY deploy/head_eval_server.py --arm lora_waypoint --log-probs --port $PORT > $D/server_$NAME.log 2>&1 &
S=$!
until curl -sf http://127.0.0.1:$PORT/health | grep -q '"ready": true' || ! kill -0 $S 2>/dev/null; do sleep 5; done
if ! kill -0 $S 2>/dev/null; then echo "$NAME: server died, see $D/server_$NAME.log"; exit 1; fi
PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=val_unseen EPISODE_IDS=$IDS EXP_SAVE=trace TAG=$NAME \
    bash eval/eval_r2r_remote.sh val_unseen 0 http://127.0.0.1:$PORT > $D/$NAME.log 2>&1
curl -s http://127.0.0.1:$PORT/health > $D/health_$NAME.json
kill $S; wait $S 2>/dev/null
echo "[$(date '+%F %T')] $NAME done: $(grep -o 'SR=.*' $D/$NAME.log | tail -1)" | tee -a $D/summary.log
