#!/usr/bin/env bash
# After training (PID $1), drive each head on 10 holdout episodes, one at a time
# (FutureNav + head need the GPU alone). Logs/traces: outputs/motion_head/drive/.
set -uo pipefail
cd "$(dirname "$0")/../.."
while kill -0 "${1:?train pid}" 2>/dev/null; do sleep 30; done
D=outputs/motion_head/drive
for TAG in reg_v1 flow_v1; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD/src:$PWD:$PWD/deploy \
        ~/miniforge3/envs/futurenav/bin/python deploy/head_server.py --checkpoint pretrain_models/FutureNav-4B-Base \
        --head outputs/motion_head/$TAG/best.pt --port 8643 > $D/server_$TAG.log 2>&1 &
    S=$!
    until grep -q "ready on" $D/server_$TAG.log || ! kill -0 $S 2>/dev/null; do sleep 5; done
    PATH=$HOME/miniforge3/envs/habitat/bin:$PATH DATASET_SPLIT=train EPISODE_IDS=$PWD/$D/ids10.txt \
        TAG=${TAG}_holdout10 bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:8643 > $D/${TAG}_holdout10.log 2>&1
    kill $S; wait $S 2>/dev/null
    echo "$TAG done $(date)"
done
