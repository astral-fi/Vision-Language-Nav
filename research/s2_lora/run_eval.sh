#!/bin/bash
# Waits for run_stage2.sh (training) to finish, then runs the offline eval (PLAN step 7).
cd ~/FutureNav/experiments/s2_lora
PY=~/miniforge3/envs/futurenav/bin/python
LOG=~/FutureNav/outputs/s2_lora/logs
while kill -0 2112759 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] offline eval start" >> $LOG/pipeline.log
PYTHONUNBUFFERED=1 $PY offline_eval.py --adapter ~/FutureNav/outputs/s2_lora/lora_v1/adapter_final > $LOG/offline_eval.log 2>&1
echo "[$(date '+%F %T')] offline eval exit $?" >> $LOG/pipeline.log
