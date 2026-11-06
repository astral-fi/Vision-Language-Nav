#!/bin/bash
# Waits for the val extraction, then train extraction (1,200 episodes), then the probe.
cd ~/FutureNav/experiments/pose_head
PY=~/miniforge3/envs/futurenav/bin/python
L=~/FutureNav/outputs/pose_head/logs
while pgrep -f "extract.py --split val" >/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] train extraction" >> $L/pipeline.log
$PY -u extract.py --split train --max-episodes 1200 > $L/extract_train.log 2>&1
echo "[$(date '+%F %T')] probe (extract exit $?)" >> $L/pipeline.log
$PY -u probe.py > $L/probe.log 2>&1
echo "[$(date '+%F %T')] probe exit $?" >> $L/pipeline.log
