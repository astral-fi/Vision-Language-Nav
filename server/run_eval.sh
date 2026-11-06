#!/usr/bin/env bash
# End-to-end FutureNav evaluation: start the model server, run the Habitat
# eval against it, report, clean up.
#
#   bash deploy/run_eval.sh                        # 50 episodes of R2R val_unseen
#   bash deploy/run_eval.sh --episodes 0           # all 1839
#   bash deploy/run_eval.sh --lam 0 --tag lam00    # geometry-tower ablation
#   bash deploy/run_eval.sh --keep-server          # leave it up for a second run
#
# Why one script for two things: habitat-sim 0.1.7 is capped at Python 3.8 and
# Qwen3-VL needs >= 3.9, so the simulator and the model live in separate conda
# envs and talk over HTTP. This drives both sides.
set -uo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

# ---- defaults ----
SPLIT=val_unseen
EPISODES=50
TAG=""
PORT=8642
LAM=""
VGGT_RECENT=""
ATTN=sdpa
NUM_HISTORY=8
KEEP_SERVER=0
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
MODEL_ENV="${MODEL_ENV:-futurenav}"
HABITAT_ENV="${HABITAT_ENV:-habitat}"
CHECKPOINT="${CHECKPOINT:-pretrain_models/FutureNav-4B-Base}"

while [ $# -gt 0 ]; do
  case "$1" in
    --split)        SPLIT=$2; shift 2 ;;
    --episodes)     EPISODES=$2; shift 2 ;;
    --tag)          TAG=$2; shift 2 ;;
    --port)         PORT=$2; shift 2 ;;
    --lam)          LAM=$2; shift 2 ;;
    --vggt-recent)  VGGT_RECENT=$2; shift 2 ;;
    --attn)         ATTN=$2; shift 2 ;;
    --num-history)  NUM_HISTORY=$2; shift 2 ;;
    --checkpoint)   CHECKPOINT=$2; shift 2 ;;
    --keep-server)  KEEP_SERVER=1; shift ;;
    -h|--help)      sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

[ -z "$TAG" ] && TAG="lam${LAM:-default}_h${NUM_HISTORY}_$(date +%m%d-%H%M)"
LOG_DIR="${PROJECT_DIR}/outputs/logs"; mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/server_${TAG}.log"
PIDFILE="${LOG_DIR}/server_${TAG}.pid"

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---- cleanup: PID-based only. Never pkill -f: the pattern matches this very
# shell's own command line, which silently kills the script itself.
STARTED_SERVER=0
cleanup() {
  if [ "$STARTED_SERVER" = 1 ] && [ "$KEEP_SERVER" = 0 ] && [ -f "$PIDFILE" ]; then
    local pid; pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping server (pid $pid)"
      kill "$pid" 2>/dev/null
      for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PIDFILE"
  elif [ "$KEEP_SERVER" = 1 ]; then
    echo "server left running on port ${PORT} (--keep-server)"
  fi
}
trap cleanup EXIT INT TERM

# ---- 0. preflight ----
log "Preflight"
"${CONDA_BASE}/envs/${HABITAT_ENV}/bin/python" deploy/check_data.py --checkpoint "${CHECKPOINT}" || {
  echo "data check failed -- fix the [FAIL] lines above before running"; exit 1; }

# ---- 1. server ----
if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  log "Reusing server already listening on ${PORT}"
else
  log "Starting model server (env: ${MODEL_ENV})"
  # setsid detaches from this process group, so a timeout or Ctrl-C on the
  # caller does not take the server down mid-load.
  ( source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${MODEL_ENV}"
    export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}"
    exec setsid python deploy/server.py \
      --checkpoint "${CHECKPOINT}" \
      --attn "${ATTN}" \
      --num-history "${NUM_HISTORY}" \
      ${LAM:+--lam "${LAM}"} \
      ${VGGT_RECENT:+--vggt-recent-frames "${VGGT_RECENT}"} \
      --port "${PORT}"
  ) > "${SERVER_LOG}" 2>&1 < /dev/null &
  sleep 2
  SERVER_PID=$(ps -eo pid,cmd | awk -v p="--port ${PORT}" '$0 ~ /deploy\/server\.py/ && index($0,p) {print $1; exit}')
  echo "${SERVER_PID}" > "${PIDFILE}"
  STARTED_SERVER=1
  echo "server pid ${SERVER_PID}, log: ${SERVER_LOG}"

  echo -n "waiting for model load "
  for i in $(seq 180); do
    if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo " ready"; break; fi
    if grep -q "Traceback" "${SERVER_LOG}" 2>/dev/null; then
      echo; echo "SERVER FAILED TO START:"; tail -25 "${SERVER_LOG}"; exit 1
    fi
    echo -n "."; sleep 2
    [ "$i" = 180 ] && { echo " TIMEOUT after 6 min"; tail -20 "${SERVER_LOG}"; exit 1; }
  done
  grep -E "^\[server\]" "${SERVER_LOG}" | head -5
fi

# ---- 2. eval ----
log "Running Habitat eval (env: ${HABITAT_ENV}, split=${SPLIT}, episodes=${EPISODES}, tag=${TAG})"
START=$(date +%s)
( source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${HABITAT_ENV}"
  TAG="${TAG}" bash eval/eval_r2r_remote.sh "${SPLIT}" "${EPISODES}" "http://127.0.0.1:${PORT}"
)
RC=$?
ELAPSED=$(( $(date +%s) - START ))

# ---- 3. report ----
log "Done in $((ELAPSED/60))m $((ELAPSED%60))s (exit ${RC})"
echo "results:    outputs/eval/r2r/${TAG}/${SPLIT}/"
echo "server log: ${SERVER_LOG}"
echo
echo "peak VRAM / cache depth reached:"
grep -oE "cache=[0-9]+f  vram=[0-9.]+/[0-9.]+GB" "${SERVER_LOG}" 2>/dev/null | tail -1 || echo "  (none logged)"
FB=$(grep -c "FALLBACK" "${SERVER_LOG}" 2>/dev/null || echo 0)
echo "parse fallbacks: ${FB}  (a burst means the model is emitting unparseable text)"
exit ${RC}
