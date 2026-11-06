#!/usr/bin/env bash
# FutureNav R2R evaluation with remote inference.
#
#   Terminal 1 (futurenav env):  python deploy/server.py --checkpoint ... --attn sdpa
#   Terminal 2 (habitat env):    bash eval/eval_r2r_remote.sh val_unseen 50
#
# Usage: bash eval/eval_r2r_remote.sh [split] [max_episodes] [server_url]
#
# Single chunk only: the server holds one VGGT cache, so it serves one episode
# at a time. That is a property of the model, not a limitation of this script.
set -euo pipefail

SPLIT=${1:-val_unseen}
MAX_EPISODES=${2:-50}
SERVER=${3:-http://127.0.0.1:8642}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")
CONFIG_PATH="${PROJECT_DIR}/eval/config/futurenav_r2r.yaml"
TAG=${TAG:-remote}
SAVE_PATH="${PROJECT_DIR}/outputs/eval/r2r/${TAG}/${SPLIT}"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# --- rendering context -------------------------------------------------------
# This is the headless habitat-sim build: it renders through EGL straight onto
# the GPU and needs no X server at all. Two things have to be right, and both
# fail with the same unhelpful message:
#     GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
#
# 1. DISPLAY must be UNSET. With it set, Magnum takes the X11/GLX path into
#    whatever X server is there (an Xvfb software rasterizer, or your desktop)
#    instead of EGL, and the software GL cannot supply the required version.
#
# 2. habitat_sim_bindings.so is built with RPATH '$ORIGIN/../../../..', so it
#    binds conda's libEGL.so.1 and libGLdispatch.so.0 rather than the system
#    libglvnd. Conda's copies do not read /usr/share/glvnd/egl_vendor.d/, so the
#    NVIDIA driver is never discovered. RPATH beats LD_LIBRARY_PATH, so the only
#    way to override it is LD_PRELOAD.
unset DISPLAY
SYS_GL=/usr/lib/x86_64-linux-gnu
if [ -e "${SYS_GL}/libEGL.so.1" ]; then
    export LD_PRELOAD="${SYS_GL}/libEGL.so.1:${SYS_GL}/libGLdispatch.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [ -n "${HABITAT_SIM_EXT:-}" ]; then
    export LD_LIBRARY_PATH=${HABITAT_SIM_EXT}:${LD_LIBRARY_PATH:-}
fi
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/VLN_CE:${SCRIPT_DIR}:${PYTHONPATH:-}"

mkdir -p "${SAVE_PATH}"
echo "================================================"
echo "  FutureNav R2R -- remote inference"
echo "  server:   ${SERVER}"
echo "  split:    ${SPLIT}"
echo "  episodes: ${MAX_EPISODES} (0 = all)"
echo "  cam hgt:  ${CAM_HEIGHT:-1.25 (training default)} m"
echo "  cam pitch:${CAM_PITCH:-0 (level, training default)} deg"
echo "  steering: ${ACKERMANN_RADIUS:+Ackermann arcs, R=${ACKERMANN_RADIUS} m}${ACKERMANN_RADIUS:-turn-in-place (training default)}"
echo "  probes:   ${PROBE_SPEC:-none}${POINTER_PROBE:+ + E2.3 pointer arms}"
echo "  pointer:  ${POINTER_SCHEDULE:+executing with shift ${POINTER_SCHEDULE}}${POINTER_SCHEDULE:-none}"
echo "  hidden:   ${DUMP_HIDDEN:+capturing per-step states}${DUMP_HIDDEN:-none}"
echo "  capture:  ${CAPTURE_DIR:-none}"
echo "  derail:   ${DERAIL_SPEC:-none}"
echo "  steer:    ${STEER_SPEC:-none}"
echo "  output:   ${SAVE_PATH}"
echo "================================================"

if ! curl -sf "${SERVER}/health" >/dev/null; then
    echo "ERROR: no server at ${SERVER}. Start deploy/server.py in the futurenav env first."
    exit 1
fi
curl -s "${SERVER}/health"; echo

cd "${PROJECT_DIR}"
python "${SCRIPT_DIR}/run.py" \
    --exp-config "${CONFIG_PATH}" \
    --split-num 1 --split-id 0 \
    --server "${SERVER}" \
    --result-path "${SAVE_PATH}" \
    --exp-save "${EXP_SAVE:-data}" \
    --num-history 8 \
    --max-steps 400 \
    --max-episodes "${MAX_EPISODES}" \
    ${EPISODE_IDS:+--episode-ids "${EPISODE_IDS}"} \
    ${INSTRUCTION_OVERRIDE:+--instruction-override "${INSTRUCTION_OVERRIDE}"} \
    ${PROBE_SPEC:+--probe-spec "${PROBE_SPEC}"} \
    ${POINTER_PROBE:+--pointer-probe} \
    ${POINTER_SCHEDULE:+--pointer-schedule "${POINTER_SCHEDULE}"} \
    ${DUMP_HIDDEN:+--dump-hidden} \
    ${CAPTURE_DIR:+--capture-dir "${CAPTURE_DIR}"} \
    ${CAPTURE_VALIDATE_EVERY:+--capture-validate-every "${CAPTURE_VALIDATE_EVERY}"} \
    ${CAPTURE_PROFILE:+--capture-profile "${CAPTURE_PROFILE}"} \
    ${CAPTURE_MIN_FREE_GB:+--capture-min-free-gb "${CAPTURE_MIN_FREE_GB}"} \
    ${DATASET_SPLIT:+--dataset-split "${DATASET_SPLIT}"} \
    ${DERAIL_SPEC:+--derail-spec "${DERAIL_SPEC}"} \
    ${STEER_SPEC:+--steer-spec "${STEER_SPEC}"} \
    ${CAM_HEIGHT:+--cam-height "${CAM_HEIGHT}"} \
    ${CAM_PITCH:+--cam-pitch "${CAM_PITCH}"} \
    ${ACKERMANN_RADIUS:+--ackermann-radius "${ACKERMANN_RADIUS}"} \
    ${WAYPOINT_EXEC:+--waypoint-exec} \
    ${S1:+--s1} \
    ${S1_CONFIG:+--s1-config "${S1_CONFIG}"}

python - "${SAVE_PATH}" <<'PY'
import json, os, sys, glob
path = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(path, "log", "stats_*.json"))):
    with open(f) as fh:
        rows.append(json.load(fh))
if not rows:
    print("no per-episode stats found"); raise SystemExit(1)
def avg(k):
    vals = [r.get(k, 0) for r in rows]
    vals = [0 if v != v or v in (float("inf"), float("-inf")) else v for v in vals]
    return sum(vals) / len(vals)
print("\nepisodes %d" % len(rows))
print("  SR   %.4f" % avg("success"))
print("  SPL  %.4f" % avg("spl"))
print("  OS   %.4f" % avg("oracle_success"))
print("  DTG  %.2f m" % avg("distance_to_goal"))
print("  PL   %.2f m" % avg("path_length"))
print("  steps %.1f" % avg("steps"))
PY
