#!/usr/bin/env bash
# Run a data-engine script in the habitat env with the headless EGL setup of
# eval/eval_r2r_remote.sh:  bash run_habitat.sh validate.py --check clean ...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
unset DISPLAY
SYS_GL=/usr/lib/x86_64-linux-gnu
[ -e "$SYS_GL/libEGL.so.1" ] && export LD_PRELOAD="$SYS_GL/libEGL.so.1:$SYS_GL/libGLdispatch.so.0"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export PYTHONPATH="$HERE:$ROOT:$ROOT/VLN_CE:${PYTHONPATH:-}"
cd "$ROOT"
exec ~/miniforge3/envs/habitat/bin/python "$HERE/$@"
