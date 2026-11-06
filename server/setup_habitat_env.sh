#!/usr/bin/env bash
# Environment 2 of 2: the simulator. Python 3.8 -- this is forced, not a choice.
#
#   bash deploy/setup_habitat_env.sh
#
# habitat-sim 0.1.7 is published only for py3.6/3.7/3.8 on the aihabitat
# channel. There is no py3.9 build, despite what the top-level README says.
# Since Qwen3-VL needs transformers >= 4.57 (Python >= 3.9), the model cannot
# live here -- it runs in the 'futurenav' env and is reached over HTTP.
set -euo pipefail

ENV_NAME=${ENV_NAME:-habitat}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
HABITAT_LAB_DIR=${HABITAT_LAB_DIR:-${PROJECT_DIR}/third_party/habitat-lab}
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -n "${ENV_NAME}" python=3.8 -y
fi
conda activate "${ENV_NAME}"

# numpy must stay below 1.24: VLN_CE uses the np.float / np.int aliases that
# 1.24 removed (habitat_extensions/sensors.py, actions.py).
pip install "numpy==1.23.5"

conda install -c aihabitat -c conda-forge -y \
    habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7

if [ ! -d "${HABITAT_LAB_DIR}" ]; then
    mkdir -p "$(dirname "${HABITAT_LAB_DIR}")"
    git clone --branch v0.1.7 --depth 1 \
        https://github.com/facebookresearch/habitat-lab.git "${HABITAT_LAB_DIR}"
fi

# habitat_baselines/rl/requirements.txt pins tensorflow==1.13.1 (for the old
# tensorboard video writer). No py3.8 wheel exists, and nothing on the eval
# path imports it, so drop the pin before setup.py reads it.
REQ="${HABITAT_LAB_DIR}/habitat_baselines/rl/requirements.txt"
if grep -q "^tensorflow==" "${REQ}"; then
    sed -i 's/^tensorflow==.*/# tensorflow removed: no py3.8 wheel, unused by the eval path/' "${REQ}"
fi

pushd "${HABITAT_LAB_DIR}" >/dev/null
# Install deps by hand rather than via the pinned requirement files: the ddppo
# set would drag in its own torch, and VLN_CE's list pins tensorflow==1.13.1,
# which has no py3.8 wheel and is never imported by the eval path.
pip install \
    "numpy==1.23.5" \
    "gym==0.23.1" \
    "yacs>=0.1.5" \
    "attrs>=19.1.0" \
    "opencv-python==4.8.1.78" \
    "Pillow<10" \
    tqdm imageio "imageio-ffmpeg" scipy numba \
    ifcfg tensorboard "moviepy>=1.0.1" \
    dtw==1.4.0 fastdtw==0.3.4 networkx jsonlines msgpack_numpy gdown

# habitat-lab 0.1.7 sets InstructionSensor.observation_space = spaces.Discrete(0),
# legal in gym <= 0.21 but gym >= 0.23 asserts n > 0, so Env construction dies with
#     AssertionError: n (counts) have to be positive
# The space is a placeholder (the sensor returns a dict), so Discrete(1) is safe.
VLNPY="${HABITAT_LAB_DIR}/habitat/tasks/vln/vln.py"
if grep -q "spaces.Discrete(0)" "${VLNPY}"; then
    sed -i 's/spaces\.Discrete(0)/spaces.Discrete(1)/' "${VLNPY}"
    echo "patched ${VLNPY}: Discrete(0) -> Discrete(1)"
fi

# lmdb 2.1.1 ships a cp38 wheel built against a Python 3.9+ C API; it fails at
# import with "undefined symbol: Py_SET_REFCNT", and lmdb's cffi fallback then
# tries to compile and also fails. habitat_baselines imports lmdb transitively
# (il/trainers/eqa_cnn_pretrain_trainer.py), so this breaks the whole package.
pip install "lmdb==1.4.1"

# CPU torch, deliberately. habitat-lab imports torch (habitat/core/simulator.py)
# and habitat_baselines needs it, but no tensor work happens in this env -- the
# model runs in the 'futurenav' env behind HTTP. Installing it BEFORE
# `setup.py develop --all` is what matters: otherwise easy_install resolves the
# ddppo torch requirement itself, drops a torch egg in site-packages, and that
# egg fails at import with "libcudnn.so.9: cannot open shared object file".
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
# --all is required: VLN_CE imports habitat_baselines.
python setup.py develop --all
popd >/dev/null

# The conda habitat-sim install pulls numpy 1.24.x back in. Re-pin last:
# VLN_CE uses np.float / np.int, which 1.24 removed.
pip install --force-reinstall --no-deps "numpy==1.23.5"

PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/VLN_CE" python - <<'PY'
import numpy, gym, habitat, habitat_sim, habitat_baselines, habitat_extensions, lmdb
from VLN_CE.vlnce_baselines.config.default import get_config
print("numpy          ", numpy.__version__)
print("gym            ", gym.__version__)
print("habitat-sim    ", habitat_sim.__version__)
assert tuple(int(x) for x in numpy.__version__.split(".")[:2]) < (1, 24), \
    "numpy >= 1.24 removed np.float, which VLN_CE still uses"
cfg = get_config("eval/config/futurenav_r2r.yaml")
print("config loads   ", cfg.TASK_CONFIG.DATASET.SPLIT)
print("habitat-lab, habitat_baselines, habitat_extensions, VLN_CE configs all OK")
PY

echo
echo "Habitat environment '${ENV_NAME}' ready."
echo "Next: get the Matterport3D scenes (see deploy/README.md section 'Data')."
