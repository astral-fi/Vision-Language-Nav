#!/usr/bin/env bash
# Environment 1 of 2: the model. Python 3.10, torch + transformers, no habitat.
#
#   bash deploy/setup_model_env.sh
#
# Safe to re-run. Does NOT install flash-attn: it takes 30+ minutes to build,
# frequently fails, and the eval path uses sdpa anyway.
set -euo pipefail

ENV_NAME=${ENV_NAME:-futurenav}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

# cu124 wheels; the 4090 is sm_89 and is covered by the default build.
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# transformers 4.57 is the first release with Qwen3-VL. Pin below 5.x: this
# codebase is written against the 4.x API (torch_dtype=, the old processor
# kwargs), and an unbounded ">=4.57" resolves to v5 and drifts.
pip install \
    "transformers>=4.57,<5" \
    accelerate \
    qwen-vl-utils \
    timm \
    safetensors \
    "huggingface_hub[cli]" \
    pillow \
    "numpy<2" \
    opencv-python \
    scipy \
    einops

python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "no CUDA device visible"
cap = torch.cuda.get_device_capability()
print("device      ", torch.cuda.get_device_name(0), "sm_%d%d" % cap)
assert cap[0] >= 8, "bf16 requires sm_80+; this build will fall back to fp16"
from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: F401
print("qwen3_vl     import OK")
PY

echo
echo "Model environment '${ENV_NAME}' ready."
echo "Next: bash deploy/fetch_checkpoint.sh"
