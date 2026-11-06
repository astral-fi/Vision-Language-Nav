#!/usr/bin/env bash
# Downloads FutureNav-4B-Base (~11.7 GB) into pretrain_models/.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"
mkdir -p pretrain_models
hf download llxs/FutureNav --include "FutureNav-4B-Base/*" --local-dir pretrain_models \
  || huggingface-cli download llxs/FutureNav --include "FutureNav-4B-Base/*" --local-dir pretrain_models
du -sh pretrain_models/FutureNav-4B-Base
