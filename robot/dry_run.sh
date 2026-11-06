#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
bash run_bot1.sh "turn to you right until you see the door" --dry-run --max-steps 1 --max-duration 45
