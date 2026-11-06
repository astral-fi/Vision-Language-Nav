#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$(readlink -f "$0")")/../run_waffle.sh" \
  'Turn left until you see an open door. Exit the room through that door, then stop beside the Rubiks cube on your left.'
