#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DXNN:?Set DXNN=/path/to/hand.dxnn}"
_DEFAULT_LAYOUT="$(pwd)/models/dxnn_layout.mediapipe_hand_lite_dual.json"
LAYOUT="${DXNN_LAYOUT:-$_DEFAULT_LAYOUT}"
ARGS=(--backend npu --dxnn "$DXNN" --piano --camera "${CAMERA:-0}")
if [[ -n "$LAYOUT" ]]; then
  ARGS+=(--dxnn-layout "$LAYOUT")
fi
exec python3 main.py "${ARGS[@]}" "$@"
