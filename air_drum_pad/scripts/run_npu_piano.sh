#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 로컬 LCD: 그래픽 로그인 세션에서 실행해야 함. SSH만 쓸 때는 X11 포워딩 또는 아래 두 줄을 맞춤.
export DISPLAY="${DISPLAY:-:0}"
if [[ -z "${XAUTHORITY:-}" && -f "${HOME}/.Xauthority" ]]; then
  export XAUTHORITY="${HOME}/.Xauthority"
fi
export PYTHONUNBUFFERED=1

_DEFAULT_DXNN="$(pwd)/models/vendor/hand_landmark_lite.dxnn"
if [[ -z "${DXNN:-}" && -f "${_DEFAULT_DXNN}" ]]; then
  export DXNN="${_DEFAULT_DXNN}"
fi
: "${DXNN:?Set DXNN=/path/to/hand.dxnn (또는 models/vendor/hand_landmark_lite.dxnn 빌드)}"
_DEFAULT_LAYOUT="$(pwd)/models/dxnn_layout.mediapipe_hand_lite_dual.json"
LAYOUT="${DXNN_LAYOUT:-$_DEFAULT_LAYOUT}"
MH="${MAX_HANDS:-2}"
ARGS=(--backend npu --dxnn "$DXNN" --max-hands "$MH" --piano --camera "${CAMERA:-0}")
if [[ -n "$LAYOUT" ]]; then
  ARGS+=(--dxnn-layout "$LAYOUT")
fi
exec python3 main.py "${ARGS[@]}" "$@"
