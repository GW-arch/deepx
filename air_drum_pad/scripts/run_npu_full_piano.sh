#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export DISPLAY="${DISPLAY:-:0}"
if [[ -z "${XAUTHORITY:-}" && -f "${HOME}/.Xauthority" ]]; then
  export XAUTHORITY="${HOME}/.Xauthority"
fi
export PYTHONUNBUFFERED=1

_DEFAULT_DXNN="$(pwd)/models/vendor/hand_landmark_lite.dxnn"
if [[ -z "${DXNN:-}" && -f "${_DEFAULT_DXNN}" ]]; then
  export DXNN="${_DEFAULT_DXNN}"
fi
: "${DXNN:?Set DXNN=/path/to/hand_landmark_lite.dxnn}"

_DEFAULT_PALM="$(pwd)/models/vendor/palm_detection_lite.tflite"
PALM_TFLITE="${PALM_TFLITE:-$_DEFAULT_PALM}"
: "${PALM_TFLITE:?Set PALM_TFLITE=/path/to/palm_detection_lite.tflite}"

_DEFAULT_LAYOUT="$(pwd)/models/dxnn_layout.mediapipe_hand_lite.json"
LAYOUT="${DXNN_LAYOUT:-$_DEFAULT_LAYOUT}"

_DEFAULT_CORR="$(pwd)/models/npu_landmark_correction.dataset.json"
CORRECTION="${LANDMARK_CORRECTION:-}"
if [[ -z "$CORRECTION" && -f "$_DEFAULT_CORR" && "${USE_LANDMARK_CORRECTION:-1}" != "0" ]]; then
  CORRECTION="$_DEFAULT_CORR"
fi

MH="${MAX_HANDS:-2}"
ARGS=(
  --backend npu-full
  --dxnn "$DXNN"
  --palm-tflite "$PALM_TFLITE"
  --max-hands "$MH"
  --piano
  --camera "${CAMERA:-0}"
)
if [[ -n "$LAYOUT" ]]; then
  ARGS+=(--dxnn-layout "$LAYOUT")
fi
if [[ -n "$CORRECTION" ]]; then
  ARGS+=(--landmark-correction "$CORRECTION")
fi
if [[ "${ASYNC_PALM:-0}" == "1" ]]; then
  ARGS+=(--async-palm)
fi
if [[ -n "${PALM_REDETECT_EVERY:-}" ]]; then
  ARGS+=(--palm-redetect-every "$PALM_REDETECT_EVERY")
fi

exec python3 main.py "${ARGS[@]}" "$@"
