#!/usr/bin/env bash
# ONNX → .dxnn (DX-COM). DEEPX DX-AllSuite / DXNN SDK 설치 후 사용.
set -euo pipefail

ONNX="${1:?usage: $0 model.onnx output.dxnn [extra_args...]}"
OUT="${2:?}"
shift 2

if [[ -n "${DX_COM:-}" ]]; then
  echo "Using DX_COM=${DX_COM}"
  exec bash -c "${DX_COM} $(printf '%q ' "$ONNX" "$OUT" "$@")"
fi

# 흔한 CLI 이름 후보 (설치 환경마다 다름 — 실패 시 DX_COM 환경변수로 전체 명령 지정)
# dx-all-suite 설치 시 venv 안의 dxcom:
for venv in "${HOME}/dx-all-suite/dx-compiler/venv-dx-compiler-local" "${HOME}/dx-all-suite/dx-compiler/venv-dx-compiler"; do
  if [[ -x "${venv}/bin/dxcom" ]]; then
    echo "Found: ${venv}/bin/dxcom"
    exec "${venv}/bin/dxcom" "$ONNX" "$OUT" "$@"
  fi
done
for cmd in dx_com dx-com dxcom DX-COM; do
  if command -v "$cmd" &>/dev/null; then
    echo "Found: $cmd"
    exec "$cmd" "$ONNX" "$OUT" "$@"
  fi
done

cat <<EOF >&2
DX-COM 실행 파일을 찾지 못했습니다.

  1) DX-AllSuite 문서에 따라 DX-COM 설치
  2) 아래처럼 컴파일 명령 전체를 환경변수로 지정:

     export DX_COM='dx_com --config /path/to/compile.json --input'
     $0 model.onnx out.dxnn

또는 수동으로 ONNX와 (필요 시) compile 설정 JSON을 DX-COM에 넣어 .dxnn 을 생성한 뒤:

     python3 main.py --backend npu --dxnn ./hand.dxnn --dxnn-layout models/vendor/dxnn_layout.mediapipe_hand_lite.json
EOF
exit 127
