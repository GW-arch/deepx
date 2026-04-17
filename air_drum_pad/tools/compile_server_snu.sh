#!/usr/bin/env bash
# SNU 실습 컴파일 서버 — ONNX 업로드 → dx_com → .dxnn 다운로드
# 문서: https://sites.google.com/view/dxs-2603-snu/ (실습5 DX-AS NPU 컴파일)
# 비밀번호는 Git에 넣지 말고, SSH 키 또는 DX_COMPILE_PASSFILE(chmod 600) 사용.
set -euo pipefail

HOST="${DX_COMPILE_HOST:-43.203.143.33}"
PORT="${DX_COMPILE_PORT:-443}"
USER="${DX_COMPILE_USER:?환경변수 DX_COMPILE_USER (예: user12)}"
REMOTE_SAMPLE="/home/${USER}/sample"
REMOTE_OUT="/home/${USER}/output"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${COMPILE_ONNX:-$ROOT/models/vendor/hand_landmark_lite.onnx}"
JSON="${COMPILE_JSON:-$ROOT/models/dxcom/hand_landmark_lite.json}"
OUT_NAME="${COMPILE_OUT_NAME:-hand_landmark_lite.dxnn}"

if [[ "${USER}" =~ ^user([0-9]+)$ ]]; then
  N="${BASH_REMATCH[1]}"
  CPUS="${N},$((N + 1))"
else
  CPUS="${DX_COMPILE_TASKSET:-0,1}"
fi

ssh_base=(ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new "${USER}@${HOST}")
scp_to=(scp -P "${PORT}" -o StrictHostKeyChecking=accept-new)
scp_from=(scp -P "${PORT}" -o StrictHostKeyChecking=accept-new)

if [[ -n "${DX_COMPILE_PASSFILE:-}" ]]; then
  SSHWRAP=(sshpass -f "${DX_COMPILE_PASSFILE}")
elif [[ -n "${SSHPASS:-}" ]]; then
  SSHWRAP=(sshpass -e)
else
  SSHWRAP=()
fi

run_remote() {
  if ((${#SSHWRAP[@]})); then
    "${SSHWRAP[@]}" "${ssh_base[@]}" "$@"
  else
    "${ssh_base[@]}" "$@"
  fi
}

upload() {
  echo "[1/4] 업로드: ${ONNX} ${JSON} → ${HOST}:${REMOTE_SAMPLE}/"
  if ((${#SSHWRAP[@]})); then
    "${SSHWRAP[@]}" "${scp_to[@]}" "${ONNX}" "${JSON}" "${USER}@${HOST}:${REMOTE_SAMPLE}/"
  else
    "${scp_to[@]}" "${ONNX}" "${JSON}" "${USER}@${HOST}:${REMOTE_SAMPLE}/"
  fi
}

compile() {
  echo "[2/4] 컴파일: taskset -c ${CPUS} dx_com …"
  run_remote bash -s <<EOF
set -euo pipefail
cd
mkdir -p output
taskset -c ${CPUS} dx_com \\
  -m ${REMOTE_SAMPLE}/$(basename "${ONNX}") \\
  -c ${REMOTE_SAMPLE}/$(basename "${JSON}") \\
  -o ${REMOTE_OUT}
ls -la ${REMOTE_OUT}/${OUT_NAME}
EOF
}

download() {
  echo "[3/4] 다운로드: ${OUT_NAME} → ${ROOT}/models/vendor/"
  mkdir -p "${ROOT}/models/vendor"
  if ((${#SSHWRAP[@]})); then
    "${SSHWRAP[@]}" "${scp_from[@]}" "${USER}@${HOST}:${REMOTE_OUT}/${OUT_NAME}" "${ROOT}/models/vendor/${OUT_NAME}"
  else
    "${scp_from[@]}" "${USER}@${HOST}:${REMOTE_OUT}/${OUT_NAME}" "${ROOT}/models/vendor/${OUT_NAME}"
  fi
}

verify() {
  echo "[4/4] parse_model"
  parse_model -m "${ROOT}/models/vendor/${OUT_NAME}" | head -22
}

case "${1:-all}" in
  -h|--help)
    echo "usage: $0 [all|upload|compile|download|verify]"
    echo "  DX_COMPILE_USER=user12 [DX_COMPILE_HOST=43.203.143.33] [DX_COMPILE_PORT=443]"
    echo "  선택: DX_COMPILE_PASSFILE=~/.snupass (chmod 600, 한 줄 비밀번호) 또는 ssh-copy-id 로 무암호"
    echo "  선택: COMPILE_ONNX=… COMPILE_JSON=… COMPILE_OUT_NAME=…"
    exit 0
    ;;
  upload) upload ;;
  compile) compile ;;
  download) download ;;
  verify) verify ;;
  all) upload && compile && download && verify ;;
  *) echo "usage: $0 [all|upload|compile|download|verify]  (-h 도움말)"; exit 1 ;;
esac
