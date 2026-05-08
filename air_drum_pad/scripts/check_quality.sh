#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[quality] Python syntax check"
python3 -m py_compile \
  main.py hand_tracker.py strike_detector.py drumkit_audio.py \
  tools/*.py tests/*.py

echo "[quality] Unit tests"
python3 -m unittest discover -s tests -v

echo "[quality] Palm decode tests"
python3 tools/test_palm_decode.py

if [[ "${RUN_BENCH_SMOKE:-1}" != "0" ]]; then
  echo "[quality] Dataset benchmark smoke"
  python3 tools/benchmark_dataset.py \
    --backends cpu-baseline,npu-full \
    --limit "${BENCH_LIMIT:-3}" \
    --warmup 1 \
    --no-compare
else
  echo "[quality] Dataset benchmark smoke skipped (RUN_BENCH_SMOKE=0)"
fi

echo "[quality] OK"
