#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 main.py --backend cpu --piano --camera "${CAMERA:-0}" "$@"
