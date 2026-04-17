#!/usr/bin/env python3
"""
Palm TFLite 입력/출력 shape 스모크 (tensorflow 또는 tflite_runtime 필요).

  python3 tools/smoke_palm_interpreter.py --variant lite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from palm_letterbox import rgb_uint8_to_palm_input_tensor
from palm_mp_spec import PALM_INPUT_H, PALM_INPUT_W


def _load_interpreter():
    try:
        import tensorflow as tf

        return "tensorflow", tf.lite.Interpreter
    except Exception:
        pass
    try:
        import tflite_runtime.interpreter as tflite

        return "tflite_runtime", tflite.Interpreter
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("lite", "full"), default="lite")
    ap.add_argument(
        "--tflite",
        type=Path,
        default=None,
        help="기본: models/vendor/palm_detection_{variant}.tflite (export_mediapipe_palm_onnx.py 선행)",
    )
    args = ap.parse_args()

    lib, Interpreter = _load_interpreter()
    if Interpreter is None:
        print(
            "tensorflow 또는 tflite_runtime 패키지가 필요합니다. "
            "설치 후 재실행하세요.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    tflite_path = args.tflite
    if tflite_path is None:
        tflite_path = _ROOT / "models" / "vendor" / f"palm_detection_{args.variant}.tflite"
    if not tflite_path.is_file():
        print(f"missing {tflite_path} — run tools/export_mediapipe_palm_onnx.py first", file=sys.stderr)
        return 1

    print("runtime:", lib, flush=True)
    intr = Interpreter(model_path=str(tflite_path))
    intr.allocate_tensors()
    inp = intr.get_input_details()[0]
    print("input", inp["name"], inp["shape"], inp["dtype"], flush=True)
    for i, o in enumerate(intr.get_output_details()):
        print("output", i, o["name"], o["shape"], o["dtype"], flush=True)

    import numpy as np

    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    x, _ = rgb_uint8_to_palm_input_tensor(rgb)
    if inp["dtype"] in (np.float32, "float32"):
        x = x.astype(np.float32)
    elif inp["dtype"] in (np.uint8, "uint8"):
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
    else:
        print("unsupported input dtype", inp["dtype"], file=sys.stderr)
        return 1

    intr.set_tensor(inp["index"], x)
    intr.invoke()
    print("invoke OK", flush=True)
    for o in intr.get_output_details():
        arr = intr.get_tensor(o["index"])
        print(" out", o["name"], arr.shape, arr.dtype, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
