#!/usr/bin/env python3
"""
MediaPipe 번들 Palm TFLite → ONNX 시도 + vendor 복사.

  python3 tools/export_mediapipe_palm_onnx.py --variant lite

ONNX 변환은 tflite2onnx 가 palm 그래프에서 실패할 수 있음(Phase 3에서 다른 경로).
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _mediapipe_palm_tflite(variant: str) -> Path:
    import mediapipe as mp

    root = Path(mp.__file__).resolve().parent
    name = "palm_detection_lite.tflite" if variant == "lite" else "palm_detection_full.tflite"
    p = root / "modules" / "palm_detection" / name
    if not p.is_file():
        raise FileNotFoundError(f"mediapipe palm tflite not found: {p}")
    return p


def _try_tflite2onnx(tflite: Path, onnx_out: Path) -> bool:
    spec = importlib.util.find_spec("tflite2onnx")
    if spec is None:
        print("tflite2onnx not installed; skip ONNX conversion", flush=True)
        return False
    from tflite2onnx import convert as t2o_convert

    try:
        t2o_convert(str(tflite), str(onnx_out))
        print(f"onnx written: {onnx_out} ({onnx_out.stat().st_size} bytes)", flush=True)
        return True
    except Exception as e:
        print("tflite2onnx failed (expected for some palm builds):", e, file=sys.stderr, flush=True)
        if onnx_out.is_file():
            onnx_out.unlink(missing_ok=True)
        return False


def _onnx_io(onnx_path: Path) -> None:
    import onnx

    m = onnx.load(str(onnx_path))

    def _shape(t) -> list:
        dims = []
        for d in t.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(str(d.dim_param))
            else:
                dims.append(-1)
        return dims

    print("ONNX inputs:", [(i.name, _shape(i)) for i in m.graph.input], flush=True)
    print("ONNX outputs:", [(o.name, _shape(o)) for o in m.graph.output], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("lite", "full"), default="lite")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "models" / "vendor",
        help="tflite/onnx 출력 디렉터리",
    )
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    src = _mediapipe_palm_tflite(args.variant)
    tflite = out_dir / f"palm_detection_{args.variant}.tflite"
    shutil.copy2(src, tflite)
    print(f"copied: {src} -> {tflite}", flush=True)

    onnx_path = out_dir / f"palm_detection_{args.variant}.onnx"
    if _try_tflite2onnx(tflite, onnx_path) and onnx_path.is_file():
        _onnx_io(onnx_path)

    print("\n다음 단계:", flush=True)
    print("  1) docs/PLAN_NPU_FULL_HAND_PIPELINE.md Phase 1 (palm_decode) 구현", flush=True)
    print("  2) ONNX 실패 시 TF/tf2onnx 등으로 palm ONNX 확보 후 DX-COM → .dxnn", flush=True)
    print("  3) python3 tools/smoke_palm_interpreter.py --variant", args.variant, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
