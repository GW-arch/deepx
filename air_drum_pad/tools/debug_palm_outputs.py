#!/usr/bin/env python3
"""Compare CPU TFLite palm outputs with experimental DXNN palm outputs.

Typical use:

  python3 tools/debug_palm_outputs.py --image dataset/frame_000.png
  python3 tools/debug_palm_outputs.py --backend tflite --image dataset/frame_000.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from palm_decode import (
    DET_SCORE_IDX,
    DET_XMAX_IDX,
    DET_XMIN_IDX,
    DET_YMAX_IDX,
    DET_YMIN_IDX,
    decode_palm_tensors,
    generate_ssd_anchors,
)
from palm_letterbox import rgb_uint8_to_palm_input_tensor


def _load_interpreter() -> Any:
    try:
        import tensorflow as tf

        return tf.lite.Interpreter
    except Exception:
        pass

    try:
        import tflite_runtime.interpreter as tflite

        return tflite.Interpreter
    except Exception as exc:
        raise RuntimeError("tensorflow or tflite_runtime is required for TFLite palm inference") from exc


def _read_rgb(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed to read camera frame: {args.camera}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    default = _ROOT / "dataset" / "frame_000.png"
    bgr = cv2.imread(str(default), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"provide --image or --camera; default image missing: {default}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _tensor_stats(arr: np.ndarray) -> dict[str, Any]:
    flat = arr.astype(np.float32).reshape(-1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
        "std": float(flat.std()) if flat.size else None,
    }


def _run_tflite(model_path: Path, tensor: np.ndarray) -> list[np.ndarray]:
    Interpreter = _load_interpreter()
    intr = Interpreter(model_path=str(model_path))
    intr.allocate_tensors()
    inp = intr.get_input_details()[0]
    out_details = intr.get_output_details()

    x = tensor
    if inp["dtype"] in (np.float32, "float32"):
        x = x.astype(np.float32)
    elif inp["dtype"] in (np.uint8, "uint8"):
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
    else:
        raise RuntimeError(f"unsupported TFLite input dtype: {inp['dtype']}")

    intr.set_tensor(inp["index"], x)
    intr.invoke()
    return [intr.get_tensor(o["index"]) for o in out_details]


def _run_dxnn(model_path: Path, tensor: np.ndarray) -> list[np.ndarray]:
    try:
        from dx_engine import InferenceEngine
    except Exception as exc:
        raise RuntimeError("dx_engine is required for DXNN palm inference") from exc

    ie = InferenceEngine(str(model_path))
    try:
        x = (tensor[0] * 255.0).clip(0, 255).astype(np.uint8)[None, ...]
        return list(ie.run([x]))
    finally:
        ie.dispose()


def _summarize(
    name: str,
    raw_outputs: list[np.ndarray],
    anchors: np.ndarray,
    letterbox_meta: Any,
    score_thresh: float,
    top_k: int,
) -> dict[str, Any]:
    dets = decode_palm_tensors(
        raw_outputs,
        anchors,
        letterbox_meta=letterbox_meta,
        score_thresh=score_thresh,
    )
    top = []
    for det in dets[:top_k]:
        top.append(
            {
                "score": float(det[DET_SCORE_IDX]),
                "box_yxyx": [
                    float(det[DET_YMIN_IDX]),
                    float(det[DET_XMIN_IDX]),
                    float(det[DET_YMAX_IDX]),
                    float(det[DET_XMAX_IDX]),
                ],
            }
        )
    return {
        "backend": name,
        "raw_outputs": [_tensor_stats(np.asarray(out)) for out in raw_outputs],
        "detections": int(dets.shape[0]),
        "top": top,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=None, help="RGB source image path; default: dataset/frame_000.png")
    p.add_argument("--camera", type=int, default=-1, help="Capture one frame when --image is omitted")
    p.add_argument("--tflite", type=Path, default=_ROOT / "models" / "vendor" / "palm_detection_lite.tflite")
    p.add_argument("--dxnn", type=Path, default=_ROOT / "models" / "vendor" / "palm_detection_lite.dxnn")
    p.add_argument("--backend", choices=("both", "tflite", "dxnn"), default="both")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", type=Path, default=None, help="Optional path for full JSON summary")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rgb = _read_rgb(args)
    tensor, meta = rgb_uint8_to_palm_input_tensor(rgb)
    anchors = generate_ssd_anchors()

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    if args.backend in ("both", "tflite"):
        try:
            summaries.append(
                _summarize(
                    "tflite",
                    _run_tflite(args.tflite, tensor),
                    anchors,
                    meta,
                    args.score_thresh,
                    args.top_k,
                )
            )
        except Exception as exc:
            errors.append({"backend": "tflite", "error": str(exc)})

    if args.backend in ("both", "dxnn"):
        try:
            summaries.append(
                _summarize(
                    "dxnn",
                    _run_dxnn(args.dxnn, tensor),
                    anchors,
                    meta,
                    args.score_thresh,
                    args.top_k,
                )
            )
        except Exception as exc:
            errors.append({"backend": "dxnn", "error": str(exc)})

    report = {
        "image_shape": list(rgb.shape),
        "score_thresh": args.score_thresh,
        "summaries": summaries,
        "errors": errors,
    }

    print(json.dumps(report, indent=2), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 1 if not summaries else 0


if __name__ == "__main__":
    raise SystemExit(main())
