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
    DET_NUM_COORDS,
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


def _split_palm_outputs(raw_outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw_scores = raw_boxes = None
    for t in raw_outputs:
        arr = t[0] if t.ndim == 3 else t
        if arr.ndim == 2 and arr.shape[-1] == DET_NUM_COORDS:
            raw_boxes = arr.astype(np.float32)
        elif arr.ndim == 2 and arr.shape[-1] <= 2:
            raw_scores = arr.astype(np.float32)
        elif arr.ndim == 1:
            raw_scores = arr.astype(np.float32)
    if raw_scores is None or raw_boxes is None:
        raise ValueError("could not split palm outputs into scores and boxes")
    return raw_scores.reshape(-1), raw_boxes.reshape(-1, DET_NUM_COORDS)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = a.astype(np.float64).reshape(-1)
    bb = b.astype(np.float64).reshape(-1)
    if aa.size != bb.size or aa.size == 0:
        return None
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0.0:
        return None
    return float(np.dot(aa, bb) / denom)


def _compare_outputs(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, Any]:
    ref_scores, ref_boxes = _split_palm_outputs(reference)
    cand_scores, cand_boxes = _split_palm_outputs(candidate)
    best_ref_idx = int(np.argmax(ref_scores))
    best_cand_idx = int(np.argmax(cand_scores))
    return {
        "score_corr": _pearson(ref_scores, cand_scores),
        "score_mae": float(np.mean(np.abs(ref_scores - cand_scores))),
        "box_corr": _pearson(ref_boxes.reshape(-1), cand_boxes.reshape(-1)),
        "box_mae": float(np.mean(np.abs(ref_boxes - cand_boxes))),
        "ref_score_max": float(ref_scores[best_ref_idx]),
        "candidate_score_at_ref_max": float(cand_scores[best_ref_idx]),
        "candidate_score_max": float(cand_scores[best_cand_idx]),
        "ref_best_index": best_ref_idx,
        "candidate_best_index": best_cand_idx,
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


def _run_onnx(model_path: Path, tensor: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is required for ONNX palm inference") from exc

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [dim if isinstance(dim, int) else -1 for dim in inp.shape]
    x = tensor.astype(np.float32)
    if len(shape) == 4 and shape[1] == 3:
        x = np.transpose(x, (0, 3, 1, 2))
    elif len(shape) == 4 and shape[-1] == 3:
        pass
    else:
        raise RuntimeError(f"unsupported ONNX input shape for palm model: {inp.shape}")
    outputs = sess.run(None, {inp.name: x})
    meta = {
        "input_name": inp.name,
        "input_shape": list(inp.shape),
        "input_type": inp.type,
        "input": _tensor_stats(x),
        "outputs": [(o.name, list(o.shape), o.type) for o in sess.get_outputs()],
    }
    return [np.asarray(out) for out in outputs], meta


def _dxnn_input_variants(tensor: np.ndarray) -> dict[str, np.ndarray]:
    nhwc_01 = tensor.astype(np.float32)
    nchw_01 = np.transpose(nhwc_01, (0, 3, 1, 2))
    return {
        "nhwc_u8": (nhwc_01[0] * 255.0).clip(0, 255).astype(np.uint8)[None, ...],
        "nchw_u8": (nchw_01 * 255.0).clip(0, 255).astype(np.uint8),
        "nhwc_f32_0_255": nhwc_01 * 255.0,
        "nchw_f32_0_255": nchw_01 * 255.0,
        "nhwc_f32_0_1": nhwc_01,
        "nchw_f32_0_1": nchw_01,
    }


def _run_dxnn(model_path: Path, tensor: np.ndarray, variant: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    try:
        from dx_engine import InferenceEngine
    except Exception as exc:
        raise RuntimeError("dx_engine is required for DXNN palm inference") from exc

    ie = InferenceEngine(str(model_path))
    try:
        variants = _dxnn_input_variants(tensor)
        if variant not in variants:
            known = ", ".join(sorted(variants))
            raise ValueError(f"unknown DXNN input variant: {variant}; known: {known}")
        x = variants[variant]
        meta: dict[str, Any] = {"input_variant": variant, "input": _tensor_stats(x)}
        for attr_name, method_name in [
            ("dxnn_inputs", "get_input_tensors_info"),
            ("dxnn_outputs", "get_output_tensors_info"),
        ]:
            method = getattr(ie, method_name, None)
            if method is not None:
                try:
                    meta[attr_name] = str(method())
                except Exception as exc:
                    meta[attr_name] = f"<error: {exc}>"
        return list(ie.run([x])), meta
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
    p.add_argument("--onnx", type=Path, default=_ROOT / "models" / "vendor" / "palm_detection_lite.onnx")
    p.add_argument("--dxnn", type=Path, default=_ROOT / "models" / "vendor" / "palm_detection_lite.dxnn")
    p.add_argument("--backend", choices=("all", "both", "tflite", "onnx", "dxnn"), default="both")
    p.add_argument(
        "--dxnn-input-variant",
        choices=tuple(_dxnn_input_variants(np.zeros((1, 192, 192, 3), dtype=np.float32)).keys()) + ("all",),
        default="nhwc_u8",
        help="DXNN input packing to test. Existing runtime path is nhwc_u8.",
    )
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
    raw_by_backend: dict[str, list[np.ndarray]] = {}

    if args.backend in ("all", "both", "tflite"):
        try:
            raw_outputs = _run_tflite(args.tflite, tensor)
            raw_by_backend["tflite"] = raw_outputs
            summaries.append(
                _summarize(
                    "tflite",
                    raw_outputs,
                    anchors,
                    meta,
                    args.score_thresh,
                    args.top_k,
                )
            )
        except Exception as exc:
            errors.append({"backend": "tflite", "error": str(exc)})

    if args.backend in ("all", "onnx"):
        try:
            raw_outputs, onnx_meta = _run_onnx(args.onnx, tensor)
            raw_by_backend["onnx"] = raw_outputs
            summary = _summarize(
                "onnx",
                raw_outputs,
                anchors,
                meta,
                args.score_thresh,
                args.top_k,
            )
            summary["onnx_meta"] = onnx_meta
            summaries.append(summary)
        except Exception as exc:
            errors.append({"backend": "onnx", "error": str(exc)})

    if args.backend in ("all", "both", "dxnn"):
        variants = (
            sorted(_dxnn_input_variants(tensor).keys())
            if args.dxnn_input_variant == "all"
            else [args.dxnn_input_variant]
        )
        for variant in variants:
            try:
                raw_outputs, dxnn_meta = _run_dxnn(args.dxnn, tensor, variant)
                raw_by_backend[f"dxnn:{variant}"] = raw_outputs
                summary = _summarize(
                    f"dxnn:{variant}",
                    raw_outputs,
                    anchors,
                    meta,
                    args.score_thresh,
                    args.top_k,
                )
                summary["dxnn_meta"] = dxnn_meta
                summaries.append(summary)
            except Exception as exc:
                errors.append({"backend": f"dxnn:{variant}", "error": str(exc)})

    report = {
        "image_shape": list(rgb.shape),
        "score_thresh": args.score_thresh,
        "summaries": summaries,
        "comparisons_vs_tflite": {
            name: _compare_outputs(raw_by_backend["tflite"], raw_outputs)
            for name, raw_outputs in raw_by_backend.items()
            if name != "tflite" and "tflite" in raw_by_backend
        },
        "errors": errors,
    }

    print(json.dumps(report, indent=2), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 1 if not summaries else 0


if __name__ == "__main__":
    raise SystemExit(main())
