#!/usr/bin/env python3
"""Fit an NPU→CPU landmark correction from the replay dataset.

The current NPU hand landmark model is INT8 quantized.  On the same palm ROI,
its landmarks are consistently biased relative to the float32 TFLite CPU
baseline.  This tool learns a small per-hand-label/per-landmark affine xy
mapping:

    [x_cpu]   [a b c] [x_npu]
    [y_cpu] = [d e f] [y_npu]
                     [  1  ]

The output JSON can be passed to runtime/benchmark with:

    --landmark-correction models/npu_landmark_correction.dataset.json

This is a dataset-specific calibration, not a replacement for a better INT8
quantization dataset/model.  Use held-out captures before enabling it in demos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from tools import benchmark_dataset as bd  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit NPU landmark correction JSON.")
    p.add_argument("--dataset", type=str, default="dataset")
    p.add_argument("--glob", type=str, default="frame_*.png")
    p.add_argument(
        "--output",
        type=str,
        default="models/npu_landmark_correction.dataset.json",
        help="Correction JSON output path.",
    )
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2))
    p.add_argument("--model-complexity", type=int, default=0, choices=(0, 1))
    p.add_argument("--dxnn", type=str, default="")
    p.add_argument("--dxnn-layout", type=str, default="")
    p.add_argument("--palm-tflite", type=str, default="")
    p.add_argument("--hand-tflite", type=str, default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--kind",
        choices=("affine", "bias"),
        default="affine",
        help="affine=2x3 per landmark; bias=identity+bias per landmark.",
    )
    p.add_argument("--ridge", type=float, default=1e-4, help="Ridge regularization for affine.")
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--print-json", action="store_true", help="Also print summary JSON to stdout.")
    return p.parse_args()


def _make_tracker_args(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.max_hands = args.max_hands
    ns.model_complexity = args.model_complexity
    ns.dxnn = args.dxnn
    ns.dxnn_layout = args.dxnn_layout
    ns.palm_tflite = args.palm_tflite
    ns.palm_dxnn = ""
    ns.hand_tflite = args.hand_tflite
    ns.landmark_correction = ""
    ns.palm_redetect_every = 0
    ns.async_palm = False
    ns.warmup = args.warmup
    ns.runs = 1
    ns.frame_interval_ms = 0.0
    return ns


def _identity_matrix() -> np.ndarray:
    return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _fit_bias(samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    if not samples:
        return _identity_matrix(), 0.0
    deltas = np.asarray([ref - test for ref, test in samples], dtype=np.float64)
    bias = deltas.mean(axis=0)
    mat = _identity_matrix()
    mat[0, 2] = float(bias[0])
    mat[1, 2] = float(bias[1])
    pred = np.asarray([test + bias for ref, test in samples], dtype=np.float64)
    ref = np.asarray([ref for ref, _test in samples], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.sum((pred - ref) ** 2, axis=1))))
    return mat, rmse


def _fit_affine(
    samples: list[tuple[np.ndarray, np.ndarray]],
    *,
    ridge: float,
    min_samples: int,
) -> tuple[np.ndarray, float, str]:
    if len(samples) < min_samples:
        mat, rmse = _fit_bias(samples)
        return mat, rmse, "bias_fallback"

    # X maps NPU/test xy to CPU/reference xy.
    X = np.asarray([[test[0], test[1], 1.0] for _ref, test in samples], dtype=np.float64)
    Y = np.asarray([ref for ref, _test in samples], dtype=np.float64)
    reg = np.diag([float(ridge), float(ridge), 0.0])
    try:
        beta = np.linalg.solve(X.T @ X + reg, X.T @ Y)  # 3x2
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    pred = X @ beta
    rmse = float(np.sqrt(np.mean(np.sum((pred - Y) ** 2, axis=1))))
    # Store as 2x3 for runtime.
    mat = np.asarray(
        [
            [beta[0, 0], beta[1, 0], beta[2, 0]],
            [beta[0, 1], beta[1, 1], beta[2, 1]],
        ],
        dtype=np.float64,
    )
    return mat, rmse, "affine"


def _apply_correction(
    hands_by_frame: list[list[dict[str, Any]]],
    correction: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    labels = correction["labels"]
    out: list[list[dict[str, Any]]] = []
    for hands in hands_by_frame:
        new_hands: list[dict[str, Any]] = []
        for h in hands:
            label = str(h["label"])
            arr = np.asarray(h["landmarks"], dtype=np.float32).copy()
            transforms = labels.get(label) or labels.get("__all__")
            if transforms:
                for j, item in enumerate(transforms[: arr.shape[0]]):
                    m = np.asarray(item["matrix"], dtype=np.float64)
                    x, y = float(arr[j, 0]), float(arr[j, 1])
                    arr[j, 0] = m[0, 0] * x + m[0, 1] * y + m[0, 2]
                    arr[j, 1] = m[1, 0] * x + m[1, 1] * y + m[1, 2]
            new_hands.append({"label": label, "landmarks": arr})
        out.append(new_hands)
    return out


def fit_correction(
    ref_hands: list[list[dict[str, Any]]],
    test_hands: list[list[dict[str, Any]]],
    *,
    kind: str,
    ridge: float,
    min_samples: int,
) -> dict[str, Any]:
    labels = ("Right", "Left")
    out_labels: dict[str, list[dict[str, Any]]] = {}

    for label in labels:
        transforms: list[dict[str, Any]] = []
        for lm_idx in range(21):
            samples: list[tuple[np.ndarray, np.ndarray]] = []
            for ref_frame, test_frame in zip(ref_hands, test_hands):
                ref_by = bd._first_by_label(ref_frame)
                test_by = bd._first_by_label(test_frame)
                if label not in ref_by or label not in test_by:
                    continue
                ref_xy = np.asarray(ref_by[label]["landmarks"][lm_idx, :2], dtype=np.float64)
                test_xy = np.asarray(test_by[label]["landmarks"][lm_idx, :2], dtype=np.float64)
                samples.append((ref_xy, test_xy))
            if kind == "bias":
                mat, rmse = _fit_bias(samples)
                fit_kind = "bias"
            else:
                mat, rmse, fit_kind = _fit_affine(samples, ridge=ridge, min_samples=min_samples)
            transforms.append(
                {
                    "landmark": lm_idx,
                    "matrix": mat.tolist(),
                    "n": len(samples),
                    "rmse": rmse,
                    "fit": fit_kind,
                }
            )
        out_labels[label] = transforms

    return {
        "version": 1,
        "type": "affine_xy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_backend": "cpu-baseline",
        "source_backend": "npu-full",
        "coordinate_space": "image_normalized_xy",
        "fit_requested": kind,
        "ridge": ridge,
        "min_samples": min_samples,
        "labels": out_labels,
    }


def _print_landmark_comparison(title: str, comparison: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print("backend          hand      n    mean   tips   max(avg)  max")
    for backend, by_label in comparison.items():
        for label in ("Right", "Left"):
            mean_s = by_label[label]["mean"]
            tips_s = by_label[label]["tips"]
            max_s = by_label[label]["max"]
            if mean_s["n"] == 0:
                continue
            print(
                f"{backend:<14} {label:<6} {mean_s['n']:>4} "
                f"{(mean_s['mean'] or 0.0):>7.4f} "
                f"{(tips_s['mean'] or 0.0):>6.4f} "
                f"{(max_s['mean'] or 0.0):>8.4f} "
                f"{(max_s['max'] or 0.0):>6.4f}"
            )


def main() -> int:
    args = parse_args()
    paths = bd._paths_for_dataset(args.dataset, args.glob, args.limit)
    frames = [bd._load_rgb(p) for p in paths]
    tracker_args = _make_tracker_args(args)

    print(f"[calibrate] frames={len(paths)} kind={args.kind} ridge={args.ridge}")
    print("[calibrate] running cpu-baseline ...", flush=True)
    _ref_records, ref_hands = bd.run_backend("cpu-baseline", paths, frames, tracker_args)
    print("[calibrate] running npu-full ...", flush=True)
    _test_records, test_hands = bd.run_backend("npu-full", paths, frames, tracker_args)

    before = bd.compare_landmarks(
        "cpu-baseline",
        {"cpu-baseline": ref_hands, "npu-full": test_hands},
    )
    correction = fit_correction(
        ref_hands,
        test_hands,
        kind=args.kind,
        ridge=args.ridge,
        min_samples=args.min_samples,
    )
    corrected_hands = _apply_correction(test_hands, correction)
    after = bd.compare_landmarks(
        "cpu-baseline",
        {"cpu-baseline": ref_hands, "npu-full": corrected_hands},
    )

    correction["training_frames"] = [p.name for p in paths]
    correction["training_summary"] = {
        "before": before,
        "after": after,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(correction, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calibrate] wrote {out}")

    _print_landmark_comparison("Landmark error before correction", before)
    _print_landmark_comparison("Landmark error after correction", after)

    if args.print_json:
        print(json.dumps(correction["training_summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
