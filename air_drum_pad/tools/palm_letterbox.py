#!/usr/bin/env python3
"""
MediaPipe PalmDetectionCpu 의 ImageToTensorCalculator 에 맞춘 전처리.

- 출력: float32 NHWC [1, H, W, 3], 값域 [0, 1]
- keep_aspect_ratio + zero pad (BORDER_ZERO)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_tools_dir = str(Path(__file__).resolve().parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from palm_mp_spec import PALM_INPUT_H, PALM_INPUT_W, LetterboxPadding


def rgb_uint8_to_palm_input_tensor(
    rgb: np.ndarray,
    *,
    out_h: int = PALM_INPUT_H,
    out_w: int = PALM_INPUT_W,
) -> Tuple[np.ndarray, LetterboxPadding]:
    """
    rgb: H×W×3 uint8, RGB 순서.
    반환: (tensor, meta) — tensor shape (1, out_h, out_w, 3) float32
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    scale = min(out_w / float(iw), out_h / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized.astype(np.float32) * (1.0 / 255.0)
    tensor = np.expand_dims(canvas, axis=0)
    meta = LetterboxPadding(
        pad_x_norm=pad_x / float(out_w),
        pad_y_norm=pad_y / float(out_h),
        scale=scale,
        new_w=new_w,
        new_h=new_h,
    )
    return tensor, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=-1, help=">=0 이면 한 프레임 캡처 후 shape 만 출력")
    args = ap.parse_args()
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            print("camera read failed", flush=True)
            return 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t, meta = rgb_uint8_to_palm_input_tensor(rgb)
    print("tensor", t.shape, t.dtype, float(t.min()), float(t.max()), flush=True)
    print("letterbox meta", meta, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
