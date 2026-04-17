"""
Palm SSD 출력 → Detection 디코드 (MediaPipe TensorsToDetections 규격).

Phase 1: `docs/PLAN_NPU_FULL_HAND_PIPELINE.md` 참고 — 앵커 생성·NMS·letterbox 제거 구현 예정.
상수는 `palm_mp_spec.py` 를 사용한다.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def generate_ssd_anchors() -> np.ndarray:
    """2016×4 (ycx, x, h, w) 또는 MediaPipe Anchor 포맷 — 미구현."""
    raise NotImplementedError(
        "Phase 1: mediapipe SsdAnchorsCalculator 와 동일한 앵커 생성을 구현하세요. "
        "참고: mediapipe/calculators/tflite/ssd_anchors_calculator.cc"
    )


def decode_palm_tensors(
    raw_outputs: List[np.ndarray],
    anchors: np.ndarray,
    *,
    letterbox_meta: Any,
) -> Tuple[np.ndarray, ...]:
    """raw TFLite / dxnn 출력 → normalized detections — 미구현."""
    raise NotImplementedError("Phase 1: palm_decode.decode_palm_tensors")


def weighted_nms(detections: np.ndarray) -> np.ndarray:
    """미구현."""
    raise NotImplementedError("Phase 1: palm_decode.weighted_nms")
