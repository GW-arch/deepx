"""
MediaPipe PalmDetectionCpu 그래프 상수 (palm_detection_cpu.pbtxt 와 동일).

후속 단계(tools/palm_decode.py)에서 앵커·스코어·박스 디코드에 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- ImageToTensorCalculator (palm 입력) ---
PALM_INPUT_W = 192
PALM_INPUT_H = 192
PALM_INPUT_FLOAT_MIN = 0.0
PALM_INPUT_FLOAT_MAX = 1.0

# --- SsdAnchorsCalculator ---
SSD_NUM_LAYERS = 4
SSD_MIN_SCALE = 0.1484375
SSD_MAX_SCALE = 0.75
SSD_INPUT_SIZE_W = 192
SSD_INPUT_SIZE_H = 192
SSD_ANCHOR_OFFSET_X = 0.5
SSD_ANCHOR_OFFSET_Y = 0.5
# pbtxt: strides: 8, 16, 16, 16
SSD_STRIDES = (8, 16, 16, 16)
SSD_ASPECT_RATIOS = (1.0,)
SSD_FIXED_ANCHOR_SIZE = True

# --- TensorsToDetectionsCalculator ---
DET_NUM_CLASSES = 1
DET_NUM_BOXES = 2016
DET_NUM_COORDS = 18
DET_BOX_COORD_OFFSET = 0
DET_KEYPOINT_COORD_OFFSET = 4
DET_NUM_KEYPOINTS = 7
DET_NUM_VALUES_PER_KEYPOINT = 2
DET_SIGMOID_SCORE = True
DET_SCORE_CLIPPING_THRESH = 100.0
DET_REVERSE_OUTPUT_ORDER = True
DET_X_SCALE = 192.0
DET_Y_SCALE = 192.0
DET_W_SCALE = 192.0
DET_H_SCALE = 192.0
DET_MIN_SCORE_THRESH_DEFAULT = 0.5

# --- NonMaxSuppressionCalculator ---
NMS_MIN_SUPPRESSION_THRESHOLD = 0.3
NMS_OVERLAP_TYPE_IOU = "INTERSECTION_OVER_UNION"
NMS_ALGORITHM_WEIGHTED = "WEIGHTED"


@dataclass(frozen=True)
class LetterboxPadding:
    """DetectionLetterboxRemovalCalculator 가 기대하는 패딩(정규화 좌표 기준 등)용 메타."""

    pad_x_norm: float
    pad_y_norm: float
    scale: float
    new_w: int
    new_h: int
