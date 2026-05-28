"""
MediaPipe 스타일 palm detection → hand ROI 변환.

Palm 7 keypoints → 회전·확대한 hand bounding box → affine warp 224×224.
MediaPipe `hand_landmark_tracking` 그래프의 RectTransformation 과 동일 로직.

Palm keypoint 인덱스 (MediaPipe palm_detection_cpu.pbtxt):
  0: wrist center
  2: middle_finger_mcp
이 두 점으로 **손 방향 각도**(회전)를 결정.

References:
  mediapipe/calculators/util/rect_transformation_calculator.cc
  mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe hand_detection_to_roi constants
PALM_KP_WRIST = 0
PALM_KP_MIDDLE_MCP = 2
ROI_SCALE_X = 2.6   # rect_transformation scale_x
ROI_SCALE_Y = 2.6   # rect_transformation scale_y
ROI_SHIFT_Y = -0.5   # rect_transformation shift_y (shift up along hand axis)
HAND_LANDMARK_SIZE = 224


def palm_detection_to_roi(
    detection: np.ndarray,
    image_w: int,
    image_h: int,
    *,
    target_size: int = HAND_LANDMARK_SIZE,
    scale_x: float = ROI_SCALE_X,
    scale_y: float = ROI_SCALE_Y,
    shift_y: float = ROI_SHIFT_Y,
) -> Tuple[np.ndarray, float, float, float]:
    """Palm detection (19-col) → hand ROI params.

    Parameters
    ----------
    detection : (19,) float — palm_decode detection row.
    image_w, image_h : original image pixel size.
    target_size : output square side (224).

    Returns
    -------
    (center_x_px, center_y_px, box_size_px, rotation_rad)
    All in pixel space of the original image.
    """
    from palm_decode import DET_KP_OFFSET

    # Extract wrist and middle-finger-MCP keypoints (normalized [0,1])
    wrist_x = float(detection[DET_KP_OFFSET + PALM_KP_WRIST * 2])
    wrist_y = float(detection[DET_KP_OFFSET + PALM_KP_WRIST * 2 + 1])
    mf_x = float(detection[DET_KP_OFFSET + PALM_KP_MIDDLE_MCP * 2])
    mf_y = float(detection[DET_KP_OFFSET + PALM_KP_MIDDLE_MCP * 2 + 1])

    # Rotation: MediaPipe aligns the wrist→middle-MCP vector to the ROI Y axis.
    # Equivalent to DetectionsToRectsCalculator with
    # rotation_vector_target_angle_degrees: 90.  Image y grows downward, hence
    # the negated dy inside atan2.  The previous atan2(dy, dx) - pi/2 form was
    # 180° off for an upright hand, causing the hand-landmark model to see an
    # upside-down crop and produce poor fingertip endpoints.
    dx = (mf_x - wrist_x) * image_w
    dy = (mf_y - wrist_y) * image_h
    rotation = math.pi / 2.0 - math.atan2(-dy, dx)

    # Box from detection bounding box (ymin, xmin, ymax, xmax)
    from palm_decode import DET_YMIN_IDX, DET_XMIN_IDX, DET_YMAX_IDX, DET_XMAX_IDX

    ymin = float(detection[DET_YMIN_IDX]) * image_h
    xmin = float(detection[DET_XMIN_IDX]) * image_w
    ymax = float(detection[DET_YMAX_IDX]) * image_h
    xmax = float(detection[DET_XMAX_IDX]) * image_w

    box_cx = (xmin + xmax) * 0.5
    box_cy = (ymin + ymax) * 0.5
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Apply RectTransformation-style shift before scaling/squaring.  shift_y is
    # expressed in detection-box height and negative values move toward the
    # fingers for an upright hand.
    center_x = box_cx - box_h * shift_y * math.sin(rotation)
    center_y = box_cy + box_h * shift_y * math.cos(rotation)

    # Apply square_long + scale_x/scale_y.
    long_side = max(box_w, box_h)
    roi_size = long_side * max(scale_x, scale_y)

    return center_x, center_y, roi_size, rotation


def warp_roi_affine(
    rgb: np.ndarray,
    center_x: float,
    center_y: float,
    roi_size: float,
    rotation: float,
    *,
    out_size: int = HAND_LANDMARK_SIZE,
) -> np.ndarray:
    """Affine-warp a rotated square ROI from the image into (out_size, out_size, 3).

    Parameters
    ----------
    rgb : H×W×3 uint8 source image.
    center_x, center_y : ROI center in pixel coords.
    roi_size : side length of the square ROI in pixels.
    rotation : clockwise rotation in radians.
    out_size : output square side (224).

    Returns
    -------
    (out_size, out_size, 3) uint8 — the warped hand patch.
    """
    half = roi_size * 0.5
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    # Source corners of the rotated ROI (top-left, top-right, bottom-left)
    # in the original image coordinate system
    src_pts = np.array([
        [center_x - half * cos_r - (-half) * sin_r,
         center_y - half * sin_r + (-half) * cos_r],  # top-left
        [center_x + half * cos_r - (-half) * sin_r,
         center_y + half * sin_r + (-half) * cos_r],  # top-right
        [center_x - half * cos_r - half * sin_r,
         center_y - half * sin_r + half * cos_r],      # bottom-left
    ], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [out_size, 0],
        [0, out_size],
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(
        rgb, M, (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped


def inverse_landmark_transform(
    landmarks_norm: np.ndarray,
    center_x: float,
    center_y: float,
    roi_size: float,
    rotation: float,
    image_w: int,
    image_h: int,
    *,
    roi_input_size: int = HAND_LANDMARK_SIZE,
) -> np.ndarray:
    """Map 21-landmark normalized coords from ROI patch back to original image [0,1].

    Parameters
    ----------
    landmarks_norm : (21, 3) or (63,) — x, y, z in [0, 1] relative to ROI patch.
    center_x, center_y, roi_size, rotation : ROI params from palm_detection_to_roi.
    image_w, image_h : original image size.

    Returns
    -------
    (21, 3) float32 — x, y in [0, 1] of original image, z preserved.
    """
    lm = np.asarray(landmarks_norm, dtype=np.float32).reshape(-1, 3).copy()
    half = roi_size * 0.5
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    for i in range(lm.shape[0]):
        # ROI-normalized → ROI pixel
        rx = lm[i, 0] * roi_size - half
        ry = lm[i, 1] * roi_size - half
        # Rotate back to original image pixel space
        ox = center_x + rx * cos_r - ry * sin_r
        oy = center_y + rx * sin_r + ry * cos_r
        # Normalize to [0, 1]
        lm[i, 0] = ox / image_w
        lm[i, 1] = oy / image_h
        # z: keep as-is (relative depth)

    return lm


def extract_hand_roi(
    rgb: np.ndarray,
    detection: np.ndarray,
    *,
    out_size: int = HAND_LANDMARK_SIZE,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Convenience: palm detection → warped hand ROI patch.

    Returns (patch, center_x, center_y, roi_size, rotation).
    """
    h, w = rgb.shape[:2]
    cx, cy, sz, rot = palm_detection_to_roi(detection, w, h, target_size=out_size)
    patch = warp_roi_affine(rgb, cx, cy, sz, rot, out_size=out_size)
    return patch, cx, cy, sz, rot
