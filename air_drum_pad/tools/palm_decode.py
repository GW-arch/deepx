"""
Palm SSD 출력 → Detection 디코드 (MediaPipe TensorsToDetections 규격).

Phase 1 구현: 앵커 생성 · box/keypoint 디코드 · weighted NMS · letterbox 제거.
상수는 `palm_mp_spec.py` 를 사용한다.

Detection 배열 레이아웃 (행당 19 float):
  [score, ymin, xmin, ymax, xmax, kp0_x, kp0_y, kp1_x, kp1_y, ..., kp6_x, kp6_y]
모든 좌표는 정규화 [0, 1].  letterbox 제거 후 원본 이미지 기준.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_tools_dir = str(Path(__file__).resolve().parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from palm_mp_spec import (
    DET_BOX_COORD_OFFSET,
    DET_KEYPOINT_COORD_OFFSET,
    DET_MIN_SCORE_THRESH_DEFAULT,
    DET_NUM_BOXES,
    DET_NUM_COORDS,
    DET_NUM_KEYPOINTS,
    DET_NUM_VALUES_PER_KEYPOINT,
    DET_REVERSE_OUTPUT_ORDER,
    DET_SCORE_CLIPPING_THRESH,
    DET_SIGMOID_SCORE,
    DET_X_SCALE,
    DET_Y_SCALE,
    DET_W_SCALE,
    DET_H_SCALE,
    NMS_MIN_SUPPRESSION_THRESHOLD,
    SSD_ANCHOR_OFFSET_X,
    SSD_ANCHOR_OFFSET_Y,
    SSD_ASPECT_RATIOS,
    SSD_FIXED_ANCHOR_SIZE,
    SSD_INPUT_SIZE_H,
    SSD_INPUT_SIZE_W,
    SSD_MAX_SCALE,
    SSD_MIN_SCALE,
    SSD_NUM_LAYERS,
    SSD_STRIDES,
    LetterboxPadding,
)

# ── Detection array layout ──────────────────────────────────────────
DET_SCORE_IDX = 0
DET_YMIN_IDX = 1
DET_XMIN_IDX = 2
DET_YMAX_IDX = 3
DET_XMAX_IDX = 4
DET_KP_OFFSET = 5  # (x, y) pairs start here
DET_ROW_SIZE = 5 + DET_NUM_KEYPOINTS * 2  # 19


# ── Anchor generation (SsdAnchorsCalculator) ────────────────────────

def _calculate_scale(
    min_scale: float, max_scale: float, stride_index: int, num_strides: int,
) -> float:
    if num_strides == 1:
        return (min_scale + max_scale) * 0.5
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)


def generate_ssd_anchors(
    *,
    num_layers: int = SSD_NUM_LAYERS,
    min_scale: float = SSD_MIN_SCALE,
    max_scale: float = SSD_MAX_SCALE,
    input_w: int = SSD_INPUT_SIZE_W,
    input_h: int = SSD_INPUT_SIZE_H,
    anchor_offset_x: float = SSD_ANCHOR_OFFSET_X,
    anchor_offset_y: float = SSD_ANCHOR_OFFSET_Y,
    strides: Tuple[int, ...] = SSD_STRIDES,
    aspect_ratios: Tuple[float, ...] = SSD_ASPECT_RATIOS,
    fixed_anchor_size: bool = SSD_FIXED_ANCHOR_SIZE,
    interpolated_scale_aspect_ratio: float = 1.0,
    reduce_boxes_in_lowest_layer: bool = False,
) -> np.ndarray:
    """MediaPipe SsdAnchorsCalculator 와 동일한 앵커 생성.

    Returns (N, 4) float32 — columns (y_center, x_center, h, w), 정규화 [0, 1].
    Palm detection 기본 설정에서 N = 2016.
    """
    strides_size = len(strides)
    anchors: list[list[float]] = []
    layer_id = 0

    while layer_id < strides_size:
        last = layer_id
        ar_list: list[float] = []
        sc_list: list[float] = []

        # 같은 stride 를 공유하는 연속 레이어를 하나의 그룹으로 묶는다.
        while last < strides_size and strides[last] == strides[layer_id]:
            scale = _calculate_scale(min_scale, max_scale, last, strides_size)

            if last == 0 and reduce_boxes_in_lowest_layer:
                ar_list.append(1.0)
                sc_list.append(0.1)
                ar_list.append(1.0)
                sc_list.append(scale)
            else:
                for ar in aspect_ratios:
                    ar_list.append(ar)
                    sc_list.append(scale)
                if interpolated_scale_aspect_ratio > 0.0:
                    scale_next = (
                        1.0
                        if last == strides_size - 1
                        else _calculate_scale(min_scale, max_scale, last + 1, strides_size)
                    )
                    ar_list.append(interpolated_scale_aspect_ratio)
                    sc_list.append(math.sqrt(scale * scale_next))
            last += 1

        stride = strides[layer_id]
        feat_h = math.ceil(input_h / stride)
        feat_w = math.ceil(input_w / stride)

        for y in range(feat_h):
            for x in range(feat_w):
                for i in range(len(ar_list)):
                    cx = (x + anchor_offset_x) / feat_w
                    cy = (y + anchor_offset_y) / feat_h
                    if fixed_anchor_size:
                        w = h = 1.0
                    else:
                        ratio_sqrt = math.sqrt(ar_list[i])
                        w = sc_list[i] * ratio_sqrt
                        h = sc_list[i] / ratio_sqrt
                    anchors.append([cy, cx, h, w])

        layer_id = last

    result = np.array(anchors, dtype=np.float32)
    assert result.shape == (DET_NUM_BOXES, 4), (
        f"Expected {DET_NUM_BOXES} anchors, got {result.shape[0]}"
    )
    return result


# ── Box / keypoint decode (TensorsToDetectionsCalculator) ───────────

def _decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """raw regression (N, 18) + anchors (N, 4) → decoded (N, 18).

    출력 열: [ymin, xmin, ymax, xmax, kp0_x, kp0_y, …, kp6_x, kp6_y]
    """
    a_cy, a_cx, a_h, a_w = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]

    y_ctr = raw_boxes[:, DET_BOX_COORD_OFFSET + 0] / DET_Y_SCALE * a_h + a_cy
    x_ctr = raw_boxes[:, DET_BOX_COORD_OFFSET + 1] / DET_X_SCALE * a_w + a_cx
    h = raw_boxes[:, DET_BOX_COORD_OFFSET + 2] / DET_H_SCALE * a_h
    w = raw_boxes[:, DET_BOX_COORD_OFFSET + 3] / DET_W_SCALE * a_w

    ymin = y_ctr - h * 0.5
    xmin = x_ctr - w * 0.5
    ymax = y_ctr + h * 0.5
    xmax = x_ctr + w * 0.5

    n = raw_boxes.shape[0]
    kps = np.empty((n, DET_NUM_KEYPOINTS * 2), dtype=np.float32)
    for k in range(DET_NUM_KEYPOINTS):
        off = DET_KEYPOINT_COORD_OFFSET + k * DET_NUM_VALUES_PER_KEYPOINT
        kps[:, k * 2 + 0] = raw_boxes[:, off + 1] / DET_X_SCALE * a_w + a_cx  # x
        kps[:, k * 2 + 1] = raw_boxes[:, off + 0] / DET_Y_SCALE * a_h + a_cy  # y

    return np.column_stack([ymin, xmin, ymax, xmax, kps])


# ── Public API ──────────────────────────────────────────────────────

def decode_palm_tensors(
    raw_outputs: List[np.ndarray],
    anchors: np.ndarray,
    *,
    letterbox_meta: Optional[LetterboxPadding] = None,
    score_thresh: float = DET_MIN_SCORE_THRESH_DEFAULT,
    nms_thresh: float = NMS_MIN_SUPPRESSION_THRESHOLD,
) -> np.ndarray:
    """Raw TFLite / dxnn 출력 → (K, 19) 필터링된 detection 배열.

    Parameters
    ----------
    raw_outputs : 모델 출력 텐서 리스트 (보통 2 개).
        reverse_output_order=True 이면 [scores, boxes].
    anchors : (2016, 4) — ``generate_ssd_anchors()`` 결과.
    letterbox_meta : 있으면 좌표를 원본 이미지 정규 [0, 1] 로 변환.
    score_thresh : 최소 confidence.
    nms_thresh : NMS IoU 임계값.

    Returns
    -------
    (K, 19) float32.  K = 0 이면 탐지 없음.
    """
    # shape 기반으로 score / box 텐서를 식별 (TFLite output 순서가 일정하지 않으므로).
    raw_scores = raw_boxes = None
    for t in raw_outputs:
        arr = t[0] if t.ndim == 3 else t  # batch 차원 제거
        if arr.ndim == 2 and arr.shape[-1] == DET_NUM_COORDS:
            raw_boxes = arr
        elif arr.ndim == 2 and arr.shape[-1] <= 2:
            raw_scores = arr
        elif arr.ndim == 1:
            raw_scores = arr
    # fallback: reverse_output_order 플래그 사용
    if raw_scores is None or raw_boxes is None:
        if DET_REVERSE_OUTPUT_ORDER:
            t0, t1 = raw_outputs[0], raw_outputs[-1]
        else:
            t0, t1 = raw_outputs[1], raw_outputs[0]
        raw_scores = t0[0] if t0.ndim == 3 else t0
        raw_boxes = t1[0] if t1.ndim == 3 else t1

    # score sigmoid
    if DET_SIGMOID_SCORE:
        raw_scores = np.clip(raw_scores, -DET_SCORE_CLIPPING_THRESH, DET_SCORE_CLIPPING_THRESH)
        scores = (1.0 / (1.0 + np.exp(-raw_scores))).ravel()
    else:
        scores = raw_scores.ravel()

    mask = scores >= score_thresh
    if not np.any(mask):
        return np.zeros((0, DET_ROW_SIZE), dtype=np.float32)

    decoded = _decode_boxes(raw_boxes[mask], anchors[mask])
    detections = np.column_stack([scores[mask], decoded])

    detections = weighted_nms(detections, iou_thresh=nms_thresh)

    if letterbox_meta is not None:
        detections = remove_letterbox(detections, letterbox_meta)

    return detections


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [ymin, xmin, ymax, xmax] boxes."""
    y1 = max(float(a[0]), float(b[0]))
    x1 = max(float(a[1]), float(b[1]))
    y2 = min(float(a[2]), float(b[2]))
    x2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_nms(
    detections: np.ndarray,
    *,
    iou_thresh: float = NMS_MIN_SUPPRESSION_THRESHOLD,
) -> np.ndarray:
    """MediaPipe NonMaxSuppressionCalculator (WEIGHTED) 와 동일.

    겹치는 detection 을 score 가중 평균으로 병합한다.
    """
    if detections.shape[0] == 0:
        return detections

    order = np.argsort(-detections[:, DET_SCORE_IDX])
    dets = detections[order].copy()
    remaining = list(range(len(dets)))
    output: list[np.ndarray] = []

    while remaining:
        best = remaining[0]
        best_box = dets[best, DET_YMIN_IDX : DET_XMAX_IDX + 1]
        overlap_idx = [best]
        non_overlap = []
        for idx in remaining[1:]:
            if _iou(best_box, dets[idx, DET_YMIN_IDX : DET_XMAX_IDX + 1]) >= iou_thresh:
                overlap_idx.append(idx)
            else:
                non_overlap.append(idx)

        cluster = dets[overlap_idx]
        w = cluster[:, DET_SCORE_IDX]
        total = w.sum()
        merged = np.empty(DET_ROW_SIZE, dtype=np.float32)
        merged[DET_SCORE_IDX] = dets[best, DET_SCORE_IDX]
        if total > 0:
            merged[DET_YMIN_IDX:] = (cluster[:, DET_YMIN_IDX:] * w[:, None]).sum(axis=0) / total
        else:
            merged[DET_YMIN_IDX:] = dets[best, DET_YMIN_IDX:]
        output.append(merged)
        remaining = non_overlap

    if not output:
        return np.zeros((0, DET_ROW_SIZE), dtype=np.float32)
    return np.stack(output)


def remove_letterbox(
    detections: np.ndarray,
    meta: LetterboxPadding,
) -> np.ndarray:
    """Letterbox 패딩 좌표를 원본 이미지 정규 [0, 1] 로 되돌린다."""
    if detections.shape[0] == 0:
        return detections

    cw = 1.0 - 2.0 * meta.pad_x_norm
    ch = 1.0 - 2.0 * meta.pad_y_norm
    if cw <= 0 or ch <= 0:
        return detections

    out = detections.copy()
    out[:, DET_YMIN_IDX] = (out[:, DET_YMIN_IDX] - meta.pad_y_norm) / ch
    out[:, DET_YMAX_IDX] = (out[:, DET_YMAX_IDX] - meta.pad_y_norm) / ch
    out[:, DET_XMIN_IDX] = (out[:, DET_XMIN_IDX] - meta.pad_x_norm) / cw
    out[:, DET_XMAX_IDX] = (out[:, DET_XMAX_IDX] - meta.pad_x_norm) / cw
    for k in range(DET_NUM_KEYPOINTS):
        xi = DET_KP_OFFSET + k * 2
        yi = DET_KP_OFFSET + k * 2 + 1
        out[:, xi] = (out[:, xi] - meta.pad_x_norm) / cw
        out[:, yi] = (out[:, yi] - meta.pad_y_norm) / ch
    return out


# ── CLI smoke test / 시각화 ─────────────────────────────────────────

def _self_test() -> None:
    """앵커 생성 기본 검증 (외부 의존성 없음)."""
    anchors = generate_ssd_anchors()
    print(f"anchors: {anchors.shape} {anchors.dtype}", flush=True)
    print(f"  cy  range [{anchors[:, 0].min():.4f}, {anchors[:, 0].max():.4f}]", flush=True)
    print(f"  cx  range [{anchors[:, 1].min():.4f}, {anchors[:, 1].max():.4f}]", flush=True)
    print(f"  h   range [{anchors[:, 2].min():.4f}, {anchors[:, 2].max():.4f}]", flush=True)
    print(f"  w   range [{anchors[:, 3].min():.4f}, {anchors[:, 3].max():.4f}]", flush=True)

    # NMS smoke: 두 개의 겹치는 detection
    d = np.zeros((2, DET_ROW_SIZE), dtype=np.float32)
    d[0] = [0.9, 0.1, 0.1, 0.5, 0.5] + [0.0] * (DET_ROW_SIZE - 5)
    d[1] = [0.7, 0.12, 0.12, 0.52, 0.52] + [0.0] * (DET_ROW_SIZE - 5)
    merged = weighted_nms(d, iou_thresh=0.3)
    print(f"NMS: {d.shape[0]} in → {merged.shape[0]} out, score={merged[0, 0]:.2f}", flush=True)

    # Letterbox removal smoke
    meta = LetterboxPadding(pad_x_norm=0.0, pad_y_norm=0.125, scale=1.0, new_w=192, new_h=144)
    det = np.array([[0.9, 0.2, 0.1, 0.8, 0.9] + [0.5, 0.5] * DET_NUM_KEYPOINTS], dtype=np.float32)
    out = remove_letterbox(det, meta)
    print(f"Letterbox removal: ymin {det[0, 1]:.2f}→{out[0, 1]:.2f}, "
          f"ymax {det[0, 3]:.2f}→{out[0, 3]:.2f}", flush=True)
    print("self-test OK", flush=True)


def _run_tflite(tflite_path: Path, camera: int) -> None:
    """TFLite 인터프리터로 실제 추론 후 시각화."""
    import cv2
    from palm_letterbox import rgb_uint8_to_palm_input_tensor

    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            print("tensorflow 또는 tflite_runtime 필요", file=sys.stderr, flush=True)
            return

    intr = Interpreter(model_path=str(tflite_path))
    intr.allocate_tensors()
    inp_detail = intr.get_input_details()[0]
    out_details = intr.get_output_details()
    print(f"model input: {inp_detail['shape']} {inp_detail['dtype']}", flush=True)
    for i, o in enumerate(out_details):
        print(f"model output[{i}]: {o['shape']} {o['dtype']}", flush=True)

    # 프레임 획득
    if camera >= 0:
        cap = cv2.VideoCapture(camera)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            print("camera read failed", file=sys.stderr, flush=True)
            return
    else:
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ih, iw = rgb.shape[:2]

    tensor, meta = rgb_uint8_to_palm_input_tensor(rgb)
    if inp_detail["dtype"] in (np.float32, "float32"):
        tensor = tensor.astype(np.float32)
    elif inp_detail["dtype"] in (np.uint8, "uint8"):
        tensor = (tensor * 255.0).clip(0, 255).astype(np.uint8)

    intr.set_tensor(inp_detail["index"], tensor)
    intr.invoke()
    raw_outputs = [intr.get_tensor(o["index"]) for o in out_details]

    anchors = generate_ssd_anchors()
    detections = decode_palm_tensors(
        raw_outputs, anchors, letterbox_meta=meta, score_thresh=0.5,
    )
    print(f"detections: {detections.shape[0]}", flush=True)

    # 시각화
    vis = bgr.copy()
    for det in detections:
        sc = det[DET_SCORE_IDX]
        y1 = int(det[DET_YMIN_IDX] * ih)
        x1 = int(det[DET_XMIN_IDX] * iw)
        y2 = int(det[DET_YMAX_IDX] * ih)
        x2 = int(det[DET_XMAX_IDX] * iw)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{sc:.2f}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for k in range(DET_NUM_KEYPOINTS):
            kx = int(det[DET_KP_OFFSET + k * 2] * iw)
            ky = int(det[DET_KP_OFFSET + k * 2 + 1] * ih)
            cv2.circle(vis, (kx, ky), 3, (0, 0, 255), -1)

    out_path = Path("palm_decode_smoke.png")
    cv2.imwrite(str(out_path), vis)
    print(f"saved: {out_path.resolve()}", flush=True)

    cv2.imshow("palm decode", vis)
    print("press any key to close", flush=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="Palm decode smoke test")
    ap.add_argument("--self-test", action="store_true", help="앵커·NMS·letterbox 기본 검증만 실행")
    ap.add_argument("--tflite", type=Path, default=None,
                    help="palm_detection_lite.tflite 경로 (없으면 self-test)")
    ap.add_argument("--camera", type=int, default=-1,
                    help=">=0 이면 카메라 프레임 사용, 아니면 랜덤 이미지")
    args = ap.parse_args()

    _self_test()

    if args.self_test:
        return 0

    tflite = args.tflite
    if tflite is None:
        default = Path(__file__).resolve().parents[1] / "models" / "vendor" / "palm_detection_lite.tflite"
        if default.is_file():
            tflite = default

    if tflite is not None and tflite.is_file():
        _run_tflite(tflite, args.camera)
    else:
        print("tflite 모델 없음 — self-test 만 실행됨", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
