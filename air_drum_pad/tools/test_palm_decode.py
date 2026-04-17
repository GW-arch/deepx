#!/usr/bin/env python3
"""Phase 1 단위 테스트: 자체 palm_decode vs MediaPipe Hands 내부 palm detection 비교.

MediaPipe Hands 는 palm detection 중간 텐서를 노출하지 않으므로,
최종 hand landmark 결과의 **손목(0번) 좌표**가 자체 palm detection
bounding-box 안에 들어오는지를 기준으로 비교한다.

추가로 기본 불변식(앵커 수, NMS 감소, letterbox 역변환 범위)도 검증한다.

사용법:
    python3 tools/test_palm_decode.py                  # 카메라 없이 self-test
    python3 tools/test_palm_decode.py --camera 0       # 카메라 프레임으로 비교
    python3 tools/test_palm_decode.py --image photo.png # 이미지 파일로 비교
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_tools_dir = str(Path(__file__).resolve().parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from palm_decode import (
    DET_KP_OFFSET,
    DET_NUM_KEYPOINTS,
    DET_ROW_SIZE,
    DET_SCORE_IDX,
    DET_XMAX_IDX,
    DET_XMIN_IDX,
    DET_YMAX_IDX,
    DET_YMIN_IDX,
    decode_palm_tensors,
    generate_ssd_anchors,
    remove_letterbox,
    weighted_nms,
)
from palm_mp_spec import DET_NUM_BOXES, LetterboxPadding

# ── Counters ────────────────────────────────────────────────────────
_pass = 0
_fail = 0


def _ok(name: str) -> None:
    global _pass
    _pass += 1
    print(f"  ✓ {name}")


def _nok(name: str, msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  ✗ {name}: {msg}")


def _check(name: str, cond: bool, msg: str = "") -> None:
    if cond:
        _ok(name)
    else:
        _nok(name, msg)


# ── Self-tests (no external dependencies beyond numpy) ──────────────

def test_anchor_generation() -> None:
    print("[test_anchor_generation]")
    anchors = generate_ssd_anchors()
    _check("anchor_count", anchors.shape == (DET_NUM_BOXES, 4),
           f"expected ({DET_NUM_BOXES},4), got {anchors.shape}")
    _check("anchor_dtype", anchors.dtype == np.float32)
    _check("anchor_y_range", 0.0 < anchors[:, 0].min() and anchors[:, 0].max() < 1.05,
           f"y range [{anchors[:,0].min():.4f}, {anchors[:,0].max():.4f}]")
    _check("anchor_x_range", 0.0 < anchors[:, 1].min() and anchors[:, 1].max() < 1.05,
           f"x range [{anchors[:,1].min():.4f}, {anchors[:,1].max():.4f}]")
    # all h/w should be 1.0 for fixed_anchor_size
    _check("anchor_fixed_size", np.allclose(anchors[:, 2:], 1.0),
           "h/w should all be 1.0 when fixed_anchor_size=True")


def test_nms_merge() -> None:
    print("[test_nms_merge]")
    # Two overlapping detections should merge into one
    d = np.zeros((2, DET_ROW_SIZE), dtype=np.float32)
    d[0, :5] = [0.9, 0.1, 0.1, 0.5, 0.5]
    d[1, :5] = [0.7, 0.12, 0.12, 0.52, 0.52]
    merged = weighted_nms(d, iou_thresh=0.3)
    _check("nms_reduces", merged.shape[0] == 1,
           f"expected 1, got {merged.shape[0]}")
    _check("nms_best_score", abs(merged[0, DET_SCORE_IDX] - 0.9) < 1e-5)

    # Weighted coords should be between the two
    _check("nms_weighted_ymin",
           0.1 <= merged[0, DET_YMIN_IDX] <= 0.12,
           f"ymin={merged[0, DET_YMIN_IDX]:.4f}")


def test_nms_non_overlapping() -> None:
    print("[test_nms_non_overlapping]")
    d = np.zeros((2, DET_ROW_SIZE), dtype=np.float32)
    d[0, :5] = [0.9, 0.0, 0.0, 0.1, 0.1]
    d[1, :5] = [0.7, 0.8, 0.8, 1.0, 1.0]
    merged = weighted_nms(d, iou_thresh=0.3)
    _check("nms_keeps_separate", merged.shape[0] == 2,
           f"expected 2, got {merged.shape[0]}")


def test_letterbox_removal() -> None:
    print("[test_letterbox_removal]")
    # 640x480 → 192x192 letterbox: landscape, pad_y=0.125
    meta = LetterboxPadding(pad_x_norm=0.0, pad_y_norm=0.125, scale=0.3, new_w=192, new_h=144)
    det = np.array([[0.9, 0.25, 0.1, 0.75, 0.9] + [0.5, 0.5] * DET_NUM_KEYPOINTS],
                   dtype=np.float32)
    out = remove_letterbox(det, meta)

    # After removing vertical padding: content spans [0.125, 0.875] in letterbox
    # ymin = (0.25 - 0.125) / 0.75 ≈ 0.1667
    expected_ymin = (0.25 - 0.125) / (1.0 - 2 * 0.125)
    _check("lb_ymin", abs(out[0, DET_YMIN_IDX] - expected_ymin) < 1e-4,
           f"expected {expected_ymin:.4f}, got {out[0, DET_YMIN_IDX]:.4f}")
    # x should be unchanged (pad_x=0)
    _check("lb_xmin_unchanged", abs(out[0, DET_XMIN_IDX] - 0.1) < 1e-4)
    # keypoints
    expected_kp_y = (0.5 - 0.125) / 0.75
    _check("lb_kp_y", abs(out[0, DET_KP_OFFSET + 1] - expected_kp_y) < 1e-4,
           f"expected {expected_kp_y:.4f}, got {out[0, DET_KP_OFFSET + 1]:.4f}")


def test_empty_input() -> None:
    print("[test_empty_input]")
    empty = np.zeros((0, DET_ROW_SIZE), dtype=np.float32)
    result = weighted_nms(empty)
    _check("nms_empty", result.shape == (0, DET_ROW_SIZE))
    meta = LetterboxPadding(pad_x_norm=0.0, pad_y_norm=0.1, scale=1.0, new_w=192, new_h=172)
    result = remove_letterbox(empty, meta)
    _check("lb_empty", result.shape == (0, DET_ROW_SIZE))


def test_decode_random_tensor() -> None:
    """decode_palm_tensors on random noise should return 0 detections (high score_thresh)."""
    print("[test_decode_random_tensor]")
    anchors = generate_ssd_anchors()
    # Simulate model outputs: boxes (1,2016,18), scores (1,2016,1)
    raw_boxes = np.random.randn(1, DET_NUM_BOXES, 18).astype(np.float32) * 0.1
    raw_scores = np.full((1, DET_NUM_BOXES, 1), -10.0, dtype=np.float32)  # sigmoid → ~0
    dets = decode_palm_tensors([raw_boxes, raw_scores], anchors, score_thresh=0.5)
    _check("random_no_detect", dets.shape[0] == 0,
           f"expected 0 detections from noise, got {dets.shape[0]}")


# ── Camera / image test: compare vs MediaPipe Hands ─────────────────

def _load_palm_interpreter(tflite_path: str):
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=tflite_path)
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        return None


def test_vs_mediapipe(rgb: np.ndarray, tflite_path: str) -> None:
    """Compare our palm decode against MediaPipe Hands landmark output.

    MediaPipe 가 검출한 각 손의 wrist(0번) 좌표가 자체 palm bounding-box
    안에 들어오면 pass.  검출 수가 같은지도 확인 (허용 오차 ±1).
    """
    import cv2
    from palm_letterbox import rgb_uint8_to_palm_input_tensor

    print("[test_vs_mediapipe]")
    ih, iw = rgb.shape[:2]

    # ── Our palm decode ──
    intr = _load_palm_interpreter(tflite_path)
    if intr is None:
        _nok("palm_interpreter", "tensorflow / tflite_runtime 없음")
        return
    intr.allocate_tensors()
    inp = intr.get_input_details()[0]
    outs = intr.get_output_details()

    tensor, meta = rgb_uint8_to_palm_input_tensor(rgb)
    if inp["dtype"] in (np.float32, "float32"):
        tensor = tensor.astype(np.float32)
    intr.set_tensor(inp["index"], tensor)
    intr.invoke()
    raw = [intr.get_tensor(o["index"]) for o in outs]

    anchors = generate_ssd_anchors()
    our_dets = decode_palm_tensors(raw, anchors, letterbox_meta=meta, score_thresh=0.5)
    n_ours = our_dets.shape[0]
    print(f"  our palm detections: {n_ours}")
    for i, d in enumerate(our_dets):
        print(f"    det[{i}] score={d[0]:.3f}  "
              f"box=[{d[DET_YMIN_IDX]:.3f},{d[DET_XMIN_IDX]:.3f},"
              f"{d[DET_YMAX_IDX]:.3f},{d[DET_XMAX_IDX]:.3f}]")

    # ── MediaPipe Hands ──
    try:
        import mediapipe as mp
    except ImportError:
        _nok("mediapipe_import", "mediapipe not installed")
        return
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5,
    )
    mp_res = hands.process(rgb)
    hands.close()

    mp_wrists = []
    if mp_res.multi_hand_landmarks:
        for hlm in mp_res.multi_hand_landmarks:
            w = hlm.landmark[0]
            mp_wrists.append((w.x, w.y))  # normalized [0,1]
    n_mp = len(mp_wrists)
    print(f"  mediapipe hands: {n_mp}")
    for i, (wx, wy) in enumerate(mp_wrists):
        print(f"    hand[{i}] wrist=({wx:.3f}, {wy:.3f})")

    # ── Compare ──

    # Count comparison (allow ±1 difference)
    _check("detection_count_close", abs(n_ours - n_mp) <= 1,
           f"ours={n_ours}, mediapipe={n_mp}")

    if n_ours == 0 and n_mp == 0:
        _ok("both_empty")
        return

    if n_ours == 0:
        _nok("ours_empty", "our decode found 0 palms but MediaPipe found hands")
        return

    # For each MediaPipe wrist, check if it falls inside (or near) one of our palm boxes
    MARGIN = 0.15  # 15% margin around the box
    matched = 0
    for wx, wy in mp_wrists:
        for d in our_dets:
            bh = d[DET_YMAX_IDX] - d[DET_YMIN_IDX]
            bw = d[DET_XMAX_IDX] - d[DET_XMIN_IDX]
            ymin = d[DET_YMIN_IDX] - bh * MARGIN
            ymax = d[DET_YMAX_IDX] + bh * MARGIN
            xmin = d[DET_XMIN_IDX] - bw * MARGIN
            xmax = d[DET_XMAX_IDX] + bw * MARGIN
            if xmin <= wx <= xmax and ymin <= wy <= ymax:
                matched += 1
                break

    _check("wrist_in_palm_box",
           matched == n_mp or (n_mp > 0 and matched >= 1),
           f"matched {matched}/{n_mp} MediaPipe wrists inside our palm boxes (+{MARGIN*100:.0f}% margin)")

    # Keypoint proximity: palm kp0 (wrist) should be near MediaPipe wrist
    if n_mp > 0 and n_ours > 0:
        best_dist = float("inf")
        for wx, wy in mp_wrists:
            for d in our_dets:
                kp0_x = d[DET_KP_OFFSET + 0]
                kp0_y = d[DET_KP_OFFSET + 1]
                dist = ((kp0_x - wx) ** 2 + (kp0_y - wy) ** 2) ** 0.5
                best_dist = min(best_dist, dist)
        _check("kp0_near_wrist", best_dist < 0.25,
               f"min distance between palm kp0 and MP wrist = {best_dist:.4f}")
        print(f"  closest palm-kp0 ↔ MP-wrist distance: {best_dist:.4f} (normalized)")

    # Score sanity: our top detection should have reasonable confidence
    top_score = our_dets[0, DET_SCORE_IDX]
    _check("top_score_reasonable", top_score > 0.5,
           f"top score = {top_score:.3f}")

    # Box sanity: coordinates in [0, 1]
    coords_ok = True
    for d in our_dets:
        for idx in [DET_YMIN_IDX, DET_XMIN_IDX, DET_YMAX_IDX, DET_XMAX_IDX]:
            if d[idx] < -0.1 or d[idx] > 1.1:
                coords_ok = False
    _check("box_coords_valid", coords_ok, "some box coords outside [-0.1, 1.1]")


# ── Main ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Palm decode unit tests")
    ap.add_argument("--camera", type=int, default=-1, help="Camera index (default: skip camera test)")
    ap.add_argument("--image", type=str, default=None, help="Image file for comparison")
    ap.add_argument("--palm-tflite", type=str, default=None, help="Palm TFLite model path")
    args = ap.parse_args()

    print("=" * 60)
    print("Palm decode unit tests")
    print("=" * 60)

    # Self-tests (no dependencies)
    test_anchor_generation()
    test_nms_merge()
    test_nms_non_overlapping()
    test_letterbox_removal()
    test_empty_input()
    test_decode_random_tensor()

    # Camera / image comparison test
    tflite_path = args.palm_tflite
    if tflite_path is None:
        default = Path(__file__).resolve().parent.parent / "models" / "vendor" / "palm_detection_lite.tflite"
        if default.is_file():
            tflite_path = str(default)

    rgb = None
    if args.image:
        import cv2
        bgr = cv2.imread(args.image)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            print(f"\nUsing image: {args.image} ({rgb.shape})")
        else:
            print(f"\nFailed to read image: {args.image}")
    elif args.camera >= 0:
        import cv2
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, bgr = cap.read()
        cap.release()
        if ok:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            print(f"\nUsing camera {args.camera} ({rgb.shape})")
        else:
            print(f"\nCamera {args.camera} read failed")

    if rgb is not None and tflite_path:
        test_vs_mediapipe(rgb, tflite_path)
    elif rgb is not None:
        print("\n[skip] no palm TFLite model found — skipping vs-mediapipe test")
    else:
        print("\n[skip] no camera/image — skipping vs-mediapipe test")

    # Summary
    print("\n" + "=" * 60)
    total = _pass + _fail
    print(f"Results: {_pass}/{total} passed, {_fail} failed")
    print("=" * 60)
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
