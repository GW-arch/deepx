#!/usr/bin/env python3
"""
AI Air-Drum Pad — 손가락 관절·손끝 추적으로 실제 악기를 치는 것처럼 타격 감지.

DeepX M1: MediaPipe TFLite → ONNX(`tools/export_mediapipe_hand_onnx.py`) → DX-COM `.dxnn` → `--backend npu`(동일 strike_detector 입력).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict, deque

# Resolve relative paths against the script's own directory so that
# default model paths like "models/vendor/…" work from any cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)

import cv2
import numpy as np
import pygame

from drumkit_audio import (
    build_kit,
    build_piano_kit_for_slots,
    kit_keys,
    load_piano_slots_json,
    piano_kit_keys,
)
from hand_tracker import create_tracker
from strike_detector import (
    FINGER_ANGLE_CHAIN,
    FINGER_LABELS,
    FINGERTIP_INDICES,
    InstrumentStrikeDetector,
    PadStrikeDetector,
    default_pad_zones,
    load_pad_zones_json,
    sound_key_for_finger,
)

# BGR — 손가락별
FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    4: (180, 180, 255),
    8: (255, 255, 0),
    12: (0, 255, 0),
    16: (255, 0, 255),
    20: (0, 165, 255),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Air-Drum: joint + tip velocity (like striking drums)",
    )
    p.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--vy-trigger",
        type=float,
        default=0.025,
        help="손끝 하강 속도(정규화 좌표/s) 하한 (middle finger는 내부적으로 더 민감하게 보정)",
    )
    p.add_argument(
        "--joint-dps",
        type=float,
        default=16.0,
        help="관절 각속도(|deg/s|) 하한 — 손가락 관절이 실제로 움직일 때",
    )
    p.add_argument("--cooldown", type=float, default=0.10, help="같은 손가락/패드 연타 쿨다운(초)")
    p.add_argument("--max-hands", type=int, default=2, choices=[1, 2])
    p.add_argument("--model-complexity", type=int, default=0, choices=[0, 1])
    p.add_argument("--trail", type=int, default=24, help="손끝 궤적 길이(프레임)")
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="좌우 반전(selfie/거울) 화면을 끕니다. 기본은 움직임이 직관적이도록 mirror view입니다.",
    )
    p.add_argument(
        "--instruments",
        type=str,
        default="",
        metavar="PATH",
        help="피아노/손가락 매핑 JSON: 손0·손1 각 5손가락 순 sound key 10개",
    )
    p.add_argument(
        "--drum-pads",
        type=str,
        default="",
        metavar="PATH",
        help="드럼 패드 레이아웃 JSON (기본: 내장 8-패드 그리드)",
    )
    p.add_argument(
        "--list-instruments",
        action="store_true",
        help="사용 가능한 sound key 목록 출력 후 종료",
    )
    p.add_argument(
        "--piano",
        action="store_true",
        help="피아노 모드: 음명(C4 등) 합성음. 기본 instruments.piano.example.json 사용, --instruments로 변경 가능",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=("cpu", "cpu-baseline", "npu", "npu-full"),
        help="손 추론: cpu=MediaPipe, cpu-baseline=palm+hand TFLite(CPU), npu=DX-RT .dxnn, npu-full=palm TFLite + hand .dxnn",
    )
    p.add_argument(
        "--dxnn",
        type=str,
        default="",
        metavar="PATH",
        help="NPU 백엔드일 때 컴파일된 .dxnn 모델 경로",
    )
    p.add_argument(
        "--dxnn-layout",
        type=str,
        default="",
        metavar="PATH",
        help="입출력 레이아웃 JSON (예: models/dxnn_layout.example.json)",
    )
    p.add_argument(
        "--palm-tflite",
        type=str,
        default="",
        metavar="PATH",
        help="npu-full 백엔드: palm detection TFLite 경로 (CPU 폴백)",
    )
    p.add_argument(
        "--palm-dxnn",
        type=str,
        default="",
        metavar="PATH",
        help="npu-full 백엔드: palm detection .dxnn 경로 (NPU, 기본: models/vendor/palm_detection_lite.dxnn 자동탐색)",
    )
    p.add_argument(
        "--hand-tflite",
        type=str,
        default="",
        metavar="PATH",
        help="cpu-baseline 백엔드: hand landmark TFLite 경로 (기본: models/vendor/hand_landmark_lite.tflite 자동탐색)",
    )
    p.add_argument(
        "--palm-redetect-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "cpu-baseline/npu-full 실험용: palm detection 후 N프레임 동안 landmark 기반 ROI 추적만 수행. "
            "0이면 매 프레임 palm 실행(기본, 드리프트 최소)."
        ),
    )
    p.add_argument(
        "--async-palm",
        action="store_true",
        help=(
            "cpu-baseline/npu-full 실험용: palm detection을 백그라운드 스레드에서 돌리고 "
            "그 사이 이전 ROI로 hand landmark를 계속 추적합니다."
        ),
    )
    p.add_argument(
        "--landmark-correction",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "npu-full 실험용: CPU baseline 기준으로 학습한 NPU landmark affine 보정 JSON. "
            "tools/calibrate_npu_landmarks.py 로 생성."
        ),
    )
    return p.parse_args()


def draw_finger_chain(
    frame: np.ndarray,
    hand_lms: object,
    tip_id: int,
    color: tuple[int, int, int],
) -> None:
    """관절 체인 시각화 (디버그·연주 느낌)."""
    if tip_id not in FINGER_ANGLE_CHAIN:
        return
    a, b, c = FINGER_ANGLE_CHAIN[tip_id]
    h, w = frame.shape[:2]
    pts = []
    for i in (a, b, c):
        lm = hand_lms.landmark[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)


def draw_pads(
    frame: np.ndarray,
    pads: list,
    active_until: dict[str, float],
    t: float,
) -> None:
    """Draw normalized drum pads with a short bright flash for active hits."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    draw_items: list[tuple[int, int, int, int, tuple[int, int, int], bool, str]] = []
    for pad in pads:
        x1, y1 = int(pad.x1 * w), int(pad.y1 * h)
        x2, y2 = int(pad.x2 * w), int(pad.y2 * h)
        is_active = t < active_until.get(pad.label, 0.0)
        fill_color = (
            tuple(min(255, int(c) + 100) for c in pad.color)
            if is_active
            else tuple(int(c) for c in pad.color)
        )
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        draw_items.append((x1, y1, x2, y2, fill_color, is_active, pad.label))

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    for x1, y1, x2, y2, color, is_active, label in draw_items:
        thickness = 3 if is_active else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (x1 + 8, y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    if args.list_instruments:
        if args.piano:
            print(" ".join(piano_kit_keys()))
        else:
            print(" ".join(kit_keys()))
        return 0

    pygame.init()
    slots: tuple[str, ...] | None = None
    side_by_mp: dict[int, int] = {}
    piano_json = args.instruments.strip() if args.instruments.strip() else "instruments.piano.example.json"

    if args.piano:
        slots = load_piano_slots_json(piano_json)
        kit = build_piano_kit_for_slots(slots)
        sound_mapper = lambda h, lm, s=slots, mh=args.max_hands: sound_key_for_finger(
            h, lm, max_hands=mh, sound_slots=s
        )
        det: InstrumentStrikeDetector | None = InstrumentStrikeDetector(
            vy_trigger=args.vy_trigger,
            joint_dps_trigger=args.joint_dps,
            cooldown_s=args.cooldown,
            max_hands=args.max_hands,
            sound_mapper=sound_mapper,
        )
        pad_zones = []
        pad_det: PadStrikeDetector | None = None
    else:
        kit = build_kit()
        if args.instruments.strip():
            print(
                "--instruments is ignored in drum pad mode; use --drum-pads for pad layout.",
                file=sys.stderr,
            )
        if args.drum_pads.strip():
            pad_zones = load_pad_zones_json(args.drum_pads.strip(), frozenset(kit.keys()))
        else:
            pad_zones = default_pad_zones()
        pad_det = PadStrikeDetector(
            pad_zones,
            vy_trigger=args.vy_trigger,
            joint_dps_trigger=args.joint_dps,
            cooldown_s=args.cooldown,
        )
        det = None

    trails: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=max(4, args.trail)),
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    tracker = create_tracker(
        args.backend,
        max_hands=args.max_hands,
        model_complexity=args.model_complexity,
        dxnn_path=args.dxnn,
        dxnn_layout=args.dxnn_layout if args.dxnn_layout.strip() else None,
        palm_tflite=args.palm_tflite if args.palm_tflite.strip() else None,
        palm_dxnn=args.palm_dxnn if args.palm_dxnn.strip() else None,
        hand_tflite=args.hand_tflite if args.hand_tflite.strip() else None,
        palm_redetect_every=args.palm_redetect_every,
        async_palm=args.async_palm,
        landmark_correction=(
            args.landmark_correction if args.landmark_correction.strip() else None
        ),
    )

    fps_t0 = time.perf_counter()
    frames = 0
    mode = "piano" if args.piano else "drum"
    be = f"{args.backend.upper()}"
    if args.backend == "npu" and args.dxnn.strip():
        be = f"NPU:{Path(args.dxnn).name}"
    mapping_hint = "(손,손가락)→음" if args.piano else "on-screen rectangle pad → drum sound"
    print(
        f"Air-Drum [{mode}] backend={be}: q=quit | tip↓ + joint motion → hit | {mapping_hint}",
        flush=True,
    )

    # --- Fullscreen window ---
    title = "AI Air-Drum (piano)" if args.piano else "AI Air-Drum (drum)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- Detect screen size for layout ---
    # Canvas = full screen.  Layout: [video | sidebar]
    # sidebar is ~25% width for mapping + feedback.
    screen_w, screen_h = 1920, 1080  # fallback
    try:
        import subprocess as _sp
        _xr = _sp.check_output(
            ["xrandr"], env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
            stderr=_sp.DEVNULL, timeout=2,
        ).decode()
        for _line in _xr.splitlines():
            if "*" in _line:
                _res = _line.split()[0]
                screen_w, screen_h = (int(x) for x in _res.split("x"))
                break
    except Exception:
        pass
    SIDEBAR_W = max(320, screen_w // 4)
    VIDEO_W = screen_w - SIDEBAR_W
    VIDEO_H = screen_h
    SIDEBAR_BG = (30, 30, 30)

    # --- Key-mapping image for sidebar ---
    mapping_img_path = (
        _SCRIPT_DIR / "instruments" / ("piano_default.png" if args.piano else "drum_default.png")
    )
    mapping_sidebar: np.ndarray | None = None
    if mapping_img_path.is_file():
        _raw = cv2.imread(str(mapping_img_path), cv2.IMREAD_COLOR)
        if _raw is not None:
            # Scale to fit sidebar width with padding
            _fit_w = SIDEBAR_W - 20
            _scale = _fit_w / _raw.shape[1]
            _fit_h = int(_raw.shape[0] * _scale)
            mapping_sidebar = cv2.resize(_raw, (_fit_w, _fit_h), interpolation=cv2.INTER_AREA)

    # --- Strike feedback state: list of (expire_time, text, color) ---
    strike_events: list[tuple[float, str, tuple[int, int, int]]] = []
    active_pads: dict[str, float] = {}
    STRIKE_DISPLAY_SEC = 3.0
    PAD_FLASH_SEC = 0.20

    try:
        fail_streak = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_streak += 1
                if fail_streak < 30:
                    time.sleep(0.02)
                    continue
                print("Camera read failed", file=sys.stderr)
                return 1
            fail_streak = 0
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            frames += 1
            t = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = tracker.process(rgb)

            landmarks_list = res.multi_hand_landmarks or []
            handedness_list = res.multi_handedness or []

            side_by_mp.clear()
            for i, hl in enumerate(landmarks_list):
                if i < len(handedness_list):
                    lab = handedness_list[i].classification[0].label
                    side_by_mp[i] = 0 if lab == "Left" else 1
                else:
                    side_by_mp[i] = 0 if hl.landmark[0].x < 0.5 else 1

            for hand_idx, hand_lms in enumerate(landmarks_list):
                conf = 1.0
                if hand_idx < len(handedness_list):
                    conf = float(handedness_list[hand_idx].classification[0].score)

                wrist = hand_lms.landmark[0]
                wx = int(wrist.x * frame.shape[1])
                wy = int(wrist.y * frame.shape[0])
                label = "?"
                if hand_idx < len(handedness_list):
                    lr = handedness_list[hand_idx].classification[0].label
                    label = lr[0].upper()
                cv2.putText(
                    frame,
                    f"H{hand_idx}:{label}",
                    (wx - 10, wy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                for fid in FINGERTIP_INDICES:
                    col = FINGER_COLORS.get(fid, (200, 200, 200))
                    draw_finger_chain(frame, hand_lms, fid, col)

                    if args.piano:
                        assert det is not None
                        hit = det.update_finger(hand_idx, fid, t, hand_lms, conf)
                        if hit:
                            _, sk = hit
                            if sk in kit:
                                kit[sk].play()
                            # Record for on-screen feedback
                            fn = FINGER_LABELS.get(fid, "?")
                            side = "L" if side_by_mp.get(hand_idx, 0) == 0 else "R"
                            strike_events.append(
                                (
                                    t + STRIKE_DISPLAY_SEC,
                                    f"{side}:{fn} -> {sk}",
                                    FINGER_COLORS.get(fid, (200, 200, 200)),
                                )
                            )
                    else:
                        assert pad_det is not None
                        hit_pad = pad_det.update_finger(hand_idx, fid, t, hand_lms, conf)
                        if hit_pad:
                            if hit_pad.sound_key in kit:
                                kit[hit_pad.sound_key].play()
                            strike_events.append(
                                (
                                    t + STRIKE_DISPLAY_SEC,
                                    hit_pad.label,
                                    hit_pad.color,
                                )
                            )
                            active_pads[hit_pad.label] = t + PAD_FLASH_SEC

                    lm = hand_lms.landmark[fid]
                    px = int(lm.x * frame.shape[1])
                    py = int(lm.y * frame.shape[0])
                    trails[(hand_idx, fid)].append((px, py))
                    tdeque = trails[(hand_idx, fid)]
                    if len(tdeque) >= 2:
                        arr = np.array(list(tdeque), dtype=np.int32)
                        cv2.polylines(frame, [arr], False, col, 2, cv2.LINE_AA)

                    cv2.circle(frame, (px, py), 6, col, -1)
                    cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
                    fn = FINGER_LABELS.get(fid, "?")
                    cv2.putText(
                        frame,
                        fn[0].upper(),
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        col,
                        1,
                        cv2.LINE_AA,
                    )

            if frames % 30 == 0:
                elapsed = time.perf_counter() - fps_t0
                fps = frames / max(elapsed, 1e-6)
                cv2.putText(
                    frame,
                    f"FPS ~{fps:.1f} [{args.backend}]  vy>={args.vy_trigger} |joint|>={args.joint_dps:.0f}deg/s",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if not args.piano:
                draw_pads(frame, pad_zones, active_pads, t)

            # --- Compose full-screen canvas: [scaled video | sidebar] ---
            vid_scaled = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)

            sidebar = np.full((screen_h, SIDEBAR_W, 3), SIDEBAR_BG, dtype=np.uint8)

            # Sidebar title
            _stitle = "Piano" if args.piano else "Drum Pads"
            cv2.putText(sidebar, _stitle, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(sidebar, (10, 48), (SIDEBAR_W - 10, 48), (80, 80, 80), 1)

            # Key-mapping image
            _sidebar_y = 60
            if mapping_sidebar is not None:
                mh, mw = mapping_sidebar.shape[:2]
                x_pad = (SIDEBAR_W - mw) // 2
                sidebar[_sidebar_y : _sidebar_y + mh, x_pad : x_pad + mw] = mapping_sidebar
                _sidebar_y += mh + 16

            # Divider before strikes
            cv2.line(sidebar, (10, _sidebar_y), (SIDEBAR_W - 10, _sidebar_y), (80, 80, 80), 1)
            _sidebar_y += 8
            cv2.putText(sidebar, "Strikes", (10, _sidebar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            _sidebar_y += 36

            # Strike feedback list
            strike_events = [(exp, txt, col) for exp, txt, col in strike_events if exp > t]
            for i, (_, txt, col) in enumerate(strike_events[-30:]):
                y_pos = _sidebar_y + i * 34
                if y_pos + 30 > screen_h:
                    break
                cv2.putText(sidebar, txt, (14, y_pos + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

            canvas = np.hstack((vid_scaled, sidebar))
            cv2.imshow(title, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
