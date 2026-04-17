#!/usr/bin/env python3
"""
AI Air-Drum Pad — 손가락 관절·손끝 추적으로 실제 악기를 치는 것처럼 타격 감지.

DeepX M1: MediaPipe TFLite → ONNX(`tools/export_mediapipe_hand_onnx.py`) → DX-COM `.dxnn` → `--backend npu`(동일 strike_detector 입력).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import pygame

from drumkit_audio import (
    PIANO_DEFAULT_SLOTS,
    build_kit,
    build_piano_kit,
    build_piano_kit_for_slots,
    kit_keys,
    load_piano_slots_json,
    piano_kit_keys,
    piano_slots_from_inter_hand_distance,
    wide_piano_prerender_names,
)
from hand_tracker import create_tracker
from strike_detector import (
    FINGER_ANGLE_CHAIN,
    FINGER_LABELS,
    FINGERTIP_INDICES,
    InstrumentStrikeDetector,
    load_instrument_slots_json,
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
        default=0.01,
        help="손끝 하강 속도(정규화 좌표/s) 하한",
    )
    p.add_argument(
        "--joint-dps",
        type=float,
        default=120.0,
        help="관절 각속도(|deg/s|) 하한 — 손가락 관절이 실제로 움직일 때",
    )
    p.add_argument("--cooldown", type=float, default=0.12, help="같은 손가락 연타 쿨다운(초)")
    p.add_argument("--max-hands", type=int, default=2, choices=[1, 2])
    p.add_argument("--model-complexity", type=int, default=0, choices=[0, 1])
    p.add_argument("--trail", type=int, default=24, help="손끝 궤적 길이(프레임)")
    p.add_argument(
        "--instruments",
        type=str,
        default="",
        metavar="PATH",
        help="JSON: 손0·손1 각 5손가락 순으로 sound key 10개 (예: instruments.example.json)",
    )
    p.add_argument(
        "--list-instruments",
        action="store_true",
        help="사용 가능한 sound key 목록 출력 후 종료",
    )
    p.add_argument(
        "--piano",
        action="store_true",
        help="피아노 모드: 음명(C4 등) 합성음. --instruments 없으면 양손 손목 거리로 음역 자동",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=("cpu", "npu"),
        help="손 추론: cpu=MediaPipe, npu=DX-RT .dxnn (--dxnn 필수)",
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
    slot_state: dict[str, tuple[str, ...]] = {"slots": tuple(PIANO_DEFAULT_SLOTS)}
    side_by_mp: dict[int, int] = {}
    use_piano_dynamic = bool(args.piano and not args.instruments.strip())

    if args.piano:
        if args.instruments.strip():
            slots = load_piano_slots_json(args.instruments.strip())
            kit = build_piano_kit_for_slots(slots)
            sound_mapper = lambda h, lm, s=slots, mh=args.max_hands: sound_key_for_finger(
                h, lm, max_hands=mh, sound_slots=s
            )
        else:
            kit = build_piano_kit(note_names=wide_piano_prerender_names())

            def piano_dynamic_mapper(h: int, lm: int) -> str:
                return sound_key_for_finger(
                    side_by_mp[h],
                    lm,
                    max_hands=args.max_hands,
                    sound_slots=slot_state["slots"],
                )

            sound_mapper = piano_dynamic_mapper
    else:
        kit = build_kit()
        sound_mapper = None
        if args.instruments.strip():
            slots = load_instrument_slots_json(
                args.instruments.strip(),
                valid_keys=frozenset(kit.keys()),
            )
            sound_mapper = lambda h, lm, s=slots, mh=args.max_hands: sound_key_for_finger(
                h, lm, max_hands=mh, sound_slots=s
            )

    det = InstrumentStrikeDetector(
        vy_trigger=args.vy_trigger,
        joint_dps_trigger=args.joint_dps,
        cooldown_s=args.cooldown,
        max_hands=args.max_hands,
        sound_mapper=sound_mapper,
    )

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
    )

    fps_t0 = time.perf_counter()
    frames = 0
    mode = "piano" if args.piano else "drum"
    be = f"{args.backend.upper()}"
    if args.backend == "npu" and args.dxnn.strip():
        be = f"NPU:{Path(args.dxnn).name}"
    print(
        f"Air-Drum [{mode}] backend={be}: q=quit | tip↓ + joint motion → hit | (손,손가락)→음",
        flush=True,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed", file=sys.stderr)
                return 1

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

            piano_wrist_d = 0.11
            if use_piano_dynamic:
                if len(landmarks_list) >= 2:
                    w0 = landmarks_list[0].landmark[0]
                    w1 = landmarks_list[1].landmark[0]
                    piano_wrist_d = math.hypot(w0.x - w1.x, w0.y - w1.y)
                slot_state["slots"] = piano_slots_from_inter_hand_distance(piano_wrist_d)

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

                if use_piano_dynamic and hand_idx == 0:
                    s = slot_state["slots"]
                    hint = f"d={piano_wrist_d:.2f} L:{s[0]}-{s[4]} R:{s[5]}-{s[9]}"
                    cv2.putText(
                        frame,
                        hint,
                        (8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (200, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )

                for fid in FINGERTIP_INDICES:
                    col = FINGER_COLORS.get(fid, (200, 200, 200))
                    draw_finger_chain(frame, hand_lms, fid, col)

                    hit = det.update_finger(hand_idx, fid, t, hand_lms, conf)
                    if hit:
                        _, sk = hit
                        if sk in kit:
                            kit[sk].play()

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

            title = "AI Air-Drum (piano)" if args.piano else "AI Air-Drum (drum)"
            cv2.imshow(title, frame)
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
