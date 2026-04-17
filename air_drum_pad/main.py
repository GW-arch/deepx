#!/usr/bin/env python3
"""
AI Air-Drum Pad — multi-hand / multi-fingertip → Hit zones → pygame audio.

DeepX M1: swap MediaPipe for DX-RT + Hand ONNX; keep StrikeDetector + zones.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import mediapipe as mp
import pygame

from drumkit_audio import build_kit
from strike_detector import (
    FINGER_LABELS,
    FINGERTIP_INDICES,
    StrikeDetector,
    default_zones,
)

# BGR for OpenCV — one color per fingertip role
FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    4: (180, 180, 255),  # thumb — light pink
    8: (255, 255, 0),  # index — cyan
    12: (0, 255, 0),  # middle — green
    16: (255, 0, 255),  # ring — magenta
    20: (0, 165, 255),  # pinky — orange
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Air-Drum Pad (multi-hand)")
    p.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--vy-trigger", type=float, default=0.012, help="Norm coords / sec")
    p.add_argument("--cooldown", type=float, default=0.1, help="Seconds per pad+finger")
    p.add_argument("--max-hands", type=int, default=2, choices=[1, 2], help="MediaPipe max hands")
    p.add_argument(
        "--model-complexity",
        type=int,
        default=0,
        choices=[0, 1],
        help="0=faster, 1=more accurate (multi-hand)",
    )
    return p.parse_args()


def draw_zones(frame, zones) -> None:
    h, w = frame.shape[:2]
    for i, z in enumerate(zones):
        x0, y0 = int(z.x0 * w), int(z.y0 * h)
        x1, y1 = int(z.x1 * w), int(z.y1 * h)
        hue = (i * 18) % 80
        color = (40, 180 + hue, 40)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        label = z.sound_key
        cv2.putText(
            frame,
            label,
            (x0 + 3, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 255, 220),
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    pygame.init()
    kit = build_kit()
    zones = default_zones()
    det = StrikeDetector(
        zones,
        vy_trigger=args.vy_trigger,
        cooldown_s=args.cooldown,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.5,
    )

    fps_t0 = time.perf_counter()
    frames = 0
    print(
        "Air-Drum: q=quit | Up to 2 hands × 5 fingertips | Downward strike into pads",
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
            res = hands.process(rgb)

            draw_zones(frame, zones)

            landmarks_list = res.multi_hand_landmarks or []
            handedness_list = res.multi_handedness or []

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
                    lm = hand_lms.landmark[fid]
                    hit = det.update_finger(hand_idx, fid, t, lm.x, lm.y, conf)
                    if hit:
                        _, sk = hit
                        if sk in kit:
                            kit[sk].play()

                    px = int(lm.x * frame.shape[1])
                    py = int(lm.y * frame.shape[0])
                    col = FINGER_COLORS.get(fid, (200, 200, 200))
                    cv2.circle(frame, (px, py), 7, col, -1)
                    cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)
                    fn = FINGER_LABELS.get(fid, "?")
                    cv2.putText(
                        frame,
                        fn[0].upper(),
                        (px + 6, py - 6),
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
                    f"FPS ~{fps:.1f}  hands={args.max_hands}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("AI Air-Drum Pad (multi)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
