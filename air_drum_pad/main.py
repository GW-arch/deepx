#!/usr/bin/env python3
"""
AI Air-Drum Pad — Camera → MediaPipe Hands → Hit zones + velocity strike → pygame audio.

DeepX M1: replace MediaPipe with DX-RT + compiled Hand ONNX (.dxnn); keep StrikeDetector.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import mediapipe as mp
import pygame

from drumkit_audio import build_kit
from strike_detector import StrikeDetector, default_zones


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Air-Drum Pad prototype")
    p.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--vy-trigger", type=float, default=0.012, help="Norm coords / sec")
    p.add_argument("--cooldown", type=float, default=0.12, help="Seconds per pad")
    return p.parse_args()


def draw_zones(frame, zones) -> None:
    h, w = frame.shape[:2]
    for z in zones:
        x0, y0 = int(z.x0 * w), int(z.y0 * h)
        x1, y1 = int(z.x1 * w), int(z.y1 * h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 0), 2)
        cv2.putText(
            frame,
            z.name,
            (x0 + 4, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            2,
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
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    fps_t0 = time.perf_counter()
    frames = 0
    print("Air-Drum Pad running. q=quit. Strike downward into green pads.")

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

            tip_x = tip_y = 0.5
            conf = 0.0
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark[8]
                tip_x, tip_y = lm.x, lm.y
                conf = 1.0
                if res.multi_handedness:
                    conf = float(res.multi_handedness[0].classification[0].score)

            hit = det.update(t, tip_x, tip_y, conf)
            if hit:
                _, key = hit
                kit[key].play()

            draw_zones(frame, zones)
            hx, hy = int(tip_x * frame.shape[1]), int(tip_y * frame.shape[0])
            cv2.circle(frame, (hx, hy), 10, (0, 128, 255), -1)

            if frames % 30 == 0:
                elapsed = time.perf_counter() - fps_t0
                fps = frames / max(elapsed, 1e-6)
                cv2.putText(
                    frame,
                    f"FPS ~{fps:.1f}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("AI Air-Drum Pad", frame)
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
