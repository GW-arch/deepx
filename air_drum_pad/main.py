#!/usr/bin/env python3
"""
AI Air-Drum Pad — 손가락 관절·손끝 추적으로 실제 악기를 치는 것처럼 타격 감지.

DeepX M1: default live path uses the final CPU-palm + NPU hand-landmark (`npu-full`) pipeline with the same guided-style interface used by the evaluator.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

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


def physical_side_from_screen_x(nx: float, *, mirror: bool) -> int:
    """Return 0 for the performer's left hand and 1 for right hand.

    In the default mirrored/selfie view, screen-left should be treated as the
    performer's left hand for the UI and piano-note mapping.  Use this
    mirror-aware mapping instead of MediaPipe handedness, which changes under
    image flips.
    """
    if mirror:
        return 0 if nx < 0.5 else 1
    return 0 if nx >= 0.5 else 1


def draw_guided_style_landmarks(
    frame: np.ndarray,
    hand_lms: object,
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    """Draw the same minimal hand skeleton used by guided_eval.

    The live, non-guided runtime deliberately avoids per-finger colored trails
    and thumb/pinky letter overlays, because those made the live interface look
    different from the guided evaluator and visually amplified occasional
    right-hand finger-order mistakes.  The strike detector still evaluates each
    fingertip internally; this drawing is only an interface layer.
    """
    h, w = frame.shape[:2]
    for tip_id in (4, 8, 12, 16, 20):
        lm = hand_lms.landmark[tip_id]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 7, color, -1)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)

    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)


def landmarks_for_display(hand_lms: object, *, mirror: bool) -> object:
    """Return landmarks in the coordinate system of the displayed frame.

    Accuracy checks on the captured dataset show that the raw camera frame keeps
    the model's right-hand thumb/pinky identity correct.  Therefore inference
    runs on the raw frame and the selfie mirror is applied only to the displayed
    image and landmark coordinates.
    """
    if not mirror:
        return hand_lms
    out = copy.deepcopy(hand_lms)
    for lm in out.landmark:
        lm.x = 1.0 - float(lm.x)
    return out


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
    p.add_argument("--trail", type=int, default=24, help="Deprecated/no-op: guided-style live UI no longer draws fingertip trails.")
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="좌우 반전(selfie/거울) 화면을 끕니다. 기본은 움직임이 직관적이도록 mirror view입니다.",
    )
    p.add_argument(
        "--screenshot-out",
        type=str,
        default="",
        metavar="PATH",
        help="디버그/보고서용: 지정한 경로에 현재 canvas screenshot을 저장합니다.",
    )
    p.add_argument(
        "--screenshot-delay",
        type=float,
        default=5.0,
        help="--screenshot-out 저장 전 대기 시간(초).",
    )
    p.add_argument(
        "--auto-quit-after",
        type=float,
        default=0.0,
        help="지정한 초 뒤 자동 종료합니다. 0이면 수동 종료(q).",
    )
    p.add_argument(
        "--fullscreen",
        action="store_true",
        help="Use fullscreen display. Default is the same windowed layout as guided_eval.",
    )
    p.add_argument(
        "--windowed",
        action="store_true",
        help="Deprecated/no-op: windowed display is now the default.",
    )
    p.add_argument("--display-width", type=int, default=1280, help="Windowed canvas width")
    p.add_argument("--display-height", type=int, default=720, help="Windowed canvas height")
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
        default="npu-full",
        choices=("cpu", "cpu-baseline", "npu", "npu-full"),
        help="손 추론: cpu=MediaPipe, cpu-baseline=palm+hand TFLite(CPU), npu=DX-RT .dxnn, npu-full=palm TFLite + hand .dxnn (default)",
    )
    p.add_argument(
        "--dxnn",
        type=str,
        default="models/vendor/hand_landmark_lite.dxnn",
        metavar="PATH",
        help="NPU 백엔드일 때 컴파일된 .dxnn 모델 경로",
    )
    p.add_argument(
        "--dxnn-layout",
        type=str,
        default="models/dxnn_layout.mediapipe_hand_lite.json",
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
        default="models/npu_landmark_correction.dataset.json",
        metavar="PATH",
        help=(
            "npu-full 실험용: CPU baseline 기준으로 학습한 NPU landmark affine 보정 JSON. "
            "tools/calibrate_npu_landmarks.py 로 생성."
        ),
    )
    return p.parse_args()


def hand_label_origin(
    frame: np.ndarray,
    hand_lms: object,
    *,
    reserved_top_px: int = 0,
) -> tuple[int, int]:
    """Place a hand label near the landmark bounding box, not a single wrist.

    Wrist-only labels can look shifted for piano poses because the wrist often
    sits high under the HUD while fingertips are lower.  The bounding box anchor
    keeps H0:L/H1:R visually attached to the whole hand and moves labels below
    the box if they would collide with the top overlay.
    """
    h, w = frame.shape[:2]
    xs = [int(lm.x * w) for lm in hand_lms.landmark]
    ys = [int(lm.y * h) for lm in hand_lms.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x = max(8, min(w - 100, x_min))
    y = y_min - 10
    if y < reserved_top_px + 20:
        y = y_max + 22
    y = max(24, min(h - 8, y))
    return x, y


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


def put_text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Readable overlay text matching the guided evaluator visual style."""
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_live_overlay(
    frame: np.ndarray,
    *,
    mode: str,
    backend: str,
    strike_events: list[tuple[float, str, tuple[int, int, int]]],
    t: float,
    mirror: bool,
) -> list[tuple[float, str, tuple[int, int, int]]]:
    """Draw the non-guided runtime HUD in the same style as guided_eval.

    It intentionally omits cue guidance such as READY/HIT NOW and countdowns.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    top_h = 108
    cv2.rectangle(overlay, (0, 0), (w, top_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    title = "PANDA Drum Pads" if mode == "drum" else "PANDA Piano"
    put_text_shadow(frame, title, (24, 48), 1.25, (255, 255, 255), 3)
    subtitle = (
        f"live {mode} | backend {backend} | mirror {'on' if mirror else 'off'} | "
        "q=quit | fingertip down + joint motion = strike"
    )
    put_text_shadow(frame, subtitle, (26, 86), 0.55, (230, 230, 230), 1)

    strike_events = [(exp, txt, col) for exp, txt, col in strike_events if exp > t]
    if strike_events:
        recent = strike_events[-8:]
        panel_h = 44 + len(recent) * 34
        y0 = max(top_h + 14, h - panel_h - 22)
        x0 = 24
        panel_w = min(560, max(320, w // 3))
        panel = frame.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)
        put_text_shadow(frame, "Recent strikes", (x0 + 12, y0 + 30), 0.65, (240, 240, 240), 1)
        for i, (_, txt, col) in enumerate(recent):
            put_text_shadow(frame, txt, (x0 + 16, y0 + 64 + i * 34), 0.72, col, 2)
    return strike_events


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
            h,
            lm,
            max_hands=mh,
            sound_slots=s,
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
            args.landmark_correction
            if args.landmark_correction.strip()
            and Path(args.landmark_correction).is_file()
            else None
        ),
    )

    fps_t0 = time.perf_counter()
    frames = 0
    mode = "piano" if args.piano else "drum"
    be = f"{args.backend.upper()}"
    if args.backend in ("npu", "npu-full") and args.dxnn.strip():
        be = f"{args.backend.upper()}:{Path(args.dxnn).name}"
    mapping_hint = "(손,손가락)→음" if args.piano else "on-screen rectangle pad → drum sound"
    print(
        f"Air-Drum [{mode}] backend={be}: q=quit | tip↓ + joint motion → hit | {mapping_hint}",
        flush=True,
    )

    # --- Guided-style live window: windowed by default, fullscreen only on request. ---
    title = "AI Air-Drum (piano)" if args.piano else "AI Air-Drum (drum)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(title, args.display_width, args.display_height)

    # --- Detect display/canvas size for guided-style live overlay ---
    if not args.fullscreen:
        screen_w, screen_h = args.display_width, args.display_height
    else:
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

    # --- Strike feedback state: list of (expire_time, text, color) ---
    strike_events: list[tuple[float, str, tuple[int, int, int]]] = []
    active_pads: dict[str, float] = {}
    STRIKE_DISPLAY_SEC = 3.0
    PAD_FLASH_SEC = 0.20

    run_t0 = time.perf_counter()
    screenshot_saved = False

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
            frames += 1
            t = time.perf_counter()
            raw_frame = frame
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            res = tracker.process(rgb)
            if not args.no_mirror:
                frame = cv2.flip(raw_frame, 1)
            else:
                frame = raw_frame

            landmarks_list = [
                landmarks_for_display(hl, mirror=not args.no_mirror)
                for hl in (res.multi_hand_landmarks or [])
            ]
            handedness_list = res.multi_handedness or []

            side_by_mp.clear()
            for i, hl in enumerate(landmarks_list):
                # Use physical side from display coordinates instead of raw
                # MediaPipe handedness (which flips under mirror transforms).
                # In the default mirrored/selfie view, this keeps piano notes
                # intuitive: physical left hand -> left-hand notes, physical
                # right hand -> right-hand notes.
                side_by_mp[i] = physical_side_from_screen_x(
                    hl.landmark[0].x,
                    mirror=not args.no_mirror,
                )

            for hand_idx, hand_lms in enumerate(landmarks_list):
                conf = 1.0
                if hand_idx < len(handedness_list):
                    conf = float(handedness_list[hand_idx].classification[0].score)

                hand_side = side_by_mp.get(hand_idx, hand_idx)
                label = "L" if hand_side == 0 else "R"
                draw_guided_style_landmarks(frame, hand_lms, (0, 255, 255))
                cv2.putText(
                    frame,
                    f"H{hand_idx}:{label}",
                    hand_label_origin(frame, hand_lms, reserved_top_px=76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                for fid in FINGERTIP_INDICES:
                    if args.piano:
                        assert det is not None
                        mapped_hand_idx = hand_side
                        hit = det.update_finger(mapped_hand_idx, fid, t, hand_lms, conf)
                        if hit:
                            _, sk = hit
                            if sk in kit:
                                kit[sk].play()
                            # Record for on-screen feedback
                            fn = FINGER_LABELS.get(fid, "?")
                            side = "L" if hand_side == 0 else "R"
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

            # --- Compose full-screen canvas matching guided_eval style ---
            canvas = cv2.resize(frame, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
            strike_events = draw_live_overlay(
                canvas,
                mode=mode,
                backend=be,
                strike_events=strike_events,
                t=t,
                mirror=not args.no_mirror,
            )
            cv2.imshow(title, canvas)
            run_elapsed = time.perf_counter() - run_t0
            if (
                args.screenshot_out.strip()
                and not screenshot_saved
                and run_elapsed >= max(0.0, args.screenshot_delay)
            ):
                out_path = Path(args.screenshot_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), canvas)
                screenshot_saved = True
                print(f"Saved screenshot: {out_path}", flush=True)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.auto_quit_after > 0 and run_elapsed >= args.auto_quit_after:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
