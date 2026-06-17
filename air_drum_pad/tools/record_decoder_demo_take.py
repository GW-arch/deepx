#!/usr/bin/env python3
"""Record a live decoder-based demo take with target cues.

Unlike render_forced_audio_demo.py, this records sounds from actual decoded
strike events. The guide sequence is used only for on-screen target cues.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import pygame

PROJECT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for p in (PROJECT_DIR, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import render_forced_audio_demo as forced_audio  # noqa: E402
from drumkit_audio import build_kit, build_piano_kit_for_slots, load_piano_slots_json  # noqa: E402
from hand_tracker import create_tracker  # noqa: E402
from main import (  # noqa: E402
    draw_guided_style_landmarks,
    draw_live_overlay,
    draw_pads,
    hand_label_origin,
    physical_side_from_screen_x,
)
from strike_detector import (  # noqa: E402
    FINGER_LABELS,
    FINGERTIP_INDICES,
    InstrumentStrikeDetector,
    PadStrikeDetector,
    default_pad_zones,
    load_pad_zones_json,
    sound_key_for_finger,
)


FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    4: (180, 180, 255),
    8: (255, 255, 0),
    12: (0, 255, 0),
    16: (255, 0, 255),
    20: (0, 165, 255),
}


PADDING = 24
PIANO_FINGER_BY_NOTE = {
    "G4": "LEFT THUMB",
    "F4": "LEFT INDEX",
    "E4": "LEFT MIDDLE",
    "D4": "LEFT RING",
    "C4": "LEFT PINKY",
    "C5": "RIGHT THUMB",
    "D5": "RIGHT INDEX",
    "E5": "RIGHT MIDDLE",
    "F5": "RIGHT RING",
    "G5": "RIGHT PINKY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", required=True)
    p.add_argument("--raw-output", required=True)
    p.add_argument("--output", default="", help="Final MP4 output. Omit with --raw-only.")
    p.add_argument(
        "--raw-only",
        action="store_true",
        help="Record clean raw video and decoder event JSON only; skip final audio mux.",
    )
    p.add_argument("--camera", default="/dev/video0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--pre-roll", type=float, default=5.0)
    p.add_argument(
        "--cue-delay",
        type=float,
        default=10.0,
        help="Seconds after recording starts before the first target cue appears",
    )
    p.add_argument("--backend", default="cpu", choices=("cpu", "cpu-baseline", "pinto-cpu", "pinto-npu", "npu", "npu-full"))
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2))
    p.add_argument("--model-complexity", type=int, default=0, choices=(0, 1))
    p.add_argument("--dxnn", default="models/vendor/hand_landmark_lite.dxnn")
    p.add_argument("--dxnn-layout", default="models/dxnn_layout.mediapipe_hand_lite.json")
    p.add_argument("--palm-tflite", default="")
    p.add_argument("--palm-dxnn", default="models/vendor/palm_detection_lite_minmax_local.dxnn")
    p.add_argument("--hand-tflite", default="")
    p.add_argument("--hand-onnx", default="models/vendor/pinto_hand_landmark_sparse_Nx3x224x224.onnx")
    p.add_argument("--palm-redetect-every", type=int, default=0)
    p.add_argument("--landmark-correction", default="")
    p.add_argument("--instruments", default="instruments.piano.example.json")
    p.add_argument("--drum-pads", default="")
    p.add_argument("--vy-trigger", type=float, default=0.025)
    p.add_argument("--joint-dps", type=float, default=16.0)
    p.add_argument("--cooldown", type=float, default=0.10)
    p.add_argument("--relative-tip-drop", type=float, default=0.010)
    p.add_argument("--strike-sleep", type=float, default=0.10)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--record-cues",
        action="store_true",
        help="Burn NEXT/STRIKE/sequence cue UI into the saved raw/final video. Default: preview only.",
    )
    p.add_argument("--window-name", default="PANDA decoder demo recorder")
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_DIR / p).resolve()


def visible_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(data.get("mode", "drum"))
    out: list[dict[str, Any]] = []
    for event in data.get("events", []):
        if not isinstance(event, dict) or event.get("visible") is False:
            continue
        if mode == "drum" and str(event.get("sound", "")) == "hat":
            continue
        out.append(event)
    return out


def shifted_events(events: list[dict[str, Any]], delay_s: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for event in events:
        e = dict(event)
        e["t"] = float(e.get("t", 0.0)) + delay_s
        shifted.append(e)
    return shifted


def target_for_time(events: list[dict[str, Any]], t_s: float) -> tuple[dict[str, Any] | None, float]:
    for event in events:
        dt = float(event.get("t", 0.0)) - t_s
        if dt >= -0.18:
            return event, dt
    return None, 0.0


def short_event_label(mode: str, event: dict[str, Any]) -> str:
    label = str(event.get("label") or event.get("sound") or "?")
    if mode == "drum":
        aliases = {
            "kick": "K",
            "snare": "S",
            "crash": "C",
            "ride": "R",
            "tom_l": "T",
            "tom_m": "T",
            "clap": "CL",
        }
        return aliases.get(str(event.get("sound", "")), label[:2].upper())
    return label[:2].upper()


def put_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 2) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_countdown(frame: np.ndarray, remaining_s: float, mode: str) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)
    put_text(canvas, f"{mode.upper()} decoder take starts in {remaining_s:0.1f}s", (PADDING, 74), 1.02, (255, 255, 255), 3)
    put_text(canvas, "Actual detected strikes will generate the final audio.", (PADDING, 124), 0.68, (235, 235, 235), 2)
    put_text(canvas, "Press q or ESC to abort.", (PADDING, h - 42), 0.62, (220, 220, 220), 2)
    return canvas


def draw_cue_overlay(
    canvas: np.ndarray,
    *,
    mode: str,
    events: list[dict[str, Any]],
    t_s: float,
    duration_s: float,
    hit_count: int,
) -> None:
    h, w = canvas.shape[:2]
    event, dt = target_for_time(events, t_s)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 142), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    if event is None:
        target = "DONE"
        detail = "Hold pose"
        color = (130, 255, 130)
    else:
        sound = str(event.get("sound", ""))
        label = str(event.get("label") or sound)
        if mode == "piano":
            detail = PIANO_FINGER_BY_NOTE.get(sound, "TARGET FINGER")
            target = f"{label} / {sound}"
        else:
            detail = "hit highlighted pad"
            target = label.upper()
        if abs(dt) <= 0.12:
            target = f"STRIKE NOW: {target}"
            color = (60, 255, 255)
        elif dt > 0:
            target = f"NEXT in {dt:0.1f}s: {target}"
            color = (255, 255, 255)
        else:
            target = f"NEXT: {target}"
            color = (255, 255, 255)

    put_text(canvas, target, (PADDING, 58), 1.12, color, 3)
    put_text(canvas, detail, (PADDING, 104), 0.76, (235, 235, 235), 2)
    put_text(canvas, f"actual decoder hits: {hit_count}   {t_s:04.1f}/{duration_s:04.1f}s", (w - 470, 104), 0.64, (220, 220, 220), 2)
    progress = max(0.0, min(1.0, t_s / max(duration_s, 1e-6)))
    cv2.rectangle(canvas, (PADDING, h - 116), (w - PADDING, h - 108), (60, 60, 60), -1)
    cv2.rectangle(canvas, (PADDING, h - 116), (PADDING + int((w - 2 * PADDING) * progress), h - 108), (0, 220, 255), -1)
    draw_sequence_queue(canvas, mode=mode, events=events, t_s=t_s)


def draw_sequence_queue(
    canvas: np.ndarray,
    *,
    mode: str,
    events: list[dict[str, Any]],
    t_s: float,
) -> None:
    h, w = canvas.shape[:2]
    panel_y = h - 100
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    if not events:
        return
    current_idx = len(events)
    for idx, event in enumerate(events):
        if float(event.get("t", 0.0)) >= t_s - 0.18:
            current_idx = idx
            break

    put_text(canvas, f"FULL TARGET SEQUENCE  {min(current_idx + 1, len(events))}/{len(events)}", (PADDING, panel_y + 30), 0.58, (235, 235, 235), 2)
    chip_gap = 5
    chip_count = len(events)
    chip_w = max(24, min(58, int((w - 2 * PADDING - chip_gap * (chip_count - 1)) / max(1, chip_count))))
    x = PADDING
    y = panel_y + 48
    for idx, event in enumerate(events):
        t_event = float(event.get("t", 0.0))
        label = short_event_label(mode, event)
        if idx < current_idx:
            fill = (45, 48, 54)
            edge = (80, 85, 95)
            text_col = (145, 150, 160)
        elif idx == current_idx:
            if abs(t_event - t_s) <= 0.15:
                fill = (30, 190, 215)
            else:
                fill = (70, 115, 230)
            edge = (255, 255, 255)
            text_col = (255, 255, 255)
        else:
            fill = (32, 36, 44)
            edge = (115, 120, 132)
            text_col = (230, 230, 230)
        x1 = x + chip_w
        cv2.rectangle(canvas, (x, y), (x1, y + 36), fill, -1)
        cv2.rectangle(canvas, (x, y), (x1, y + 36), edge, 2 if idx == current_idx else 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(
            canvas,
            label,
            (x + max(2, (chip_w - tw) // 2), y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            text_col,
            1,
            cv2.LINE_AA,
        )
        x = x1 + chip_gap


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cam: str | int = args.camera
    if str(cam).isdigit():
        cam = int(str(cam))
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera: {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def make_tracker(args: argparse.Namespace):
    return create_tracker(
        args.backend,
        max_hands=args.max_hands,
        model_complexity=args.model_complexity,
        dxnn_path=str(resolve_path(args.dxnn)) if args.dxnn else "",
        dxnn_layout=str(resolve_path(args.dxnn_layout)) if args.dxnn_layout else None,
        palm_tflite=str(resolve_path(args.palm_tflite)) if args.palm_tflite else None,
        palm_dxnn=str(resolve_path(args.palm_dxnn)) if args.palm_dxnn else None,
        hand_tflite=str(resolve_path(args.hand_tflite)) if args.hand_tflite else None,
        hand_onnx=str(resolve_path(args.hand_onnx)) if args.hand_onnx else None,
        palm_redetect_every=args.palm_redetect_every,
        async_palm=False,
        landmark_correction=str(resolve_path(args.landmark_correction)) if args.landmark_correction else None,
    )


def synth_actual_events(events: list[tuple[float, str]], *, mode: str, duration_s: float, sample_rate: int, gain: float) -> np.ndarray:
    seq = {
        "mode": mode,
        "duration_s": duration_s,
        "events": [
            {"t": float(t), "sound": sound, "duration": 0.52, "velocity": 1.0}
            for t, sound in events
        ],
    }
    return forced_audio.synthesize(seq, sample_rate=sample_rate, offset_s=0.0, gain=gain, include_click=False)


def write_event_sequence(
    path: Path,
    *,
    mode: str,
    duration_s: float,
    events: list[tuple[float, str]],
) -> None:
    seq = {
        "title": f"Actual decoder events from {path.stem}",
        "mode": mode,
        "duration_s": duration_s,
        "events": [
            {"t": round(float(t), 6), "sound": sound, "duration": 0.52, "velocity": 1.0}
            for t, sound in events
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seq, indent=2) + "\n", encoding="utf-8")


def retime_video_to_duration(path: Path, desired_duration_s: float) -> float:
    encoded_duration_s = forced_audio.ffprobe_duration_s(path)
    if desired_duration_s <= 0.0 or encoded_duration_s <= 0.0:
        return encoded_duration_s
    if abs(encoded_duration_s - desired_duration_s) <= 0.25:
        return encoded_duration_s
    scale = desired_duration_s / encoded_duration_s
    tmp = path.with_suffix(".retimed.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"setpts={scale:.8f}*PTS",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, path)
    return forced_audio.ffprobe_duration_s(path)


def main() -> int:
    args = parse_args()
    data = forced_audio.load_sequence(args.sequence)
    mode = str(data.get("mode", "drum"))
    base_duration_s = forced_audio.sequence_duration_s(data)
    duration_s = base_duration_s + max(0.0, float(args.cue_delay))
    cue_events = shifted_events(visible_events(data), max(0.0, float(args.cue_delay)))
    raw_output = resolve_path(args.raw_output)
    final_output = resolve_path(args.output) if args.output else None
    if not args.raw_only and final_output is None:
        raise SystemExit("--output is required unless --raw-only is set")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if final_output is not None:
        final_output.parent.mkdir(parents=True, exist_ok=True)

    pygame.init()
    if mode == "piano":
        slots = load_piano_slots_json(str(resolve_path(args.instruments)))
        kit = build_piano_kit_for_slots(slots, sample_rate=args.sample_rate)
        det: InstrumentStrikeDetector | None = InstrumentStrikeDetector(
            vy_trigger=args.vy_trigger,
            joint_dps_trigger=args.joint_dps,
            cooldown_s=args.cooldown,
            relative_tip_drop=args.relative_tip_drop,
            max_hands=args.max_hands,
            sound_mapper=lambda h, lm, s=slots, mh=args.max_hands: sound_key_for_finger(h, lm, max_hands=mh, sound_slots=s),
        )
        pad_det: PadStrikeDetector | None = None
        pad_zones = []
    else:
        kit = build_kit(sample_rate=args.sample_rate)
        pad_zones = load_pad_zones_json(str(resolve_path(args.drum_pads)), frozenset(kit.keys())) if args.drum_pads else default_pad_zones()
        pad_det = PadStrikeDetector(
            pad_zones,
            vy_trigger=args.vy_trigger,
            joint_dps_trigger=args.joint_dps,
            cooldown_s=args.cooldown,
            relative_tip_drop=args.relative_tip_drop,
        )
        det = None

    cap = open_camera(args)
    tracker = make_tracker(args)
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        cap.release()
        tracker.close()
        raise SystemExit(f"Could not open video writer: {raw_output}")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.width, args.height)
    if args.fullscreen:
        cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    actual_events: list[tuple[float, str]] = []
    strike_events: list[tuple[float, str, tuple[int, int, int]]] = []
    active_pads: dict[str, float] = {}
    start_at = time.monotonic() + args.pre_roll
    recording = False
    record_started_at = 0.0
    record_duration_s = 0.0
    frames = 0
    aborted = False
    fail_streak = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_streak += 1
                if fail_streak < 30:
                    time.sleep(0.02)
                    continue
                raise RuntimeError("Camera read failed")
            fail_streak = 0
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            if now < start_at:
                display = draw_countdown(frame, start_at - now, mode)
            else:
                if not recording:
                    recording = True
                    record_started_at = time.monotonic()
                    if det is not None:
                        det.reset()
                    if pad_det is not None:
                        pad_det.reset()
                t_s = time.monotonic() - record_started_at
                record_duration_s = t_s
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = tracker.process(rgb)
                landmarks_list = list(res.multi_hand_landmarks or [])
                handedness_list = list(res.multi_handedness or [])
                side_by_mp = {
                    i: physical_side_from_screen_x(hl.landmark[0].x, mirror=args.mirror)
                    for i, hl in enumerate(landmarks_list)
                }
                strike_accepted_this_frame = False

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
                        hand_label_origin(frame, hand_lms, reserved_top_px=142),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    for fid in FINGERTIP_INDICES:
                        if mode == "piano":
                            assert det is not None
                            hit = det.update_finger(hand_side, fid, time.perf_counter(), hand_lms, conf)
                            if hit and not strike_accepted_this_frame:
                                _, sound_key = hit
                                if sound_key in kit:
                                    kit[sound_key].play()
                                actual_events.append((t_s, sound_key))
                                side = "L" if hand_side == 0 else "R"
                                fn = FINGER_LABELS.get(fid, "?")
                                strike_events.append((time.perf_counter() + 2.5, f"{side}:{fn} -> {sound_key}", FINGER_COLORS.get(fid, (220, 220, 220))))
                                strike_accepted_this_frame = True
                        else:
                            assert pad_det is not None
                            hit_pad = pad_det.update_finger(hand_side, fid, time.perf_counter(), hand_lms, conf)
                            if hit_pad and not strike_accepted_this_frame:
                                if hit_pad.sound_key in kit:
                                    kit[hit_pad.sound_key].play()
                                actual_events.append((t_s, hit_pad.sound_key))
                                strike_events.append((time.perf_counter() + 2.5, hit_pad.label, hit_pad.color))
                                active_pads[hit_pad.label] = time.perf_counter() + 0.20
                                strike_accepted_this_frame = True

                if mode == "drum":
                    draw_pads(frame, pad_zones, active_pads, time.perf_counter())

                clean = frame.copy()
                strike_events = draw_live_overlay(
                    clean,
                    mode=mode,
                    backend=f"{args.backend} decoder",
                    strike_events=strike_events,
                    t=time.perf_counter(),
                    mirror=args.mirror,
                    backing_label="",
                )
                display = clean.copy()
                draw_cue_overlay(
                    display,
                    mode=mode,
                    events=cue_events,
                    t_s=t_s,
                    duration_s=duration_s,
                    hit_count=len(actual_events),
                )
                writer.write(display if args.record_cues else clean)
                frames += 1
                if t_s >= duration_s:
                    break
                if strike_accepted_this_frame and args.strike_sleep > 0:
                    time.sleep(args.strike_sleep)
                    if det is not None:
                        det.reset()
                    if pad_det is not None:
                        pad_det.reset()

            cv2.imshow(args.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                aborted = True
                break
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        pygame.quit()

    if aborted:
        raise SystemExit("aborted")

    video_duration = retime_video_to_duration(raw_output, record_duration_s)
    events_path = raw_output.with_suffix(".events.json")
    write_event_sequence(
        events_path,
        mode=mode,
        duration_s=max(duration_s, video_duration),
        events=actual_events,
    )
    if args.raw_only:
        print(f"wrote raw {raw_output}")
        print(f"wrote events {events_path}")
        print(f"frames={frames} duration={video_duration:.3f}s decoder_hits={len(actual_events)}")
        for t_s, sound in actual_events:
            print(f"  hit t={t_s:.3f}s sound={sound}")
        return 0

    assert final_output is not None
    audio = synth_actual_events(
        actual_events,
        mode=mode,
        duration_s=max(duration_s, video_duration),
        sample_rate=args.sample_rate,
        gain=args.gain,
    )
    wav_path = final_output.with_suffix(".decoder_audio.wav")
    forced_audio.write_wav(wav_path, audio, args.sample_rate)
    forced_audio.mux_video(raw_output, wav_path, final_output)
    wav_path.unlink(missing_ok=True)

    print(f"wrote raw {raw_output}")
    print(f"wrote video {final_output}")
    print(f"frames={frames} duration={video_duration:.3f}s decoder_hits={len(actual_events)}")
    for t_s, sound in actual_events:
        print(f"  hit t={t_s:.3f}s sound={sound}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
