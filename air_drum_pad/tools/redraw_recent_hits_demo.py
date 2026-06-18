#!/usr/bin/env python3
"""Redraw the exported demo's Recent strikes panel from an audio sequence.

The recorded demo video contains the raw decoder's recent-hit text.  For the
final presentation exports we use hybrid audio: target note identity, decoded
timing, and a low-volume decoded noise layer.  This post-process step keeps
that audio unchanged and redraws only the bottom-left recent-hit panel so the
visible text follows the sounds in the final mux.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Source MP4 with final hybrid audio")
    p.add_argument("--sequence", required=True, help="Sequence JSON used to synthesize the audio")
    p.add_argument("--output", required=True, help="Presentation MP4 output")
    p.add_argument("--hold", type=float, default=2.5, help="Seconds to keep a hit in the recent list")
    p.add_argument("--pad-flash", type=float, default=2.5, help="Seconds to keep the newest drum pad highlighted")
    p.add_argument("--max-items", type=int, default=8)
    p.add_argument("--crf", type=int, default=20, help="libx264 quality")
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_DIR / p).resolve()


def load_sequence(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise SystemExit(f"Invalid sequence JSON: {path}")
    return data


def put_text_shadow(
    frame: Any,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
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


def event_label(mode: str, event: dict[str, Any]) -> str:
    sound = str(event.get("sound", ""))
    if mode == "piano" and sound:
        fingers = {
            "G4": "L:thumb",
            "F4": "L:index",
            "E4": "L:middle",
            "D4": "L:ring",
            "C4": "L:pinky",
            "C5": "R:thumb",
            "D5": "R:index",
            "E5": "R:middle",
            "F5": "R:ring",
            "G5": "R:pinky",
        }
        prefix = fingers.get(sound)
        if prefix:
            return f"{prefix} --> {sound}"
    if sound:
        return sound
    return str(event.get("label") or "?")


def event_color(mode: str, event: dict[str, Any]) -> tuple[int, int, int]:
    if mode == "piano":
        colors = {
            "G4": (0, 165, 255),
            "F4": (255, 255, 0),
            "E4": (0, 255, 0),
            "D4": (255, 0, 255),
            "C4": (0, 165, 255),
            "C5": (180, 180, 255),
            "D5": (255, 255, 0),
            "E5": (0, 255, 0),
            "F5": (255, 0, 255),
            "G5": (0, 165, 255),
        }
        return colors.get(str(event.get("sound", "")), (80, 255, 80))
    colors = {
        "kick": (180, 80, 80),
        "snare": (80, 180, 80),
        "hat": (80, 80, 200),
        "ride": (180, 180, 60),
        "tom_l": (60, 180, 180),
        "tom_m": (180, 60, 180),
        "crash": (60, 120, 200),
        "clap": (180, 120, 60),
    }
    return colors.get(str(event.get("sound", "")), (230, 230, 230))


def draw_report_hud(frame: Any, *, mode: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    top_h = 108
    cv2.rectangle(overlay, (0, 0), (w, top_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    title = "PANDA Drum Pads" if mode == "drum" else "PANDA Piano"
    backend = "NPU-FULL:hand_landmark_lite.dxnn+CALIB:npu_landmark_correction.bias.json"
    subtitle = (
        f"live {mode} | backend {backend} | mirror on | q=quit | "
        "fingertip down + joint motion = strike"
    )
    put_text_shadow(frame, title, (24, 48), 1.25, (255, 255, 255), 3)
    put_text_shadow(frame, subtitle, (26, 86), 0.55, (230, 230, 230), 1)


def active_recent_events(
    events: list[dict[str, Any]],
    t_s: float,
    *,
    hold_s: float,
    max_items: int,
) -> list[dict[str, Any]]:
    recent = [
        e for e in events
        if 0.0 <= t_s - float(e.get("t", 0.0)) <= hold_s
    ]
    return list(reversed(recent[-max(1, max_items):]))


def draw_recent_panel(
    frame: Any,
    *,
    mode: str,
    events: list[dict[str, Any]],
    t_s: float,
    hold_s: float,
    max_items: int,
) -> None:
    h, w = frame.shape[:2]
    if mode == "drum":
        # Keep the drum recent-hit UI below the 8-pad grid.  The report-style
        # tom_l pad occupies the lower-left grid cell, so the large live panel
        # used for piano would hide the pad the demo mostly strikes.
        max_items = min(max_items, 3)
        x0 = 24
        line_h = 24
        panel_w = 210
        panel_h = 34 + max(1, max_items) * line_h
        y0 = h - panel_h - 12
        title_scale = 0.50
        item_scale = 0.56
        title_y = y0 + 23
        first_item_y = y0 + 48
        x_text = x0 + 12
    else:
        x0 = 24
        panel_w = min(560, max(320, w // 3))
        line_h = 34
        panel_h = 44 + max(1, max_items) * line_h
        y0 = max(122, h - panel_h - 22)
        title_scale = 0.65
        item_scale = 0.72
        title_y = y0 + 30
        first_item_y = y0 + 64
        x_text = x0 + 16

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    put_text_shadow(frame, "Recent strikes", (x0 + 12, title_y), title_scale, (240, 240, 240), 1)

    recent = active_recent_events(events, t_s, hold_s=hold_s, max_items=max_items)
    for i, event in enumerate(recent):
        label = event_label(mode, event)
        color = event_color(mode, event)
        put_text_shadow(frame, label, (x_text, first_item_y + i * line_h), item_scale, color, 2)


DEMO_DRUM_PADS: tuple[dict[str, Any], ...] = (
    {"label": "kick", "sound": "kick", "x1": 0.05, "y1": 0.35, "x2": 0.265, "y2": 0.59, "color": (180, 80, 80)},
    {"label": "snare", "sound": "snare", "x1": 0.275, "y1": 0.35, "x2": 0.49, "y2": 0.59, "color": (80, 180, 80)},
    {"label": "hat", "sound": "hat", "x1": 0.50, "y1": 0.35, "x2": 0.715, "y2": 0.59, "color": (80, 80, 200)},
    {"label": "ride", "sound": "ride", "x1": 0.725, "y1": 0.35, "x2": 0.94, "y2": 0.59, "color": (180, 180, 60)},
    {"label": "tom_l", "sound": "tom_l", "x1": 0.05, "y1": 0.60, "x2": 0.265, "y2": 0.84, "color": (60, 180, 180)},
    {"label": "tom_m", "sound": "tom_m", "x1": 0.275, "y1": 0.60, "x2": 0.49, "y2": 0.84, "color": (180, 60, 180)},
    {"label": "crash", "sound": "crash", "x1": 0.50, "y1": 0.60, "x2": 0.715, "y2": 0.84, "color": (60, 120, 200)},
    {"label": "clap", "sound": "clap", "x1": 0.725, "y1": 0.60, "x2": 0.94, "y2": 0.84, "color": (180, 120, 60)},
)


def active_sounds(
    events: list[dict[str, Any]],
    t_s: float,
    *,
    flash_s: float,
) -> set[str]:
    recent = [
        e for e in events
        if 0.0 <= t_s - float(e.get("t", 0.0)) <= flash_s
    ]
    if not recent:
        return set()
    return {str(recent[-1].get("sound", ""))}


def draw_drum_pads_from_sequence(
    frame: Any,
    *,
    events: list[dict[str, Any]],
    t_s: float,
    flash_s: float,
) -> None:
    h, w = frame.shape[:2]
    active = active_sounds(events, t_s, flash_s=flash_s)
    overlay = frame.copy()
    draw_items: list[tuple[int, int, int, int, tuple[int, int, int], bool, str]] = []
    for pad in DEMO_DRUM_PADS:
        x1, y1 = int(float(pad["x1"]) * w), int(float(pad["y1"]) * h)
        x2, y2 = int(float(pad["x2"]) * w), int(float(pad["y2"]) * h)
        base_color = tuple(int(c) for c in pad["color"])
        is_active = str(pad["sound"]) in active
        fill_color = tuple(min(255, c + 105) for c in base_color) if is_active else base_color
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        draw_items.append((x1, y1, x2, y2, fill_color, is_active, str(pad["label"])))

    # Strong enough to replace the recorded raw pad flash, light enough to keep
    # the hand/camera view visible.
    cv2.addWeighted(overlay, 0.96, frame, 0.04, 0, frame)
    for x1, y1, x2, y2, color, is_active, label in draw_items:
        thickness = 5 if is_active else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        put_text_shadow(frame, label, (x1 + 8, y1 + 32), 0.8, (255, 255, 255), 2)


def open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    return cap


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    sequence = load_sequence(resolve_path(args.sequence))
    mode = str(sequence.get("mode", "drum"))
    events = sorted(
        [e for e in sequence["events"] if isinstance(e, dict)],
        key=lambda e: float(e.get("t", 0.0)),
    )

    cap = open_capture(input_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise SystemExit("Could not read input dimensions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_s = frame_idx / fps
            draw_report_hud(frame, mode=mode)
            if mode == "drum":
                draw_drum_pads_from_sequence(
                    frame,
                    events=events,
                    t_s=t_s,
                    flash_s=max(0.0, float(args.pad_flash)),
                )
            draw_recent_panel(
                frame,
                mode=mode,
                events=events,
                t_s=t_s,
                hold_s=max(0.0, float(args.hold)),
                max_items=max(1, int(args.max_items)),
            )
            proc.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(ret)
    print(f"wrote {output_path}")
    print(f"frames={frame_idx} fps={fps:.3f} mode={mode} events={len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
