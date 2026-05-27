#!/usr/bin/env python3
"""Guided live evaluator for Air-Drum/Piano latency and target accuracy.

The evaluator shows a predefined timed sequence on screen, logs the first
system-detected strike events, and writes cue/event/match CSV files plus a
summary report.  The reported latency is cue-to-detection latency: it includes
human reaction time, camera exposure/readout, inference, and strike detection.
True acoustic motion-to-speaker latency still requires an external synchronized
video/audio measurement, but these logs provide the controlled timing ground
truth needed for the project demo and report experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DRUM_DEFAULT_TARGETS: tuple[str, ...] = (
    "kick",
    "snare",
    "hat",
    "ride",
    "tom_l",
    "tom_m",
    "crash",
    "clap",
)
PIANO_DEFAULT_TARGETS: tuple[str, ...] = (
    "C4",
    "D4",
    "E4",
    "F4",
    "G4",
    "C5",
    "D5",
    "E5",
    "F5",
    "G5",
)
FINGER_HINTS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class Cue:
    """A scheduled prompt shown to the performer."""

    cue_id: int
    t_s: float
    target: str


@dataclass(frozen=True)
class Event:
    """A strike event detected by the running Air-Drum pipeline."""

    event_id: int
    t_s: float
    label: str
    detail: str = ""
    frame_id: int = 0


@dataclass(frozen=True)
class Match:
    """Evaluation result for one cue."""

    cue_id: int
    target: str
    cue_t_s: float
    event_id: int | None
    event_t_s: float | None
    latency_s: float | None
    outcome: str


@dataclass(frozen=True)
class EvalResult:
    matches: list[Match]
    summary: dict[str, Any]


def normalize_label(label: str, mode: str) -> str:
    """Normalize human-entered labels while preserving the app's labels."""
    text = str(label).strip()
    if mode == "drum":
        return text.lower()
    if not text:
        return text
    return text[0].upper() + text[1:]


def default_targets(mode: str) -> tuple[str, ...]:
    if mode == "drum":
        return DRUM_DEFAULT_TARGETS
    if mode == "piano":
        return PIANO_DEFAULT_TARGETS
    raise ValueError(f"unsupported mode: {mode!r}")


def parse_targets(raw: str | None, mode: str) -> tuple[str, ...]:
    """Parse comma-separated targets; fall back to the mode default."""
    if raw is None or not raw.strip():
        return default_targets(mode)
    targets = tuple(normalize_label(x, mode) for x in raw.split(",") if x.strip())
    if not targets:
        raise ValueError("--targets did not contain any non-empty labels")
    return targets


def build_cues(
    targets: Sequence[str],
    *,
    bpm: float,
    repeats: int,
    lead_in_s: float,
) -> list[Cue]:
    """Build a repeated target sequence at a fixed beat rate."""
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if lead_in_s < 0:
        raise ValueError("lead_in_s must be non-negative")
    clean_targets = [str(x).strip() for x in targets if str(x).strip()]
    if not clean_targets:
        raise ValueError("at least one target is required")

    period_s = 60.0 / bpm
    cues: list[Cue] = []
    i = 0
    for _ in range(repeats):
        for target in clean_targets:
            cues.append(Cue(cue_id=i + 1, t_s=lead_in_s + i * period_s, target=target))
            i += 1
    return cues


def load_cues_json(path: str, mode: str) -> list[Cue]:
    """Load an explicit cue sequence from JSON.

    Supported formats:
      ["kick", "snare"]
      {"bpm": 90, "lead_in_s": 2, "repeats": 3, "targets": ["kick", "snare"]}
      {"sequence": [{"t_s": 2.0, "target": "C4"}, ...]}
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        targets = [normalize_label(x, mode) for x in raw]
        return build_cues(targets, bpm=90.0, repeats=1, lead_in_s=2.0)
    if not isinstance(raw, dict):
        raise ValueError("sequence JSON must be a list or an object")

    if "sequence" in raw:
        seq = raw["sequence"]
        if not isinstance(seq, list) or not seq:
            raise ValueError("sequence must be a non-empty list")
        cues: list[Cue] = []
        for i, item in enumerate(seq):
            if isinstance(item, str):
                cues.append(Cue(i + 1, float(i), normalize_label(item, mode)))
                continue
            if not isinstance(item, dict):
                raise ValueError(f"sequence[{i}] must be a string or object")
            target = normalize_label(str(item.get("target", "")), mode)
            if not target:
                raise ValueError(f"sequence[{i}].target is required")
            if "t_s" in item:
                t_s = float(item["t_s"])
            elif "time_s" in item:
                t_s = float(item["time_s"])
            else:
                raise ValueError(f"sequence[{i}] must include t_s or time_s")
            cues.append(Cue(i + 1, t_s, target))
        return cues

    targets_raw = raw.get("targets") or raw.get("sequence_targets")
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("sequence JSON object must contain targets or sequence")
    targets = [normalize_label(x, mode) for x in targets_raw]
    return build_cues(
        targets,
        bpm=float(raw.get("bpm", 90.0)),
        repeats=int(raw.get("repeats", 1)),
        lead_in_s=float(raw.get("lead_in_s", raw.get("lead_in", 2.0))),
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _ms(value_s: float | None) -> float | None:
    if value_s is None:
        return None
    return value_s * 1000.0


def match_events(
    cues: Sequence[Cue],
    events: Sequence[Event],
    *,
    pre_window_s: float = 0.20,
    post_window_s: float = 0.70,
) -> EvalResult:
    """Greedily match same-label events to cues by closest cue time.

    Cues without a matching same-label event are false negatives.  Unmatched
    events inside the trial window are false positives.  The match latency is
    signed: positive values mean the detector fired after the visual cue.
    """
    if pre_window_s < 0 or post_window_s < 0:
        raise ValueError("matching windows must be non-negative")

    clean_cues = list(cues)
    clean_events = list(events)
    if not clean_cues:
        fp = len(clean_events)
        summary = {
            "cue_count": 0,
            "event_count": len(clean_events),
            "tp": 0,
            "fp": fp,
            "fn": 0,
            "precision": 0.0 if fp else None,
            "recall": None,
            "target_accuracy": None,
        }
        return EvalResult(matches=[], summary=summary)

    candidates: list[tuple[float, int, int]] = []
    for ci, cue in enumerate(clean_cues):
        for ei, event in enumerate(clean_events):
            if event.label != cue.target:
                continue
            latency = event.t_s - cue.t_s
            if -pre_window_s <= latency <= post_window_s:
                candidates.append((abs(latency), ci, ei))

    used_cues: set[int] = set()
    used_events: set[int] = set()
    assignments: dict[int, int] = {}
    for _distance, ci, ei in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
        if ci in used_cues or ei in used_events:
            continue
        used_cues.add(ci)
        used_events.add(ei)
        assignments[ci] = ei

    matches: list[Match] = []
    latencies_s: list[float] = []
    for ci, cue in enumerate(clean_cues):
        ei = assignments.get(ci)
        if ei is None:
            matches.append(
                Match(
                    cue_id=cue.cue_id,
                    target=cue.target,
                    cue_t_s=cue.t_s,
                    event_id=None,
                    event_t_s=None,
                    latency_s=None,
                    outcome="miss",
                )
            )
            continue
        event = clean_events[ei]
        latency = event.t_s - cue.t_s
        latencies_s.append(latency)
        matches.append(
            Match(
                cue_id=cue.cue_id,
                target=cue.target,
                cue_t_s=cue.t_s,
                event_id=event.event_id,
                event_t_s=event.t_s,
                latency_s=latency,
                outcome="hit",
            )
        )

    trial_start = clean_cues[0].t_s - pre_window_s
    trial_end = clean_cues[-1].t_s + post_window_s
    events_in_trial = [
        i for i, event in enumerate(clean_events) if trial_start <= event.t_s <= trial_end
    ]
    fp = sum(1 for i in events_in_trial if i not in used_events)
    tp = len(assignments)
    fn = len(clean_cues) - tp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / len(clean_cues) if clean_cues else None
    abs_latencies_s = [abs(x) for x in latencies_s]

    summary = {
        "cue_count": len(clean_cues),
        "event_count": len(clean_events),
        "events_in_trial_window": len(events_in_trial),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "target_accuracy": recall,
        "pre_window_ms": pre_window_s * 1000.0,
        "post_window_ms": post_window_s * 1000.0,
        "mean_latency_ms": _ms(_mean(latencies_s)),
        "median_latency_ms": _ms(_percentile(latencies_s, 0.50)),
        "std_latency_ms": _ms(_std(latencies_s)),
        "mean_abs_latency_ms": _ms(_mean(abs_latencies_s)),
        "p95_abs_latency_ms": _ms(_percentile(abs_latencies_s, 0.95)),
    }
    return EvalResult(matches=matches, summary=summary)


def write_outputs(
    output_dir: Path,
    *,
    cues: Sequence[Cue],
    events: Sequence[Event],
    matches: Sequence[Match],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write cues/events/matches and human-readable summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, rows: Sequence[Any], fieldnames: Sequence[str]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    write_csv("cues.csv", cues, ("cue_id", "t_s", "target"))
    write_csv("events.csv", events, ("event_id", "t_s", "label", "detail", "frame_id"))
    write_csv(
        "matches.csv",
        matches,
        ("cue_id", "target", "cue_t_s", "event_id", "event_t_s", "latency_s", "outcome"),
    )

    result_obj = {"summary": summary, "metadata": metadata}
    (output_dir / "summary.json").write_text(
        json.dumps(result_obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}{suffix}"
        return f"{value}{suffix}"

    md = [
        "# Guided Latency and Accuracy Evaluation",
        "",
        "## Scope",
        "",
        "This run measures cue-to-detection latency and target accuracy from a controlled on-screen sequence. The latency includes human reaction, camera capture, inference, and strike detection. It is not a direct acoustic motion-to-speaker measurement unless paired with an external synchronized audio/video capture.",
        "",
        "## Summary",
        "",
        f"- Mode: `{metadata.get('mode', 'unknown')}`",
        f"- Backend: `{metadata.get('backend', 'unknown')}`",
        f"- Cues: {summary.get('cue_count', 0)}",
        f"- Detected events: {summary.get('event_count', 0)}",
        f"- True positives: {summary.get('tp', 0)}",
        f"- False positives: {summary.get('fp', 0)}",
        f"- False negatives: {summary.get('fn', 0)}",
        f"- Precision: {fmt(summary.get('precision'))}",
        f"- Recall / target accuracy: {fmt(summary.get('recall'))}",
        f"- Mean signed latency: {fmt(summary.get('mean_latency_ms'), ' ms')}",
        f"- Median signed latency: {fmt(summary.get('median_latency_ms'), ' ms')}",
        f"- Mean absolute latency: {fmt(summary.get('mean_abs_latency_ms'), ' ms')}",
        f"- P95 absolute latency: {fmt(summary.get('p95_abs_latency_ms'), ' ms')}",
        "",
        "## Files",
        "",
        "- `cues.csv`: scheduled target prompts.",
        "- `events.csv`: detected strike events.",
        "- `matches.csv`: cue-to-event matches and per-cue latency.",
        "- `summary.json`: machine-readable summary and run metadata.",
        "- `review.mp4`: optional overlay recording when video recording is enabled.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def piano_target_hints(slots: Sequence[str]) -> dict[str, str]:
    """Build note -> suggested finger labels for the default 10-slot piano layout."""
    hints: dict[str, str] = {}
    hands = ("left", "right")
    for hand_i, hand_name in enumerate(hands):
        for finger_i, note in enumerate(slots[hand_i * 5 : hand_i * 5 + 5]):
            if note not in hints:
                hints[note] = f"{hand_name} {FINGER_HINTS[finger_i]}"
    return hints


def _put_text_with_shadow(
    cv2: Any,
    frame: Any,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _make_beep_sound(pygame: Any, np: Any, *, sample_rate: int = 22050) -> Any:
    duration_s = 0.065
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    env = np.exp(-t * 18.0) * (1.0 - np.exp(-t * 240.0))
    wave = 0.35 * env * np.sin(2.0 * np.pi * 1760.0 * t)
    i16 = (np.clip(wave, -1.0, 1.0) * 32767.0).astype(np.int16)
    stereo = np.column_stack((i16, i16))
    return pygame.sndarray.make_sound(stereo)


def _draw_guidance_overlay(
    cv2: Any,
    frame: Any,
    *,
    mode: str,
    cues: Sequence[Cue],
    current_idx: int,
    elapsed_s: float,
    pre_window_s: float,
    post_window_s: float,
    tp_so_far: int,
    fp_so_far: int,
    hints: dict[str, str],
) -> None:
    h, w = frame.shape[:2]
    if current_idx >= len(cues):
        target = "DONE"
        next_text = "Press q to close"
        dt = 0.0
    else:
        cue = cues[current_idx]
        target = cue.target
        dt = cue.t_s - elapsed_s
        next_target = cues[current_idx + 1].target if current_idx + 1 < len(cues) else "end"
        next_text = f"Next: {next_target}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 155), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    phase = "WAIT"
    color = (0, 220, 255)
    if current_idx < len(cues):
        if dt > 0.35:
            phase = f"READY {dt:.1f}s"
            color = (0, 220, 255)
        elif dt > 0.0:
            phase = "GET READY"
            color = (0, 255, 255)
        elif elapsed_s <= cues[current_idx].t_s + post_window_s:
            phase = "HIT NOW"
            color = (0, 255, 0)
        else:
            phase = "NEXT"
            color = (0, 165, 255)

    _put_text_with_shadow(cv2, frame, phase, (18, 40), 1.0, color, 2)
    _put_text_with_shadow(cv2, frame, target, (18, 112), 2.0, (255, 255, 255), 4)
    if target in hints:
        _put_text_with_shadow(cv2, frame, hints[target], (300, 112), 0.9, (180, 220, 255), 2)
    _put_text_with_shadow(
        cv2,
        frame,
        f"{mode} | cue {min(current_idx + 1, len(cues))}/{len(cues)} | TP {tp_so_far} FP {fp_so_far} | {next_text}",
        (18, 142),
        0.55,
        (230, 230, 230),
        1,
    )
    _put_text_with_shadow(
        cv2,
        frame,
        f"window: -{pre_window_s:.2f}s/+{post_window_s:.2f}s | q=quit",
        (max(20, w - 430), max(32, h - 18)),
        0.48,
        (230, 230, 230),
        1,
    )


def _draw_pad_targets(cv2: Any, frame: Any, pads: Sequence[Any], current_target: str | None) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for pad in pads:
        x1, y1 = int(pad.x1 * w), int(pad.y1 * h)
        x2, y2 = int(pad.x2 * w), int(pad.y2 * h)
        active = pad.label == current_target
        fill = tuple(min(255, int(c) + (90 if active else 0)) for c in pad.color)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    for pad in pads:
        x1, y1 = int(pad.x1 * w), int(pad.y1 * h)
        x2, y2 = int(pad.x2 * w), int(pad.y2 * h)
        active = pad.label == current_target
        border = (255, 255, 255) if active else tuple(int(c) for c in pad.color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 4 if active else 2, cv2.LINE_AA)
        label = str(pad.label)
        _put_text_with_shadow(cv2, frame, label, (x1 + 10, y1 + 32), 0.75, (255, 255, 255), 2)


def _draw_landmarks(cv2: Any, np: Any, frame: Any, hand_lms: Any, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    for tip_id in (4, 8, 12, 16, 20):
        lm = hand_lms.landmark[tip_id]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 7, color, -1)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)
    # minimal skeleton without importing MediaPipe drawing utilities
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


def _piano_targets_from_slots(slots: Sequence[str]) -> tuple[str, ...]:
    try:
        from drumkit_audio import note_name_to_midi

        left = tuple(sorted(slots[:5], key=note_name_to_midi))
        right = tuple(sorted(slots[5:10], key=note_name_to_midi))
        return left + right
    except Exception:
        return tuple(slots[:10])


def run_live(args: argparse.Namespace) -> int:
    # Heavy dependencies are imported only in the live path so unit tests can
    # exercise the evaluator math without camera/audio/display setup.
    import cv2
    import numpy as np

    from drumkit_audio import (
        PIANO_DEFAULT_SLOTS,
        build_kit,
        build_piano_kit_for_slots,
        load_piano_slots_json,
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

    os.chdir(PROJECT_DIR)

    mode = args.mode
    slots: tuple[str, ...] = PIANO_DEFAULT_SLOTS
    pad_zones: list[Any] = []
    if mode == "piano":
        piano_json = args.instruments.strip() if args.instruments.strip() else "instruments.piano.example.json"
        slots = load_piano_slots_json(piano_json) if Path(piano_json).is_file() else PIANO_DEFAULT_SLOTS
        target_default = _piano_targets_from_slots(slots)
        hints = piano_target_hints(slots)
    else:
        if args.drum_pads.strip():
            # Use the known built-in drum keys for validation without opening audio first.
            from drumkit_audio import kit_keys

            pad_zones = load_pad_zones_json(args.drum_pads.strip(), frozenset(kit_keys()))
        else:
            pad_zones = default_pad_zones()
        target_default = tuple(p.label for p in pad_zones)
        hints = {}

    if args.sequence_json.strip():
        cues = load_cues_json(args.sequence_json.strip(), mode)
    else:
        targets = parse_targets(args.targets, mode) if args.targets else target_default
        cues = build_cues(targets, bpm=args.bpm, repeats=args.repeats, lead_in_s=args.lead_in)

    output_dir = Path(args.output_dir) if args.output_dir.strip() else Path("eval_runs") / datetime.now().strftime(f"%Y%m%d_%H%M%S_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_sound = not args.no_sound
    kit: dict[str, Any] = {}
    cue_sound: Any | None = None
    pygame: Any | None = None
    if use_sound:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame as _pygame

            pygame = _pygame
            if mode == "piano":
                kit = build_piano_kit_for_slots(slots)
            else:
                kit = build_kit()
            cue_sound = _make_beep_sound(pygame, np)
        except Exception as exc:  # pragma: no cover - depends on local audio device
            print(f"[guided-eval] audio unavailable ({exc}); continuing with --no-sound behavior", file=sys.stderr)
            use_sound = False
            kit = {}
            cue_sound = None

    if mode == "piano":
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
        pad_det: PadStrikeDetector | None = None
    else:
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
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {args.camera}")

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
        landmark_correction=args.landmark_correction if args.landmark_correction.strip() else None,
    )

    title = "PANDA Guided Eval"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    writer: Any | None = None
    if not args.no_record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_dir / "review.mp4"), fourcc, 30.0, (args.width, args.height))

    events: list[Event] = []
    beeped_cues: set[int] = set()
    start_t = time.perf_counter()
    frame_id = 0
    last_flash: tuple[float, str] | None = None

    print(
        f"[guided-eval] mode={mode} backend={args.backend} cues={len(cues)} output={output_dir}",
        flush=True,
    )
    print("[guided-eval] q=quit early; results are still written", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[guided-eval] camera read failed", file=sys.stderr)
                break
            frame_id += 1
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            elapsed = now - start_t
            while len(beeped_cues) < len(cues) and cues[len(beeped_cues)].t_s <= elapsed:
                cue = cues[len(beeped_cues)]
                beeped_cues.add(cue.cue_id)
                if cue_sound is not None:
                    cue_sound.play()

            current_idx = 0
            while current_idx < len(cues) and elapsed > cues[current_idx].t_s + args.post_window:
                current_idx += 1
            current_target = cues[current_idx].target if current_idx < len(cues) else None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = tracker.process(rgb)
            landmarks_list = res.multi_hand_landmarks or []
            handedness_list = res.multi_handedness or []

            for hand_idx, hand_lms in enumerate(landmarks_list):
                conf = 1.0
                hand_label = f"H{hand_idx}"
                if hand_idx < len(handedness_list):
                    cls = handedness_list[hand_idx].classification[0]
                    conf = float(cls.score)
                    hand_label = f"{cls.label[0].upper()}{hand_idx}"
                _draw_landmarks(cv2, np, frame, hand_lms, (0, 255, 255))
                wrist = hand_lms.landmark[0]
                cv2.putText(
                    frame,
                    hand_label,
                    (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                for fid in FINGERTIP_INDICES:
                    if mode == "piano":
                        assert det is not None
                        hit = det.update_finger(hand_idx, fid, elapsed, hand_lms, conf)
                        if not hit:
                            continue
                        _track_id, label = hit
                        label = normalize_label(label, mode)
                        if label in kit:
                            kit[label].play()
                        detail = f"{hand_label}:{FINGER_LABELS.get(fid, fid)}"
                    else:
                        assert pad_det is not None
                        hit_pad = pad_det.update_finger(hand_idx, fid, elapsed, hand_lms, conf)
                        if not hit_pad:
                            continue
                        label = normalize_label(hit_pad.label, mode)
                        if hit_pad.sound_key in kit:
                            kit[hit_pad.sound_key].play()
                        detail = f"{hand_label}:{FINGER_LABELS.get(fid, fid)}:{hit_pad.sound_key}"

                    event = Event(
                        event_id=len(events) + 1,
                        t_s=elapsed,
                        label=label,
                        detail=detail,
                        frame_id=frame_id,
                    )
                    events.append(event)
                    last_flash = (elapsed + 0.35, label)

            partial = match_events(cues[: max(0, current_idx)], events, pre_window_s=args.pre_window, post_window_s=args.post_window)
            tp_so_far = int(partial.summary.get("tp", 0))
            fp_so_far = int(partial.summary.get("fp", 0))

            if mode == "drum":
                _draw_pad_targets(cv2, frame, pad_zones, current_target)
            _draw_guidance_overlay(
                cv2,
                frame,
                mode=mode,
                cues=cues,
                current_idx=current_idx,
                elapsed_s=elapsed,
                pre_window_s=args.pre_window,
                post_window_s=args.post_window,
                tp_so_far=tp_so_far,
                fp_so_far=fp_so_far,
                hints=hints,
            )
            if last_flash is not None and elapsed <= last_flash[0]:
                _put_text_with_shadow(cv2, frame, f"detected: {last_flash[1]}", (18, frame.shape[0] - 48), 0.8, (0, 255, 255), 2)

            if writer is not None:
                writer.write(frame)
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if elapsed > cues[-1].t_s + args.post_window + args.finish_hold:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        tracker.close()
        if pygame is not None:
            pygame.quit()

    result = match_events(cues, events, pre_window_s=args.pre_window, post_window_s=args.post_window)
    metadata = {
        "mode": mode,
        "backend": args.backend,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bpm": args.bpm,
        "repeats": args.repeats,
        "lead_in_s": args.lead_in,
        "targets": [c.target for c in cues],
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "vy_trigger": args.vy_trigger,
        "joint_dps": args.joint_dps,
        "cooldown_s": args.cooldown,
        "mirror_view": not args.no_mirror,
        "note": "Latency is cue-to-detection latency, not independently verified acoustic motion-to-speaker latency.",
    }
    write_outputs(
        output_dir,
        cues=cues,
        events=events,
        matches=result.matches,
        summary=result.summary,
        metadata=metadata,
    )
    print(json.dumps(result.summary, indent=2), flush=True)
    print(f"[guided-eval] wrote {output_dir}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Show predefined beat/target cues and measure cue-to-detection "
            "latency plus target accuracy from the live Air-Drum pipeline."
        )
    )
    p.add_argument("--mode", choices=("drum", "piano"), default="drum")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--bpm", type=float, default=72.0, help="Cue tempo; lower is easier for human trials")
    p.add_argument("--repeats", type=int, default=3, help="Repeat the target list this many times")
    p.add_argument("--lead-in", type=float, default=3.0, help="Seconds before first cue")
    p.add_argument("--pre-window", type=float, default=0.20, help="Accept hits this many seconds before cue")
    p.add_argument("--post-window", type=float, default=0.70, help="Accept hits this many seconds after cue")
    p.add_argument("--finish-hold", type=float, default=2.0, help="Seconds to keep recording after the last window")
    p.add_argument("--targets", type=str, default="", help="Comma-separated target labels/notes")
    p.add_argument("--sequence-json", type=str, default="", help="Optional explicit sequence JSON")
    p.add_argument("--output-dir", type=str, default="", help="Default: eval_runs/YYYYmmdd_HHMMSS_mode")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable the default mirrored selfie view.",
    )
    p.add_argument("--no-sound", action="store_true", help="Disable cue beep and instrument playback")
    p.add_argument("--no-record", action="store_true", help="Do not write review.mp4")

    p.add_argument("--vy-trigger", type=float, default=0.025)
    p.add_argument("--joint-dps", type=float, default=16.0)
    p.add_argument("--cooldown", type=float, default=0.10)
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2))
    p.add_argument("--model-complexity", type=int, default=0, choices=(0, 1))
    p.add_argument("--instruments", type=str, default="", help="Piano slot JSON; default instruments.piano.example.json")
    p.add_argument("--drum-pads", type=str, default="", help="Drum pad layout JSON")

    p.add_argument("--backend", choices=("cpu", "cpu-baseline", "npu", "npu-full"), default="cpu")
    p.add_argument("--dxnn", type=str, default="")
    p.add_argument("--dxnn-layout", type=str, default="")
    p.add_argument("--palm-tflite", type=str, default="")
    p.add_argument("--palm-dxnn", type=str, default="")
    p.add_argument("--hand-tflite", type=str, default="")
    p.add_argument("--palm-redetect-every", type=int, default=0)
    p.add_argument("--async-palm", action="store_true")
    p.add_argument("--landmark-correction", type=str, default="")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
