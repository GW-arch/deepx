#!/usr/bin/env python3
"""Mux a recorded demo video with a predefined audio timeline.

This tool intentionally does not run hand tracking. It reads a sequence JSON,
synthesizes the requested piano/drum hits at fixed timestamps, and optionally
muxes that audio onto an input video.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from drumkit_audio import _KIT_BUILDERS, _midi_to_hz, _piano_tone, note_name_to_midi  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", required=True, help="Sequence JSON with fixed event timestamps")
    p.add_argument("--input", help="Recorded source video. Omit when generating only guide WAV.")
    p.add_argument("--output", help="Final MP4 output path")
    p.add_argument("--guide-output", help="Optional standalone WAV guide/click track path")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--audio-offset", type=float, default=0.0, help="Shift sequence audio in seconds")
    p.add_argument(
        "--trim-start",
        type=float,
        default=0.0,
        help="Drop sequence events before this second and subtract it from remaining event times",
    )
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument(
        "--click-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto enables click for guide-only output and disables it for final video muxing",
    )
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_DIR / p).resolve()


def load_sequence(path: str) -> dict[str, Any]:
    data = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Sequence JSON must be an object")
    if "events" not in data or not isinstance(data["events"], list):
        raise SystemExit('Sequence JSON requires an "events" array')
    return data


def ffprobe_duration_s(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def write_wav(path: Path, mono: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    i16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    stereo = np.column_stack((i16, i16)).reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def piano_sample(sound: str, sample_rate: int, duration: float, velocity: float) -> np.ndarray:
    hz = _midi_to_hz(note_name_to_midi(sound))
    return np.asarray(_piano_tone(sample_rate, hz, duration=duration), dtype=np.float32) * velocity


def drum_samples(sample_rate: int) -> dict[str, np.ndarray]:
    np.random.seed(11)
    return {
        key: np.asarray(builder(sample_rate), dtype=np.float32)
        for key, builder in _KIT_BUILDERS.items()
    }


def click_sample(sample_rate: int, *, strong: bool = False) -> np.ndarray:
    duration = 0.055 if strong else 0.045
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    freq = 1200.0 if strong else 880.0
    env = np.exp(-t * 55.0)
    return (0.35 if strong else 0.22) * env * np.sin(2.0 * np.pi * freq * t)


def mix_at(audio: np.ndarray, sample: np.ndarray, t_s: float, sample_rate: int) -> None:
    start = int(round(t_s * sample_rate))
    if start < 0:
        sample = sample[-start:]
        start = 0
    if start >= audio.size or sample.size == 0:
        return
    end = min(audio.size, start + sample.size)
    audio[start:end] += sample[: end - start]


def sequence_duration_s(data: dict[str, Any]) -> float:
    explicit = data.get("duration_s")
    max_event = 0.0
    for event in data["events"]:
        if isinstance(event, dict):
            max_event = max(max_event, float(event.get("t", 0.0)) + float(event.get("duration", 0.5)))
    return max(float(explicit or 0.0), max_event + 0.75)


def trim_sequence(data: dict[str, Any], trim_start_s: float) -> dict[str, Any]:
    trim = max(0.0, float(trim_start_s))
    if trim <= 0.0:
        return data
    out = dict(data)
    events: list[dict[str, Any]] = []
    for event in data["events"]:
        if not isinstance(event, dict):
            continue
        t_s = float(event.get("t", 0.0))
        if t_s < trim:
            continue
        e = dict(event)
        e["t"] = t_s - trim
        events.append(e)
    out["events"] = events
    if "duration_s" in out:
        out["duration_s"] = max(0.0, float(out["duration_s"]) - trim)
    return out


def synthesize(
    data: dict[str, Any],
    *,
    sample_rate: int,
    offset_s: float,
    gain: float,
    include_click: bool,
) -> np.ndarray:
    mode = str(data.get("mode", "drum"))
    duration_s = sequence_duration_s(data)
    total = int(math.ceil((duration_s + max(0.0, offset_s)) * sample_rate))
    audio = np.zeros(total, dtype=np.float32)
    drums = drum_samples(sample_rate) if mode == "drum" else {}

    click = data.get("click", {})
    if include_click and isinstance(click, dict) and click.get("enabled", False):
        beat_s = float(click.get("beat_s", 0.5))
        start_s = float(click.get("start_s", 0.0))
        end_s = float(click.get("end_s", duration_s))
        accent_every = int(click.get("accent_every", 4))
        i = 0
        t = start_s
        while t <= end_s + 1e-9:
            mix_at(audio, click_sample(sample_rate, strong=(i % max(1, accent_every) == 0)), t + offset_s, sample_rate)
            i += 1
            t += beat_s

    for event in data["events"]:
        if not isinstance(event, dict):
            continue
        t_s = float(event.get("t", 0.0)) + offset_s
        sound = str(event.get("sound", ""))
        velocity = float(event.get("velocity", 1.0))
        if mode == "piano":
            duration = float(event.get("duration", 0.52))
            sample = piano_sample(sound, sample_rate, duration, velocity)
        else:
            sample = drums.get(sound)
            if sample is None:
                raise SystemExit(f"Unknown drum sound: {sound!r}")
            sample = sample * velocity
        mix_at(audio, sample, t_s, sample_rate)

    audio *= float(gain)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio *= 0.98 / peak
    return audio


def mux_video(input_path: Path, wav_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-i",
        str(wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    data = trim_sequence(load_sequence(args.sequence), args.trim_start)
    if args.click_mode == "auto":
        include_click = bool(args.guide_output and not args.input)
    else:
        include_click = args.click_mode == "on"

    audio = synthesize(
        data,
        sample_rate=args.sample_rate,
        offset_s=args.audio_offset,
        gain=args.gain,
        include_click=include_click,
    )

    if args.guide_output:
        guide_path = resolve_path(args.guide_output)
        write_wav(guide_path, audio, args.sample_rate)
        print(f"wrote guide {guide_path}")

    if args.input or args.output:
        if not args.input or not args.output:
            raise SystemExit("--input and --output must be provided together")
        input_path = resolve_path(args.input)
        output_path = resolve_path(args.output)
        wav_path = output_path.with_suffix(".forced_audio.wav")
        video_duration = ffprobe_duration_s(input_path)
        if video_duration > audio.size / args.sample_rate:
            pad = int(math.ceil(video_duration * args.sample_rate)) - audio.size
            audio = np.pad(audio, (0, max(0, pad)))
        write_wav(wav_path, audio, args.sample_rate)
        mux_video(input_path, wav_path, output_path)
        wav_path.unlink(missing_ok=True)
        print(f"wrote video {output_path}")

    if not args.guide_output and not args.input:
        raise SystemExit("Provide --guide-output or --input/--output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
