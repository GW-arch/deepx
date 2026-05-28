#!/usr/bin/env python3
"""PANDA demo launcher for an easy "We Will Rock You"-style air-drum video.

This script does not bundle copyrighted music.  For an actual Queen backing
track, pass your own legally obtained audio file:

    python3 demo_run.py --backing-track ~/Music/we_will_rock_you.mp3

If no backing track is supplied, the script first looks for a single audio file
in ``dataset/``.  If none is found, it generates a simple royalty-free
"stomp stomp clap" guide loop so the live PANDA drum sounds can be recorded on
top of it.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
BACKING_EXTENSIONS = (".mp3", ".wav", ".ogg")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=(
            "Run PANDA drum mode with an optional background track and a very "
            "easy kick-kick-snare demo pattern."
        ),
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backing-track", type=str, default="", metavar="PATH")
    p.add_argument("--backing-volume", type=float, default=0.35)
    p.add_argument("--loop", action="store_true", help="Loop the backing track.")
    p.add_argument(
        "--bpm",
        type=float,
        default=80.0,
        help="Tempo for the generated guide loop when --backing-track is omitted.",
    )
    p.add_argument(
        "--bars",
        type=int,
        default=40,
        help="Number of generated guide bars when --backing-track is omitted.",
    )
    p.add_argument(
        "--guide-out",
        type=str,
        default="/tmp/panda_we_will_rock_you_style_guide.wav",
        help="Where to write the generated royalty-free guide WAV.",
    )
    p.add_argument(
        "--no-guide",
        action="store_true",
        help="Do not generate a guide loop when no backing audio is found.",
    )
    p.add_argument(
        "--print-pattern",
        action="store_true",
        help="Print the demo pattern and exit.",
    )
    args, passthrough = p.parse_known_args()
    return args, passthrough


def print_pattern() -> None:
    print(
        "\nPANDA air-drum demo pattern (We Will Rock You-style):\n\n"
        "Count:  1   &   2   &   3   &   4   &\n"
        "Pads:   kick kick snare -   kick kick snare -\n\n"
        "Easy ending: repeat for 3 bars, then use crash on the final hit:\n"
        "        kick kick snare | kick kick crash\n"
    )


def find_dataset_backing_track() -> Path | None:
    """Auto-select one user-supplied audio file from dataset/, if unambiguous."""
    if not DATASET_DIR.is_dir():
        return None
    candidates = sorted(
        p
        for p in DATASET_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in BACKING_EXTENSIONS
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(
            "Multiple audio files found in dataset/; pass --backing-track explicitly:\n"
            + "\n".join(f"  - {p}" for p in candidates),
            file=sys.stderr,
        )
    return None


def _envelope(n: int, attack: float = 0.002, decay: float = 0.12, sr: int = 44100) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t / max(decay, 1e-6))
    a = max(1, int(attack * sr))
    env[:a] *= np.linspace(0.0, 1.0, a)
    return env


def _stomp(sr: int) -> np.ndarray:
    n = int(0.22 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    # Low, dull thump: short chirp plus a little filtered-looking noise.
    f0, f1 = 95.0, 42.0
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-6) * t * t)
    tone = np.sin(phase)
    noise = np.random.default_rng(7).uniform(-0.25, 0.25, n)
    return 0.75 * (tone + noise) * _envelope(n, decay=0.08, sr=sr)


def _clap(sr: int) -> np.ndarray:
    n = int(0.18 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(11)
    noise = rng.uniform(-1.0, 1.0, n)
    env = (
        np.exp(-((t - 0.010) ** 2) / 0.000020)
        + 0.75 * np.exp(-((t - 0.032) ** 2) / 0.000030)
        + 0.35 * np.exp(-((t - 0.055) ** 2) / 0.000050)
    )
    return 0.45 * noise * env


def _count_tick(sr: int) -> np.ndarray:
    n = int(0.07 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    return 0.30 * np.sin(2.0 * math.pi * 880.0 * t) * _envelope(n, decay=0.025, sr=sr)


def _mix_at(buf: np.ndarray, sample: np.ndarray, t_s: float, sr: int) -> None:
    start = max(0, int(round(t_s * sr)))
    end = min(buf.shape[0], start + sample.shape[0])
    if end > start:
        buf[start:end] += sample[: end - start]


def generate_guide(path: Path, *, bpm: float, bars: int, sr: int = 44100) -> Path:
    """Generate a simple non-copyright guide: count-in + kick-kick-clap loop."""
    bpm = max(40.0, float(bpm))
    bars = max(1, int(bars))
    beat_s = 60.0 / bpm
    bar_s = beat_s * 4.0
    count_in_bars = 1
    total_s = (count_in_bars + bars) * bar_s + 0.5
    buf = np.zeros(int(total_s * sr), dtype=np.float64)
    stomp = _stomp(sr)
    clap = _clap(sr)
    tick = _count_tick(sr)

    # Four-count intro.
    for beat in range(4):
        _mix_at(buf, tick, beat * beat_s, sr)

    start_s = count_in_bars * bar_s
    for bar in range(bars):
        b0 = start_s + bar * bar_s
        # 1, &, 2, then 3, &, 4.
        for offset, sample in (
            (0.0, stomp),
            (0.5 * beat_s, stomp),
            (1.0 * beat_s, clap),
            (2.0 * beat_s, stomp),
            (2.5 * beat_s, stomp),
            (3.0 * beat_s, clap),
        ):
            _mix_at(buf, sample, b0 + offset, sr)

    # Gentle limiter and stereo conversion.
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = np.tanh(buf / max(0.75, peak) * 1.2) * 0.55
    stereo = np.column_stack([buf, buf])
    i16 = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())
    return path


def main() -> int:
    args, passthrough = parse_args()
    if args.print_pattern:
        print_pattern()
        return 0

    backing = args.backing_track.strip()
    if backing:
        backing_path = Path(backing).expanduser()
    elif (dataset_backing := find_dataset_backing_track()) is not None:
        backing_path = dataset_backing
        print(f"Using dataset backing track: {backing_path}", flush=True)
    elif args.no_guide:
        backing_path = None
    else:
        backing_path = generate_guide(
            Path(args.guide_out).expanduser(),
            bpm=args.bpm,
            bars=args.bars,
        )
        print(f"Generated royalty-free guide loop: {backing_path}", flush=True)

    print_pattern()

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "main.py"),
        "--camera",
        str(args.camera),
    ]
    if backing_path is not None:
        cmd += [
            "--backing-track",
            str(backing_path),
            "--backing-volume",
            str(args.backing_volume),
        ]
        if args.loop:
            cmd.append("--backing-loop")
    cmd += passthrough

    print("Launching:", " ".join(cmd), flush=True)
    os.execv(sys.executable, cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
