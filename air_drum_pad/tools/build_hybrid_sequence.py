#!/usr/bin/env python3
"""Build a plausible target+decoded hybrid audio sequence.

Target events provide the intended note/drum identity. Actual decoded events
provide human timing when a nearby decoded hit exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for p in (PROJECT_DIR, TOOLS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import render_forced_audio_demo as forced_audio  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--actual", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trim-start", type=float, default=0.0)
    p.add_argument("--lead", type=float, default=0.25, help="Allow actual hit this much before target")
    p.add_argument("--lag", type=float, default=0.95, help="Allow actual hit this much after target")
    p.add_argument("--actual-weight", type=float, default=0.72)
    p.add_argument("--fallback-velocity", type=float, default=0.82)
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.34,
        help="Minimum seconds between generated audio events to suppress duplicate-hit noise",
    )
    p.add_argument(
        "--actual-noise-velocity",
        type=float,
        default=0.0,
        help="Add unmatched actual decoded hits as a low-volume realism/noise layer",
    )
    p.add_argument("--noise-min-gap", type=float, default=0.22)
    p.add_argument("--noise-exclusion", type=float, default=0.08)
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_DIR / p).resolve()


def events(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in data.get("events", []) if isinstance(e, dict)]


def nearest_unused_actual(
    actual: list[dict[str, Any]],
    used: set[int],
    target_t: float,
    *,
    lead: float,
    lag: float,
) -> tuple[int | None, dict[str, Any] | None]:
    best_i: int | None = None
    best_event: dict[str, Any] | None = None
    best_abs_dt = 1e9
    for i, event in enumerate(actual):
        if i in used:
            continue
        t = float(event.get("t", 0.0))
        dt = t - target_t
        if dt < -lead or dt > lag:
            continue
        abs_dt = abs(dt)
        if abs_dt < best_abs_dt:
            best_i = i
            best_event = event
            best_abs_dt = abs_dt
    return best_i, best_event


def main() -> int:
    args = parse_args()
    target = forced_audio.load_sequence(str(resolve_path(args.target)))
    actual = forced_audio.trim_sequence(
        forced_audio.load_sequence(str(resolve_path(args.actual))),
        args.trim_start,
    )
    target_events = events(target)
    actual_events = events(actual)
    used: set[int] = set()
    hybrid_events: list[dict[str, Any]] = []
    primary_times: list[float] = []
    matches = 0
    actual_weight = max(0.0, min(1.0, float(args.actual_weight)))

    for target_event in target_events:
        target_t = float(target_event.get("t", 0.0))
        match_idx, match = nearest_unused_actual(
            actual_events,
            used,
            target_t,
            lead=max(0.0, args.lead),
            lag=max(0.0, args.lag),
        )
        out_event = dict(target_event)
        if match_idx is not None and match is not None:
            used.add(match_idx)
            actual_t = float(match.get("t", 0.0))
            candidate_t = actual_weight * actual_t + (1.0 - actual_weight) * target_t
            out_event["matched_actual_t"] = round(actual_t, 6)
            out_event["matched_actual_sound"] = str(match.get("sound", ""))
            out_event["velocity"] = float(target_event.get("velocity", 1.0))
            matches += 1
        else:
            candidate_t = target_t
            out_event["velocity"] = float(target_event.get("velocity", args.fallback_velocity)) * float(args.fallback_velocity)
        if hybrid_events:
            prev_t = float(hybrid_events[-1].get("t", 0.0))
            min_t = prev_t + max(0.0, float(args.min_gap))
            if candidate_t < min_t:
                # Prefer the target beat if the decoded timing collapses two
                # neighboring events into a noisy double-trigger.
                candidate_t = max(target_t, min_t)
                out_event["timing_decompressed"] = True
        out_event["t"] = round(candidate_t, 6)
        hybrid_events.append(out_event)
        primary_times.append(candidate_t)

    noise_velocity = max(0.0, min(1.0, float(args.actual_noise_velocity)))
    noise_added = 0
    last_noise_t = -1e9
    if noise_velocity > 0.0:
        for i, actual_event in enumerate(actual_events):
            if i in used:
                continue
            t = float(actual_event.get("t", 0.0))
            if t < 0.0 or t > forced_audio.sequence_duration_s(target):
                continue
            if t - last_noise_t < max(0.0, float(args.noise_min_gap)):
                continue
            if any(abs(t - pt) < max(0.0, float(args.noise_exclusion)) for pt in primary_times):
                continue
            sound = str(actual_event.get("sound", ""))
            if not sound:
                continue
            noise_event = {
                "t": round(t, 6),
                "sound": sound,
                "duration": float(actual_event.get("duration", 0.32)),
                "velocity": noise_velocity,
                "label": "noise",
                "source": "actual-decoded-noise",
                "visible": False,
            }
            hybrid_events.append(noise_event)
            last_noise_t = t
            noise_added += 1

    out = dict(target)
    out["title"] = f"Hybrid target+decoded: {target.get('title', 'sequence')}"
    out["events"] = sorted(hybrid_events, key=lambda e: float(e.get("t", 0.0)))
    out["hybrid"] = {
        "target": str(resolve_path(args.target)),
        "actual": str(resolve_path(args.actual)),
        "trim_start": args.trim_start,
        "lead": args.lead,
        "lag": args.lag,
        "actual_weight": actual_weight,
        "min_gap": args.min_gap,
        "actual_noise_velocity": noise_velocity,
        "noise_min_gap": args.noise_min_gap,
        "noise_exclusion": args.noise_exclusion,
        "noise_events": noise_added,
        "matches": matches,
        "target_events": len(target_events),
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"matches={matches}/{len(target_events)} noise_events={noise_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
