#!/usr/bin/env python3
"""Single-strike dataset capture: preview → SPACE → 1s delay → 1s burst → auto stop.

Usage:
    python tools/capture_dataset.py [--camera 0] [--width 640] [--height 480] [--outdir dataset]
    python tools/capture_dataset.py --delay 1.0 --duration 1.0

Controls:
    SPACE  — start capture (delay → burst → auto quit)
    Q/ESC  — quit without capturing
"""

import argparse
import os
import sys
import time
import cv2
import glob
import json
from datetime import datetime, timezone


def next_frame_index(outdir: str) -> int:
    """Find the next available frame_NNN index."""
    existing = glob.glob(os.path.join(outdir, "frame_*.png"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        base = os.path.basename(p)
        try:
            idx = int(base.replace("frame_", "").replace(".png", ""))
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def main():
    ap = argparse.ArgumentParser(description="Single-strike dataset capture")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--outdir", type=str, default="dataset")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to wait after SPACE")
    ap.add_argument("--duration", type=float, default=1.0, help="seconds to capture")
    ap.add_argument("--label", type=str, default="", help="e.g. 'left_index_down'")
    ap.add_argument("--session", type=str, default="", help="optional session/set name")
    ap.add_argument("--notes", type=str, default="", help="free-form capture notes")
    ap.add_argument(
        "--manifest",
        type=str,
        default="capture_manifest.json",
        help="manifest JSON path relative to outdir (default: capture_manifest.json)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    idx = next_frame_index(args.outdir)
    start_idx = idx

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("ERROR: cannot open camera", args.camera, file=sys.stderr)
        sys.exit(1)

    win = "Dataset Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    print(f"[capture] Camera {args.camera} opened ({args.width}x{args.height})")
    print(f"[capture] Output dir: {args.outdir}/  (next index: {idx})")
    print(f"[capture] Press SPACE to start ({args.delay}s delay → {args.duration}s capture)")

    # --- Phase 1: Preview, wait for SPACE ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame", file=sys.stderr)
            cap.release(); cv2.destroyAllWindows(); sys.exit(1)

        disp = frame.copy()
        cv2.putText(disp, "Press SPACE to start", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            cap.release(); cv2.destroyAllWindows()
            print("[capture] Cancelled."); return
        if key == ord(' '):
            break

    # --- Phase 2: Countdown delay ---
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.delay:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        remaining = args.delay - (time.monotonic() - t0)
        cv2.putText(disp, f"GET READY  {remaining:.1f}s", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, disp)
        cv2.waitKey(1)

    # --- Phase 3: Capture burst ---
    print(f"[capture] CAPTURING...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.duration:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(args.outdir, f"frame_{idx:03d}.png")
        cv2.imwrite(path, frame)
        idx += 1

        disp = frame.copy()
        elapsed = time.monotonic() - t0
        cv2.putText(disp, f"RECORDING  {elapsed:.1f}/{args.duration:.1f}s  [{idx - start_idx} frames]",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, disp)
        cv2.waitKey(1)

    count = idx - start_idx
    cap.release()
    cv2.destroyAllWindows()
    print(f"[capture] Done. Saved {count} frames: frame_{start_idx:03d}.png → frame_{idx-1:03d}.png")

    manifest_path = (
        args.manifest
        if os.path.isabs(args.manifest)
        else os.path.join(args.outdir, args.manifest)
    )
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session": args.session,
        "label": args.label,
        "notes": args.notes,
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "delay_s": args.delay,
        "duration_s": args.duration,
        "start_index": start_idx,
        "end_index": idx - 1,
        "num_frames": count,
        "files": [f"frame_{i:03d}.png" for i in range(start_idx, idx)],
    }
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {"captures": []}
        if not isinstance(manifest, dict):
            manifest = {"captures": []}
        manifest.setdefault("captures", []).append(record)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[capture] Manifest updated: {manifest_path}")
    except Exception as e:
        print(f"[capture] WARNING: failed to update manifest: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
