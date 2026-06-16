#!/usr/bin/env python3
"""Replay captured frames and benchmark/compare Air-Drum hand backends.

This is intentionally camera-free: it reuses `dataset/frame_*.png` so palm/hand
changes can be measured repeatedly on exactly the same frames.

Examples:
    python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full
    python3 tools/benchmark_dataset.py --backends cpu-baseline,npu-full --palm-redetect-every 5
    python3 tools/benchmark_dataset.py --backends cpu-baseline --csv out.csv --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from hand_tracker import create_tracker  # noqa: E402


TIP_INDICES = (4, 8, 12, 16, 20)
PROFILE_KEYS = (
    "mode",
    "palm_ms",
    "hand_ms",
    "total_ms",
    "async_palm_ms",
    "palm_wait_ms",
    "num_detections",
    "num_hands",
    "async_pending",
    "landmark_correction",
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay dataset frames through one or more hand-tracking backends."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Image directory or a single image file (default: dataset)",
    )
    p.add_argument("--glob", type=str, default="frame_*.png", help="Glob inside --dataset")
    p.add_argument(
        "--backends",
        type=str,
        default="cpu-baseline,npu-full",
        help="Comma-separated: cpu,cpu-baseline,pinto-cpu,npu,npu-full",
    )
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2))
    p.add_argument("--model-complexity", type=int, default=0, choices=(0, 1))
    p.add_argument("--dxnn", type=str, default="", help="Hand landmark .dxnn path")
    p.add_argument("--dxnn-layout", type=str, default="", help="Hand landmark layout JSON")
    p.add_argument("--palm-tflite", type=str, default="", help="Palm TFLite path")
    p.add_argument("--palm-dxnn", type=str, default="", help="Palm .dxnn path (not recommended)")
    p.add_argument("--hand-tflite", type=str, default="", help="Hand landmark TFLite path")
    p.add_argument("--hand-onnx", type=str, default="", help="PINTO hand landmark ONNX path")
    p.add_argument(
        "--landmark-correction",
        type=str,
        default="",
        help="NPU landmark correction JSON (applied to npu-full only).",
    )
    p.add_argument(
        "--palm-redetect-every",
        type=int,
        default=0,
        help="0=run palm every frame; N>0=skip palm for up to N tracked frames",
    )
    p.add_argument(
        "--async-palm",
        action="store_true",
        help="Run palm detection in a background thread while ROI tracking continues.",
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup invokes on first frame")
    p.add_argument("--runs", type=int, default=1, help="Number of full dataset passes")
    p.add_argument("--limit", type=int, default=0, help="Limit number of frames (0=all)")
    p.add_argument(
        "--frame-interval-ms",
        type=float,
        default=0.0,
        help="Optional sleep after each frame to emulate camera pacing (e.g. 16.7 for 60 FPS).",
    )
    p.add_argument(
        "--compare-ref",
        type=str,
        default="",
        help="Reference backend for landmark error (default: cpu-baseline if present, else first)",
    )
    p.add_argument("--no-compare", action="store_true", help="Disable landmark comparison")
    p.add_argument(
        "--debug-dir",
        type=str,
        default="",
        help="Save landmark overlay PNGs for the highest-error frames.",
    )
    p.add_argument(
        "--debug-top-k",
        type=int,
        default=10,
        help="How many high-error overlays to save when --debug-dir is set.",
    )
    p.add_argument(
        "--debug-min-error",
        type=float,
        default=0.0,
        help="Only save overlays whose max landmark error is at least this value.",
    )
    p.add_argument("--csv", type=str, default="", help="Write per-frame timing CSV")
    p.add_argument("--json", type=str, default="", help="Write summary JSON")
    return p.parse_args()


def _split_backends(raw: str) -> list[str]:
    out = [b.strip().lower() for b in raw.split(",") if b.strip()]
    if not out:
        raise SystemExit("--backends must contain at least one backend")
    valid = {
        "cpu",
        "cpu-baseline",
        "cpu_baseline",
        "pinto-cpu",
        "pinto_cpu",
        "npu",
        "npu-full",
        "npu_full",
    }
    bad = [b for b in out if b not in valid]
    if bad:
        raise SystemExit(f"Unknown backend(s): {bad}. Valid: {sorted(valid)}")
    return [
        "cpu-baseline" if b == "cpu_baseline"
        else "pinto-cpu" if b == "pinto_cpu"
        else "npu-full" if b == "npu_full"
        else b
        for b in out
    ]


def _paths_for_dataset(dataset: str, pattern: str, limit: int) -> list[Path]:
    p = Path(dataset)
    if p.is_file():
        paths = [p]
    else:
        paths = sorted(p.glob(pattern))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(
            f"No images found: dataset={dataset!r}, glob={pattern!r}. "
            "Capture frames with tools/capture_dataset.py or pass --dataset/--glob."
        )
    return paths


def _default_path(*parts: str) -> str:
    p = PROJECT_DIR.joinpath(*parts)
    return str(p) if p.is_file() else ""


def _resolve_dxnn_layout(backend: str, args: argparse.Namespace) -> str | None:
    if args.dxnn_layout.strip():
        return args.dxnn_layout.strip()
    if backend == "npu":
        return _default_path("models", "dxnn_layout.mediapipe_hand_lite_dual.json") or None
    if backend == "npu-full":
        return _default_path("models", "dxnn_layout.mediapipe_hand_lite.json") or None
    return None


def _resolve_dxnn(backend: str, args: argparse.Namespace) -> str:
    if args.dxnn.strip():
        return args.dxnn.strip()
    if backend in ("npu", "npu-full"):
        return _default_path("models", "vendor", "hand_landmark_lite.dxnn")
    return ""


def make_tracker(backend: str, args: argparse.Namespace):
    return create_tracker(
        backend,
        max_hands=args.max_hands,
        model_complexity=args.model_complexity,
        dxnn_path=_resolve_dxnn(backend, args),
        dxnn_layout=_resolve_dxnn_layout(backend, args),
        palm_tflite=args.palm_tflite.strip() or None,
        palm_dxnn=args.palm_dxnn.strip() or None,
        hand_tflite=args.hand_tflite.strip() or None,
        hand_onnx=args.hand_onnx.strip() or None,
        palm_redetect_every=args.palm_redetect_every,
        async_palm=args.async_palm,
        landmark_correction=args.landmark_correction.strip() or None,
    )


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _hands_from_result(res: Any) -> list[dict[str, Any]]:
    hands: list[dict[str, Any]] = []
    lms_list = list(res.multi_hand_landmarks or [])
    handed_list = list(res.multi_handedness or [])
    for i, hlm in enumerate(lms_list):
        if i < len(handed_list):
            cls = handed_list[i].classification[0]
            label = str(cls.label)
            score = float(cls.score)
        else:
            label = "Right" if hlm.landmark[0].x < 0.5 else "Left"
            score = 0.0
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark], dtype=np.float32)
        hands.append({"label": label, "score": score, "landmarks": arr})
    return hands


def _first_by_label(hands: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for h in hands:
        by_label.setdefault(str(h["label"]), h)
    return by_label


def _stats(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def run_backend(
    backend: str,
    paths: list[Path],
    frames_rgb: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    tracker = make_tracker(backend, args)
    records: list[dict[str, Any]] = []
    last_run_hands: list[list[dict[str, Any]]] = []
    try:
        if args.warmup > 0:
            for _ in range(args.warmup):
                tracker.process(frames_rgb[0])

        for run_idx in range(args.runs):
            run_hands: list[list[dict[str, Any]]] = []
            for frame_idx, (path, rgb) in enumerate(zip(paths, frames_rgb)):
                t0 = time.perf_counter()
                res = tracker.process(rgb)
                ms = (time.perf_counter() - t0) * 1000.0
                hands = _hands_from_result(res)
                run_hands.append(hands)
                labels = [str(h["label"]) for h in hands]
                wrists = [
                    f'{h["label"]}:{h["landmarks"][0,0]:.4f},{h["landmarks"][0,1]:.4f}'
                    for h in hands
                ]
                rec: dict[str, Any] = {
                    "backend": backend,
                    "run": run_idx,
                    "frame_index": frame_idx,
                    "image": path.name,
                    "ms": ms,
                    "num_hands": len(hands),
                    "labels": "|".join(labels),
                    "wrists": "|".join(wrists),
                }
                profile = dict(getattr(tracker, "last_profile", {}) or {})
                for key in PROFILE_KEYS:
                    if key in profile:
                        rec[key] = profile[key]
                records.append(rec)
                if args.frame_interval_ms > 0:
                    time.sleep(args.frame_interval_ms / 1000.0)
            last_run_hands = run_hands
    finally:
        tracker.close()
    return records, last_run_hands


def compare_landmarks(
    ref_backend: str,
    all_hands: dict[str, list[list[dict[str, Any]]]],
) -> dict[str, Any]:
    ref = all_hands[ref_backend]
    summary: dict[str, Any] = {}
    for backend, frames in all_hands.items():
        if backend == ref_backend:
            continue
        err_by_label: dict[str, dict[str, list[float]]] = {
            "Left": {"mean": [], "tips": [], "max": []},
            "Right": {"mean": [], "tips": [], "max": []},
        }
        for ref_hands, test_hands in zip(ref, frames):
            ref_by = _first_by_label(ref_hands)
            test_by = _first_by_label(test_hands)
            for label in ("Right", "Left"):
                if label not in ref_by or label not in test_by:
                    continue
                ref_lm = ref_by[label]["landmarks"]
                test_lm = test_by[label]["landmarks"]
                if ref_lm.shape[0] < 21 or test_lm.shape[0] < 21:
                    continue
                d = np.linalg.norm(ref_lm[:, :2] - test_lm[:, :2], axis=1)
                err_by_label[label]["mean"].append(float(d.mean()))
                err_by_label[label]["tips"].append(float(d[list(TIP_INDICES)].mean()))
                err_by_label[label]["max"].append(float(d.max()))
        summary[backend] = {
            label: {name: _stats(vals) for name, vals in metrics.items()}
            for label, metrics in err_by_label.items()
        }
    return summary


def collect_error_records(
    ref_backend: str,
    all_hands: dict[str, list[list[dict[str, Any]]]],
    paths: list[Path],
) -> list[dict[str, Any]]:
    """Per-frame landmark error records with landmark arrays for debug rendering."""
    ref = all_hands[ref_backend]
    records: list[dict[str, Any]] = []
    for backend, frames in all_hands.items():
        if backend == ref_backend:
            continue
        for frame_idx, (ref_hands, test_hands) in enumerate(zip(ref, frames)):
            ref_by = _first_by_label(ref_hands)
            test_by = _first_by_label(test_hands)
            for label in ("Right", "Left"):
                if label not in ref_by or label not in test_by:
                    continue
                ref_lm = ref_by[label]["landmarks"]
                test_lm = test_by[label]["landmarks"]
                if ref_lm.shape[0] < 21 or test_lm.shape[0] < 21:
                    continue
                d = np.linalg.norm(ref_lm[:, :2] - test_lm[:, :2], axis=1)
                records.append(
                    {
                        "backend": backend,
                        "label": label,
                        "frame_index": frame_idx,
                        "image": paths[frame_idx].name,
                        "mean_error": float(d.mean()),
                        "tips_error": float(d[list(TIP_INDICES)].mean()),
                        "max_error": float(d.max()),
                        "max_landmark": int(d.argmax()),
                        "ref_landmarks": ref_lm,
                        "test_landmarks": test_lm,
                    }
                )
    return records


def _draw_landmarks(
    image_bgr: np.ndarray,
    lm: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int = 3,
) -> None:
    h, w = image_bgr.shape[:2]
    pts = [(int(float(x) * w), int(float(y) * h)) for x, y in lm[:, :2]]
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image_bgr, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts[:21]):
        cv2.circle(image_bgr, pt, radius, color, -1, cv2.LINE_AA)
        if i in TIP_INDICES:
            cv2.circle(image_bgr, pt, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def save_debug_overlays(
    debug_dir: str,
    error_records: list[dict[str, Any]],
    frames_rgb: list[np.ndarray],
    *,
    top_k: int,
    min_error: float,
) -> list[dict[str, Any]]:
    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        r for r in sorted(error_records, key=lambda x: float(x["max_error"]), reverse=True)
        if float(r["max_error"]) >= min_error
    ][: max(0, top_k)]

    manifest: list[dict[str, Any]] = []
    for rank, rec in enumerate(selected, start=1):
        frame_idx = int(rec["frame_index"])
        canvas = cv2.cvtColor(frames_rgb[frame_idx], cv2.COLOR_RGB2BGR)
        ref_lm = np.asarray(rec["ref_landmarks"], dtype=np.float32)
        test_lm = np.asarray(rec["test_landmarks"], dtype=np.float32)
        _draw_landmarks(canvas, ref_lm, (0, 220, 0), radius=3)     # reference: green
        _draw_landmarks(canvas, test_lm, (0, 0, 255), radius=3)    # test: red

        txt = (
            f"{rec['backend']} {rec['label']} frame={frame_idx} "
            f"mean={rec['mean_error']:.4f} tips={rec['tips_error']:.4f} "
            f"max={rec['max_error']:.4f}@lm{rec['max_landmark']}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            txt,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "green=reference, red=test",
            (8, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_name = (
            f"{rank:03d}_{rec['backend']}_{rec['label']}_"
            f"frame{frame_idx:03d}_max{rec['max_error']:.4f}.png"
        )
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), canvas)
        manifest.append(
            {
                "rank": rank,
                "path": str(out_path),
                "backend": rec["backend"],
                "label": rec["label"],
                "frame_index": frame_idx,
                "image": rec["image"],
                "mean_error": rec["mean_error"],
                "tips_error": rec["tips_error"],
                "max_error": rec["max_error"],
                "max_landmark": rec["max_landmark"],
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def summarize_records(records_by_backend: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for backend, records in records_by_backend.items():
        ms = [float(r["ms"]) for r in records]
        backend_summary: dict[str, Any] = {"latency_ms": _stats(ms)}
        if any("palm_ms" in r for r in records):
            backend_summary["palm_ms"] = _stats([float(r.get("palm_ms", 0.0)) for r in records])
            backend_summary["hand_ms"] = _stats([float(r.get("hand_ms", 0.0)) for r in records])
            backend_summary["async_palm_ms"] = _stats(
                [float(r.get("async_palm_ms", 0.0)) for r in records]
            )
            backend_summary["palm_wait_ms"] = _stats(
                [float(r.get("palm_wait_ms", 0.0)) for r in records]
            )
            backend_summary["profile_total_ms"] = _stats(
                [float(r.get("total_ms", 0.0)) for r in records]
            )
            backend_summary["modes"] = dict(Counter(str(r.get("mode", "")) for r in records))
        backend_summary["hands_per_frame"] = _stats([float(r["num_hands"]) for r in records])
        out[backend] = backend_summary
    return out


def write_csv(path: str, records_by_backend: dict[str, list[dict[str, Any]]]) -> None:
    rows = [r for records in records_by_backend.values() for r in records]
    fields: list[str] = []
    for preferred in (
        "backend",
        "run",
        "frame_index",
        "image",
        "ms",
        "num_hands",
        "labels",
        "wrists",
        *PROFILE_KEYS,
    ):
        if any(preferred in r for r in rows):
            fields.append(preferred)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any], comparisons: dict[str, Any] | None, ref: str | None) -> None:
    print("\n## Latency summary")
    print("backend          n    mean    p50    p95    min    max   fps(mean)")
    for backend, data in summary.items():
        s = data["latency_ms"]
        mean = s["mean"] or 0.0
        fps = 1000.0 / mean if mean > 0 else 0.0
        print(
            f"{backend:<14} {s['n']:>4} "
            f"{mean:>7.2f} {s['p50']:>6.2f} {s['p95']:>6.2f} "
            f"{s['min']:>6.2f} {s['max']:>6.2f} {fps:>9.2f}"
        )
        if "palm_ms" in data:
            palm = data["palm_ms"]["mean"] or 0.0
            hand = data["hand_ms"]["mean"] or 0.0
            async_palm = data.get("async_palm_ms", {}).get("mean") or 0.0
            palm_wait = data.get("palm_wait_ms", {}).get("mean") or 0.0
            modes = data.get("modes", {})
            extra = ""
            if async_palm or palm_wait:
                extra = f", async_palm={async_palm:.2f} ms, wait={palm_wait:.2f} ms"
            print(f"  profile: palm={palm:.2f} ms, hand={hand:.2f} ms{extra}, modes={modes}")

    if comparisons and ref:
        print(f"\n## Landmark error vs {ref} (normalized xy distance)")
        print("backend          hand      n    mean   tips   max(avg)  max")
        for backend, by_label in comparisons.items():
            for label in ("Right", "Left"):
                mean_s = by_label[label]["mean"]
                tips_s = by_label[label]["tips"]
                max_s = by_label[label]["max"]
                if mean_s["n"] == 0:
                    continue
                print(
                    f"{backend:<14} {label:<6} {mean_s['n']:>4} "
                    f"{(mean_s['mean'] or 0.0):>7.4f} "
                    f"{(tips_s['mean'] or 0.0):>6.4f} "
                    f"{(max_s['mean'] or 0.0):>8.4f} "
                    f"{(max_s['max'] or 0.0):>6.4f}"
                )


def main() -> int:
    args = parse_args()
    backends = _split_backends(args.backends)
    paths = _paths_for_dataset(args.dataset, args.glob, args.limit)
    frames_rgb = [_load_rgb(p) for p in paths]

    print(
        f"[benchmark] frames={len(paths)} runs={args.runs} warmup={args.warmup} "
        f"backends={','.join(backends)} palm_redetect_every={args.palm_redetect_every} "
        f"async_palm={args.async_palm} frame_interval_ms={args.frame_interval_ms}",
        flush=True,
    )

    records_by_backend: dict[str, list[dict[str, Any]]] = {}
    hands_by_backend: dict[str, list[list[dict[str, Any]]]] = {}
    for backend in backends:
        print(f"[benchmark] running {backend} ...", flush=True)
        records, hands = run_backend(backend, paths, frames_rgb, args)
        records_by_backend[backend] = records
        hands_by_backend[backend] = hands

    summary = summarize_records(records_by_backend)

    ref_backend: str | None = None
    comparisons: dict[str, Any] | None = None
    if not args.no_compare and len(backends) >= 2:
        ref_backend = args.compare_ref.strip() or ("cpu-baseline" if "cpu-baseline" in backends else backends[0])
        if ref_backend not in hands_by_backend:
            raise SystemExit(f"--compare-ref {ref_backend!r} is not in --backends")
        comparisons = compare_landmarks(ref_backend, hands_by_backend)

    print_summary(summary, comparisons, ref_backend)

    debug_manifest: list[dict[str, Any]] = []
    if args.debug_dir.strip():
        if not ref_backend:
            print("[benchmark] --debug-dir ignored because landmark comparison is disabled.")
        else:
            error_records = collect_error_records(ref_backend, hands_by_backend, paths)
            debug_manifest = save_debug_overlays(
                args.debug_dir.strip(),
                error_records,
                frames_rgb,
                top_k=args.debug_top_k,
                min_error=args.debug_min_error,
            )
            print(
                f"\n[benchmark] wrote {len(debug_manifest)} debug overlay(s): "
                f"{args.debug_dir.strip()}"
            )

    output = {
        "dataset": str(Path(args.dataset)),
        "glob": args.glob,
        "frames": [p.name for p in paths],
        "runs": args.runs,
        "warmup": args.warmup,
        "palm_redetect_every": args.palm_redetect_every,
        "async_palm": args.async_palm,
        "frame_interval_ms": args.frame_interval_ms,
        "landmark_correction": args.landmark_correction.strip() or None,
        "summary": summary,
        "compare_ref": ref_backend,
        "landmark_comparison": comparisons,
        "debug_overlays": debug_manifest,
    }
    if args.csv.strip():
        write_csv(args.csv.strip(), records_by_backend)
        print(f"\n[benchmark] wrote CSV: {args.csv.strip()}")
    if args.json.strip():
        Path(args.json.strip()).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[benchmark] wrote JSON: {args.json.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
