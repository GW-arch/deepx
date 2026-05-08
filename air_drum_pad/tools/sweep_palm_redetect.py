#!/usr/bin/env python3
"""Sweep palm re-detection intervals and collect latency/accuracy summaries.

This is a thin orchestration wrapper around tools/benchmark_dataset.py.  It is
useful for quickly plotting the trade-off between:

  * accuracy/stability (`--palm-redetect-every 0`, palm every frame)
  * latency (`--palm-redetect-every N`, ROI tracking between palm passes)
  * experimental async palm (`--async-palm`)

Example:
    python3 tools/sweep_palm_redetect.py --values 0,1,2,3,5,10 \
      --backends cpu-baseline,npu-full --csv /tmp/palm_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep --palm-redetect-every values.")
    p.add_argument("--values", type=str, default="0,1,2,3,5,10")
    p.add_argument("--dataset", type=str, default="dataset")
    p.add_argument("--glob", type=str, default="frame_*.png")
    p.add_argument("--backends", type=str, default="cpu-baseline,npu-full")
    p.add_argument("--max-hands", type=int, default=2, choices=(1, 2))
    p.add_argument("--model-complexity", type=int, default=0, choices=(0, 1))
    p.add_argument("--dxnn", type=str, default="")
    p.add_argument("--dxnn-layout", type=str, default="")
    p.add_argument("--palm-tflite", type=str, default="")
    p.add_argument("--palm-dxnn", type=str, default="")
    p.add_argument("--hand-tflite", type=str, default="")
    p.add_argument("--landmark-correction", type=str, default="")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--frame-interval-ms", type=float, default=0.0)
    p.add_argument("--compare-ref", type=str, default="")
    p.add_argument("--async-palm", action="store_true")
    p.add_argument("--csv", type=str, default="", help="Write compact sweep CSV")
    p.add_argument("--json", type=str, default="", help="Write full sweep JSON")
    return p.parse_args()


def parse_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 0:
            raise SystemExit("--values must be non-negative")
        values.append(v)
    if not values:
        raise SystemExit("--values must contain at least one integer")
    return values


def stat_get(data: dict[str, Any], *path: str) -> float | int | str | None:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def build_benchmark_cmd(args: argparse.Namespace, value: int, json_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "tools/benchmark_dataset.py",
        "--dataset", args.dataset,
        "--glob", args.glob,
        "--backends", args.backends,
        "--max-hands", str(args.max_hands),
        "--model-complexity", str(args.model_complexity),
        "--palm-redetect-every", str(value),
        "--warmup", str(args.warmup),
        "--runs", str(args.runs),
        "--json", str(json_path),
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.frame_interval_ms > 0:
        cmd += ["--frame-interval-ms", str(args.frame_interval_ms)]
    if args.compare_ref.strip():
        cmd += ["--compare-ref", args.compare_ref.strip()]
    if args.async_palm:
        cmd += ["--async-palm"]
    for opt in (
        "dxnn",
        "dxnn_layout",
        "palm_tflite",
        "palm_dxnn",
        "hand_tflite",
        "landmark_correction",
    ):
        val = getattr(args, opt)
        if val.strip():
            cmd += ["--" + opt.replace("_", "-"), val.strip()]
    return cmd


def compact_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    value = result["palm_redetect_every"]
    async_palm = result.get("async_palm", False)
    compare_ref = result.get("compare_ref")
    comparison = result.get("landmark_comparison") or {}
    for backend, data in (result.get("summary") or {}).items():
        row: dict[str, Any] = {
            "palm_redetect_every": value,
            "async_palm": async_palm,
            "backend": backend,
            "latency_mean_ms": stat_get(data, "latency_ms", "mean"),
            "latency_p95_ms": stat_get(data, "latency_ms", "p95"),
            "latency_min_ms": stat_get(data, "latency_ms", "min"),
            "latency_max_ms": stat_get(data, "latency_ms", "max"),
            "palm_mean_ms": stat_get(data, "palm_ms", "mean"),
            "hand_mean_ms": stat_get(data, "hand_ms", "mean"),
            "async_palm_mean_ms": stat_get(data, "async_palm_ms", "mean"),
            "palm_wait_mean_ms": stat_get(data, "palm_wait_ms", "mean"),
            "modes": json.dumps(data.get("modes", {}), ensure_ascii=False),
            "compare_ref": compare_ref,
        }
        if backend in comparison:
            for label in ("Right", "Left"):
                label_stats = comparison[backend].get(label, {})
                row[f"{label.lower()}_err_mean"] = stat_get(label_stats, "mean", "mean")
                row[f"{label.lower()}_err_tips"] = stat_get(label_stats, "tips", "mean")
                row[f"{label.lower()}_err_max"] = stat_get(label_stats, "max", "max")
                row[f"{label.lower()}_err_n"] = stat_get(label_stats, "mean", "n")
        rows.append(row)
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    print("\n## Palm re-detect sweep")
    print("N   async  backend          mean_ms  p95_ms  right_err  left_err  modes")
    for r in rows:
        print(
            f"{int(r['palm_redetect_every']):<3} "
            f"{str(bool(r['async_palm'])):<6} "
            f"{r['backend']:<14} "
            f"{float(r.get('latency_mean_ms') or 0.0):>7.2f} "
            f"{float(r.get('latency_p95_ms') or 0.0):>7.2f} "
            f"{float(r.get('right_err_mean') or 0.0):>9.4f} "
            f"{float(r.get('left_err_mean') or 0.0):>8.4f} "
            f"{r.get('modes', '{}')}"
        )


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    values = parse_values(args.values)
    all_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="air_drum_sweep_") as td:
        tmp = Path(td)
        for value in values:
            out_json = tmp / f"bench_redetect_{value}.json"
            cmd = build_benchmark_cmd(args, value, out_json)
            print(f"[sweep] running N={value}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
            result = json.loads(out_json.read_text(encoding="utf-8"))
            all_results.append(result)
            rows.extend(compact_rows(result))

    print_rows(rows)
    if args.csv.strip():
        write_csv(args.csv.strip(), rows)
        print(f"\n[sweep] wrote CSV: {args.csv.strip()}")
    if args.json.strip():
        Path(args.json.strip()).write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[sweep] wrote JSON: {args.json.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
