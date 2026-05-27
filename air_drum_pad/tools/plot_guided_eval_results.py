#!/usr/bin/env python3
"""Plot guided evaluation summaries for the final report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


def _load_run(path: Path) -> dict[str, Any]:
    obj = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    summary = obj["summary"]
    metadata = obj.get("metadata", {})
    return {
        "path": str(path),
        "mode": metadata.get("mode", path.name.split("_")[-1]),
        "backend": metadata.get("backend", "unknown"),
        "piano_hand": metadata.get("piano_hand"),
        "bpm": metadata.get("bpm"),
        "protocol": metadata.get("protocol", "sequence"),
        "scheduled_cue_count": metadata.get("scheduled_cue_count", summary.get("cue_count", 0)),
        "cue_count": summary.get("cue_count", 0),
        "event_count": summary.get("event_count", 0),
        "tp": summary.get("tp", 0),
        "fp": summary.get("fp", 0),
        "fn": summary.get("fn", 0),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "mean_latency_ms": summary.get("mean_latency_ms"),
        "p95_abs_latency_ms": summary.get("p95_abs_latency_ms"),
        "global_event_cooldown_s": metadata.get("global_event_cooldown_s"),
        "mirror_view": metadata.get("mirror_view"),
    }


def _pct(value: float | None) -> float:
    return 0.0 if value is None else float(value) * 100.0


def _num(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _label(run: dict[str, Any]) -> str:
    mode = str(run["mode"]).capitalize()
    if str(run["mode"]) == "piano":
        hand = run.get("piano_hand")
        if hand in {"left", "right"}:
            return f"Piano\n1 hand ({hand})"
        return "Piano\n2 hands"
    return mode


def plot(runs: list[dict[str, Any]], out_path: Path) -> None:
    labels = [_label(r) for r in runs]
    precision = [_pct(r["precision"]) for r in runs]
    recall = [_pct(r["recall"]) for r in runs]
    mean_latency = [_num(r["mean_latency_ms"]) for r in runs]
    p95_latency = [_num(r["p95_abs_latency_ms"]) for r in runs]
    tp = [r["tp"] for r in runs]
    fp = [r["fp"] for r in runs]
    fn = [r["fn"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    protocols = {str(r.get("protocol", "sequence")) for r in runs}
    protocol_label = "Block-Repetition Trials" if protocols == {"blocks"} else "Guided Trials"
    fig.suptitle(f"Guided Evaluation Results (Mirrored, {protocol_label})", fontsize=15, fontweight="bold")

    x = range(len(labels))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], precision, width=width, label="Precision", color="#60a5fa", edgecolor="#1f2937")
    axes[0].bar([i + width / 2 for i in x], recall, width=width, label="Recall / target accuracy", color="#34d399", edgecolor="#1f2937")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent (%)")
    axes[0].set_title("Cue Matching Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", color="#e5e7eb")
    for i, (p, r) in enumerate(zip(precision, recall)):
        axes[0].text(i - width / 2, p + 2, f"{p:.1f}%", ha="center", fontsize=8)
        axes[0].text(i + width / 2, r + 2, f"{r:.1f}%", ha="center", fontsize=8)

    axes[1].bar(labels, mean_latency, label="Mean signed", color="#f59e0b", edgecolor="#1f2937")
    axes[1].scatter(labels, p95_latency, label="P95 abs.", color="#dc2626", zorder=3)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Cue-to-Detection Latency")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", color="#e5e7eb")
    for i, v in enumerate(mean_latency):
        axes[1].text(i, v + max(mean_latency + [1]) * 0.04, f"{v:.0f}", ha="center", fontsize=8)

    bottom_fp = tp
    bottom_fn = [a + b for a, b in zip(tp, fp)]
    axes[2].bar(labels, tp, label="TP", color="#22c55e", edgecolor="#1f2937")
    axes[2].bar(labels, fp, bottom=bottom_fp, label="FP", color="#f97316", edgecolor="#1f2937")
    axes[2].bar(labels, fn, bottom=bottom_fn, label="FN", color="#ef4444", edgecolor="#1f2937")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Match Counts")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", color="#e5e7eb")
    max_count = max((r["tp"] + r["fp"] + r["fn"] for r in runs), default=1)
    axes[2].set_ylim(0, max_count + 22)
    for i, r in enumerate(runs):
        total = r["tp"] + r["fp"] + r["fn"]
        axes[2].text(i, total + 0.5, f"TP {r['tp']}\nFP {r['fp']}\nFN {r['fn']}", ha="center", va="bottom", fontsize=8)

    fig.text(
        0.5,
        0.01,
        "Latency is cue-to-detection and includes human reaction time; it is not an acoustic motion-to-speaker measurement.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot guided evaluation summary JSON files.")
    p.add_argument("runs", nargs="+", help="Run directories containing summary.json")
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "guided_eval_results.png"))
    args = p.parse_args()
    runs = [_load_run(Path(x)) for x in args.runs]
    plot(runs, Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
