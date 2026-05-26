#!/usr/bin/env python3
"""Generate report-only figures for the AI Air-Drum Pad final report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, wh, text, fc="#edf2ff", ec="#4c6ef5", fontsize=10):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        linewidth=1.6,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1f2937")
    return patch


def _arrow(ax, start, end, color="#475569"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
            shrinkA=6,
            shrinkB=6,
        )
    )


def system_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "AI Air-Drum Pad Runtime Pipeline", ha="center", va="top", fontsize=18, fontweight="bold")

    boxes = [
        ((0.04, 0.58), (0.14, 0.20), "USB camera\nOpenCV/V4L2", "#e0f2fe", "#0284c7"),
        ((0.23, 0.58), (0.16, 0.20), "Hand tracking\nMediaPipe / TFLite\nDX-RT .dxnn", "#ede9fe", "#7c3aed"),
        ((0.44, 0.58), (0.16, 0.20), "Landmarks\n21 keypoints/hand", "#fef3c7", "#d97706"),
        ((0.65, 0.58), (0.16, 0.20), "Strike detector\n$v_y$ + $|d\\theta/dt|$\n+ cooldown", "#dcfce7", "#16a34a"),
        ((0.86, 0.58), (0.10, 0.20), "Audio\npygame\nPCM", "#fee2e2", "#dc2626"),
    ]
    centers = []
    for xy, wh, text, fc, ec in boxes:
        _box(ax, xy, wh, text, fc=fc, ec=ec)
        centers.append((xy[0] + wh[0] / 2, xy[1] + wh[1] / 2))
    for a, b in zip(centers, centers[1:]):
        _arrow(ax, (a[0] + 0.08, a[1]), (b[0] - 0.08, b[1]))

    _box(ax, (0.29, 0.18), (0.18, 0.20), "Piano mode\nhand×finger → note\nL: G4..C4, R: C5..G5", fc="#f8fafc", ec="#64748b", fontsize=9)
    _box(ax, (0.54, 0.18), (0.18, 0.20), "Drum mode\nfingertip inside\nrectangle pad → sound", fc="#f8fafc", ec="#64748b", fontsize=9)
    _arrow(ax, (0.72, 0.58), (0.40, 0.40), color="#64748b")
    _arrow(ax, (0.72, 0.58), (0.63, 0.40), color="#64748b")

    ax.text(0.5, 0.06, "The same vision and strike detector support both musical interfaces; only mapping changes.", ha="center", fontsize=10, color="#475569")
    fig.savefig(OUT / "system_pipeline.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def strike_logic() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "Strike Decision Logic", ha="center", va="top", fontsize=18, fontweight="bold")

    nodes = {
        "lm": ((0.08, 0.72), (0.22, 0.13), "Tracked fingertip\nand joint landmarks"),
        "vel": ((0.39, 0.72), (0.22, 0.13), "Tip downward speed\n$v_y=\\Delta y/\\Delta t$"),
        "ang": ((0.39, 0.48), (0.22, 0.13), "Joint angular speed\n$|d\\theta/dt|$"),
        "gate": ((0.69, 0.59), (0.22, 0.16), "Both thresholds met?\n+ confidence\n+ cooldown"),
        "map": ((0.39, 0.20), (0.22, 0.13), "Mode mapper\nPiano note or drum pad"),
        "out": ((0.69, 0.20), (0.22, 0.13), "Play pre-rendered\nsound buffer"),
    }
    for key, (xy, wh, text) in nodes.items():
        _box(ax, xy, wh, text, fc="#ffffff", ec="#334155")

    _arrow(ax, (0.30, 0.785), (0.39, 0.785))
    _arrow(ax, (0.30, 0.74), (0.39, 0.545))
    _arrow(ax, (0.61, 0.785), (0.69, 0.69))
    _arrow(ax, (0.61, 0.545), (0.69, 0.64))
    _arrow(ax, (0.80, 0.59), (0.52, 0.33))
    _arrow(ax, (0.61, 0.265), (0.69, 0.265))

    ax.text(0.5, 0.08, "A hit is emitted only when downward motion and finger articulation occur together, reducing false positives from arm-only motion.", ha="center", fontsize=10, color="#475569")
    fig.savefig(OUT / "strike_logic.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def latency_chart() -> None:
    labels = ["CPU\nMediaPipe", "CPU-baseline\nTFLite", "NPU-full\nPalm CPU + Hand NPU", "NPU dual-halves\nHand NPU"]
    values = [35, 105, 111, 16]
    colors = ["#60a5fa", "#f59e0b", "#a78bfa", "#34d399"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1f2937", linewidth=1.0)
    ax.set_ylabel("Approx. end-to-end vision latency (ms)")
    ax.set_title("Backend Latency Comparison from Prototype Measurements", fontsize=15, fontweight="bold")
    ax.axhline(16.7, color="#ef4444", linestyle="--", linewidth=1.3, label="60 FPS frame budget (16.7 ms)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value} ms", ha="center", va="bottom", fontsize=10)
    ax.text(0.5, -0.22, "Four runtime configurations are compared to isolate palm detection, hand-landmark inference, and NPU dispatch effects; final audio latency still requires manual high-speed-camera capture.", transform=ax.transAxes, ha="center", fontsize=9, color="#475569")
    fig.savefig(OUT / "backend_latency.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    system_pipeline()
    strike_logic()
    latency_chart()
    print(f"Saved report figures to {OUT}")


if __name__ == "__main__":
    main()
