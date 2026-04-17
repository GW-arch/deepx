#!/usr/bin/env python3
"""Generate instrument-mapping diagrams for each supported instrument preset.

Outputs PNG images into instruments/ showing:
  - Instrument name & description
  - How to run (CLI command)
  - Two-hand finger diagram with per-finger sound/note labels

Usage:
    python3 tools/gen_instrument_diagrams.py          # all presets
    python3 tools/gen_instrument_diagrams.py --only drum_default
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

# ── matplotlib (Agg backend — no display needed) ──────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumkit_audio import kit_keys, PIANO_DEFAULT_SLOTS

OUT_DIR = Path(__file__).resolve().parent.parent / "instruments"

# ── finger geometry (simplified hand outline) ──────────────────────────────

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")

# Relative (x, y) positions of each fingertip on a stylised hand silhouette.
# Origin bottom-centre of palm; y increases upward.  Designed so left hand
# mirrors to right hand by negating x.
_FINGER_TIP_POS = [
    (-1.6, 1.8),   # thumb  — angled outward
    (-0.7, 3.6),   # index
    (-0.1, 3.9),   # middle
    (0.5, 3.5),    # ring
    (1.1, 3.0),    # pinky
]

# Finger "bone" segments: list of (x, y) waypoints from palm to tip.
_FINGER_BONES = [
    [(-0.8, 0.5), (-1.2, 1.1), (-1.6, 1.8)],         # thumb
    [(-0.6, 1.0), (-0.65, 2.3), (-0.7, 3.6)],         # index
    [(-0.1, 1.0), (-0.1, 2.5), (-0.1, 3.9)],          # middle
    [(0.4, 1.0), (0.45, 2.3), (0.5, 3.5)],            # ring
    [(0.85, 0.9), (0.95, 2.0), (1.1, 3.0)],           # pinky
]

# Palm outline (closed polygon)
_PALM_OUTLINE = [
    (-0.8, 0.5), (-0.95, 1.0), (-0.6, 1.0),  # thumb side
    (-0.65, 2.3), (-0.7, 3.6),                # index tip
    (-0.1, 3.9),                                # middle tip
    (0.5, 3.5),                                 # ring tip
    (1.1, 3.0),                                 # pinky tip
    (0.95, 2.0), (0.85, 0.9),                  # pinky base
    (0.9, 0.3), (-0.3, -0.2), (-0.8, 0.5),    # wrist
]

# Colors per finger
_FINGER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

# ── drawing helpers ────────────────────────────────────────────────────────


def _draw_hand(
    ax: plt.Axes,
    slots: list[str],
    *,
    cx: float,
    cy: float,
    scale: float = 1.0,
    mirror: bool = False,
    hand_label: str = "Hand 0",
):
    """Draw a single hand with labeled fingertips."""
    mx = -1.0 if mirror else 1.0

    # palm fill
    palm_xs = [cx + mx * p[0] * scale for p in _PALM_OUTLINE]
    palm_ys = [cy + p[1] * scale for p in _PALM_OUTLINE]
    ax.fill(palm_xs, palm_ys, color="#fde8d0", edgecolor="#d4a574", linewidth=1.5, zorder=1)

    for fi in range(5):
        color = _FINGER_COLORS[fi]
        # bones
        bones = _FINGER_BONES[fi]
        bx = [cx + mx * b[0] * scale for b in bones]
        by = [cy + b[1] * scale for b in bones]
        ax.plot(bx, by, color=color, linewidth=2.5, solid_capstyle="round", zorder=2)

        # fingertip circle
        tx = cx + mx * _FINGER_TIP_POS[fi][0] * scale
        ty = cy + _FINGER_TIP_POS[fi][1] * scale
        ax.plot(tx, ty, "o", color=color, markersize=14, zorder=3)

        # label
        label = slots[fi] if fi < len(slots) else "?"
        # label offset: thumb labels go further out, others go up
        if fi == 0:
            lx = tx + mx * 0.6 * scale
            ly = ty + 0.1 * scale
            ha = "left" if not mirror else "right"
        else:
            lx = tx
            ly = ty + 0.45 * scale
            ha = "center"

        ax.text(
            lx, ly, label,
            fontsize=10, fontweight="bold", ha=ha, va="bottom",
            color=color, zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85, lw=0.8),
        )

        # finger name (small, below tip)
        ax.text(
            tx, ty - 0.35 * scale, FINGER_NAMES[fi],
            fontsize=6.5, ha="center", va="top", color="#666", zorder=4,
        )

    # hand label at wrist
    ax.text(
        cx, cy - 0.7 * scale, hand_label,
        fontsize=11, fontweight="bold", ha="center", va="top", color="#444",
    )


def generate_diagram(
    title: str,
    subtitle: str,
    run_cmd: str,
    slots: list[str],
    out_path: Path,
    *,
    hand0_label: str = "Hand 0 (Left)",
    hand1_label: str = "Hand 1 (Right)",
):
    """Create and save a single instrument-mapping diagram."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(-7, 7)
    ax.set_ylim(-2.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title block
    ax.text(0, 7.2, title, fontsize=20, fontweight="bold", ha="center", va="top", color="#2c3e50")
    ax.text(0, 6.6, subtitle, fontsize=11, ha="center", va="top", color="#7f8c8d", style="italic")

    # Run command box
    cmd_box = FancyBboxPatch(
        (-6.2, 5.5), 12.4, 0.75,
        boxstyle="round,pad=0.12", fc="#ecf0f1", ec="#bdc3c7", lw=1.2,
    )
    ax.add_patch(cmd_box)
    ax.text(-6.0, 5.88, "$ " + run_cmd, fontsize=8.5, fontfamily="monospace",
            va="center", color="#2c3e50")

    # Legend
    for fi in range(5):
        lx = -5.5 + fi * 2.6
        ax.plot(lx, 5.15, "o", color=_FINGER_COLORS[fi], markersize=7, zorder=5)
        ax.text(lx + 0.25, 5.15, FINGER_NAMES[fi], fontsize=8, va="center", color="#555")

    # Draw hands
    h0_slots = slots[:5]
    h1_slots = slots[5:10] if len(slots) >= 10 else slots[:5]
    _draw_hand(ax, h0_slots, cx=-3.3, cy=0.8, scale=1.0, mirror=False, hand_label=hand0_label)
    _draw_hand(ax, h1_slots, cx=3.3, cy=0.8, scale=1.0, mirror=True, hand_label=hand1_label)

    # Divider line
    ax.plot([0, 0], [-1.0, 5.0], "--", color="#ccc", linewidth=1, zorder=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── presets ────────────────────────────────────────────────────────────────

def _default_drum_slots() -> list[str]:
    return ["kick", "snare", "hat", "ride", "tom_l",
            "tom_m", "hat_o", "crash", "clap", "rim"]


def _default_piano_slots() -> list[str]:
    return list(PIANO_DEFAULT_SLOTS)


def _all_drum_keys() -> list[str]:
    return list(kit_keys())


PRESETS: dict[str, dict] = {
    "drum_default": {
        "title": "Drum Kit — Default Mapping",
        "subtitle": "10-slot drum: hand × finger → percussion sound",
        "run_cmd": "python3 main.py --camera 0",
        "slots": _default_drum_slots,
    },
    "piano_default": {
        "title": "Piano — Default C Major (Dynamic)",
        "subtitle": "10-slot piano: C4–E5 pentatonic, distance-adaptive range",
        "run_cmd": "python3 main.py --piano --camera 0",
        "slots": _default_piano_slots,
    },
    "piano_custom": {
        "title": "Piano — Custom JSON Example",
        "subtitle": "Fixed 10-note layout from instruments.piano.example.json",
        "run_cmd": "python3 main.py --piano --instruments instruments.piano.example.json --camera 0",
        "slots": lambda: ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"],
    },
    "drum_all_instruments": {
        "title": "All Available Drum Sounds",
        "subtitle": "16 built-in percussion samples (first 10 mapped by default)",
        "run_cmd": "python3 main.py --instruments custom.json --camera 0",
        "slots": _all_drum_keys,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate instrument mapping diagrams")
    parser.add_argument("--only", type=str, default="", help="Generate only this preset")
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    targets = {args.only: PRESETS[args.only]} if args.only else PRESETS

    print(f"Generating {len(targets)} diagram(s) → {out_dir}/")

    for name, cfg in targets.items():
        slots = cfg["slots"]() if callable(cfg["slots"]) else cfg["slots"]

        # For >10 slots, show first 10 mapped + note extras
        display_slots = slots[:10]
        subtitle = cfg["subtitle"]
        if len(slots) > 10:
            extras = slots[10:]
            subtitle += f"  |  +{len(extras)} extra: {', '.join(extras)}"

        # Pad to 10 if fewer
        while len(display_slots) < 10:
            display_slots.append("—")

        generate_diagram(
            title=cfg["title"],
            subtitle=subtitle,
            run_cmd=cfg["run_cmd"],
            slots=display_slots,
            out_path=out_dir / f"{name}.png",
        )

    print("Done.")


if __name__ == "__main__":
    main()
