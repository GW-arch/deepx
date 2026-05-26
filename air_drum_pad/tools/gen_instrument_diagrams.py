#!/usr/bin/env python3
"""Generate instrument-mapping diagrams for each supported instrument preset.

Outputs PNG images into instruments/ showing:
  - Instrument name & description
  - How to run (CLI command)
  - Piano: two-hand finger diagram with per-finger note labels
  - Drums: on-screen rectangular pad layout with per-pad sound labels

Usage:
    python3 tools/gen_instrument_diagrams.py          # all presets
    python3 tools/gen_instrument_diagrams.py --only drum_default
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── matplotlib (Agg backend — no display needed) ──────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drumkit_audio import kit_keys, PIANO_DEFAULT_SLOTS
from strike_detector import PadZone, default_pad_zones

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
    # In camera-facing diagrams a left hand has its thumb on the viewer's
    # right, while a right hand has its thumb on the viewer's left.
    _draw_hand(ax, h0_slots, cx=-3.3, cy=0.8, scale=1.0, mirror=True, hand_label=hand0_label)
    _draw_hand(ax, h1_slots, cx=3.3, cy=0.8, scale=1.0, mirror=False, hand_label=hand1_label)

    # Divider line
    ax.plot([0, 0], [-1.0, 5.0], "--", color="#ccc", linewidth=1, zorder=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def _bgr_to_rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


def _draw_pad_layout(
    ax: plt.Axes,
    pads: list[PadZone],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    """Draw normalized camera-space pad zones with y-axis inverted for display."""
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        fc="#1f2933",
        ec="#4b5563",
        lw=1.5,
    )
    ax.add_patch(frame)
    ax.text(
        x + 0.03 * w,
        y + h - 0.04 * h,
        "camera view",
        fontsize=8.5,
        ha="left",
        va="top",
        color="#cbd5e1",
    )
    for pad in pads:
        px = x + pad.x1 * w
        # Pad coordinates are image coordinates (y increases downward);
        # matplotlib display coordinates increase upward.
        py = y + (1.0 - pad.y2) * h
        pw = (pad.x2 - pad.x1) * w
        ph = (pad.y2 - pad.y1) * h
        rgb = _bgr_to_rgb01(pad.color)
        rect = FancyBboxPatch(
            (px, py),
            pw,
            ph,
            boxstyle="round,pad=0.01",
            fc=rgb,
            ec="#ffffff",
            lw=1.2,
            alpha=0.82,
        )
        ax.add_patch(rect)
        ax.text(
            px + pw / 2,
            py + ph / 2,
            f"{pad.label}\n{pad.sound_key}",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc="#000000", ec="none", alpha=0.35),
        )


def generate_pad_diagram(
    title: str,
    subtitle: str,
    run_cmd: str,
    pads: list[PadZone],
    out_path: Path,
    *,
    footer: str = "",
) -> None:
    """Create and save a rectangular drum-pad layout diagram."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6, 7.25, title, fontsize=20, fontweight="bold", ha="center", va="top", color="#2c3e50")
    ax.text(6, 6.68, subtitle, fontsize=11, ha="center", va="top", color="#7f8c8d", style="italic")

    cmd_box = FancyBboxPatch(
        (0.55, 5.75), 10.9, 0.65,
        boxstyle="round,pad=0.12", fc="#ecf0f1", ec="#bdc3c7", lw=1.2,
    )
    ax.add_patch(cmd_box)
    ax.text(0.75, 6.075, "$ " + run_cmd, fontsize=8.5, fontfamily="monospace",
            va="center", color="#2c3e50")

    _draw_pad_layout(ax, pads, x=1.0, y=0.9, w=10.0, h=4.45)

    ax.text(
        6,
        0.45,
        footer or "Any tracked fingertip can hit any rectangle; drum mode is no longer a fixed per-finger map.",
        fontsize=10,
        ha="center",
        va="center",
        color="#475569",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── presets ────────────────────────────────────────────────────────────────

def _default_piano_slots() -> list[str]:
    return list(PIANO_DEFAULT_SLOTS)


def _all_drum_keys() -> list[str]:
    return list(kit_keys())


def _all_drum_pads() -> list[PadZone]:
    sounds = _all_drum_keys()
    colors = [
        (180, 80, 80), (80, 180, 80), (80, 80, 200), (180, 180, 60),
        (60, 180, 180), (180, 60, 180), (60, 120, 200), (180, 120, 60),
        (120, 120, 220), (120, 180, 80), (220, 120, 120), (120, 220, 180),
        (180, 120, 220), (220, 180, 120), (80, 160, 220), (160, 160, 160),
    ]
    pads: list[PadZone] = []
    cols, rows = 4, 4
    x_margin, y_top, y_bot = 0.05, 0.12, 0.92
    pad_w = (1.0 - 2 * x_margin) / cols
    pad_h = (y_bot - y_top) / rows
    for i, sound in enumerate(sounds):
        col, row = i % cols, i // cols
        x1 = x_margin + col * pad_w
        y1 = y_top + row * pad_h
        pads.append(
            PadZone(
                label=sound,
                sound_key=sound,
                x1=x1,
                y1=y1,
                x2=x1 + pad_w - 0.01,
                y2=y1 + pad_h - 0.01,
                color=colors[i % len(colors)],
            )
        )
    return pads


PRESETS: dict[str, dict] = {
    "drum_default": {
        "kind": "pads",
        "title": "Drum Pads — Default Layout",
        "subtitle": "8 on-screen rectangles: move any fingertip into a pad and strike downward",
        "run_cmd": "python3 main.py --camera 0",
        "pads": default_pad_zones,
    },
    "piano_default": {
        "kind": "hand",
        "title": "Piano — Default C Major",
        "subtitle": "10-slot piano: left thumb highest→pinky lowest, right thumb→pinky rising",
        "run_cmd": "python3 main.py --piano --camera 0",
        "slots": _default_piano_slots,
    },
    "piano_custom": {
        "kind": "hand",
        "title": "Piano — Custom JSON Example",
        "subtitle": "Fixed 10-note layout from instruments.piano.example.json",
        "run_cmd": "python3 main.py --piano --instruments instruments.piano.example.json --camera 0",
        "slots": _default_piano_slots,
    },
    "drum_all_instruments": {
        "kind": "pads",
        "title": "All Available Drum Sounds",
        "subtitle": "16 built-in percussion samples usable as pad sound keys",
        "run_cmd": "python3 main.py --drum-pads pads.example.json --camera 0",
        "pads": _all_drum_pads,
        "footer": "Use these sound keys in pads.example.json or another --drum-pads layout.",
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
        kind = cfg.get("kind", "hand")
        if kind == "pads":
            pads = cfg["pads"]() if callable(cfg["pads"]) else cfg["pads"]
            generate_pad_diagram(
                title=cfg["title"],
                subtitle=cfg["subtitle"],
                run_cmd=cfg["run_cmd"],
                pads=pads,
                out_path=out_dir / f"{name}.png",
                footer=cfg.get("footer", ""),
            )
            continue

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
