#!/usr/bin/env python3
"""Generate report-only figures for the AI Air-Drum Pad final report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_FONT_FAMILY = "Times New Roman"
SERIF_FALLBACKS = [
    REPORT_FONT_FAMILY,
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
]
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": SERIF_FALLBACKS,
        "mathtext.fontset": "stix",
    }
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/local/share/fonts/Times New Roman Bold.ttf" if bold else "/usr/local/share/fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def panda_icon() -> None:
    """Create a cute deterministic PANDA title-page icon."""
    scale = 2
    # Keep the title card wide enough that the subtitle stays comfortably
    # inside the white rounded rectangle after DOCX/PDF scaling.
    width, height = 1850, 420
    img = Image.new("RGBA", (width * scale, height * scale), (255, 250, 244, 255))
    draw = ImageDraw.Draw(img)

    def s(value: float) -> int:
        return int(round(value * scale))

    def box(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
        return (s(x1), s(y1), s(x2), s(y2))

    def text_width(text: str, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def draw_underlined_acronym(x: float, y: float) -> None:
        acronym_color = (8, 145, 178, 255)
        body_color = (51, 65, 85, 255)
        initial_font = _font(s(35), bold=True)
        rest_font = _font(s(35), bold=True)
        cursor = s(x)
        baseline_y = s(y)
        underline_y = baseline_y + s(43)
        pieces = [
            ("P", "ose-"),
            ("A", "ware  •  "),
            ("N", "PU-Driven  •  "),
            ("D", "igital "),
            ("A", "udio"),
        ]
        for initial, rest in pieces:
            draw.text((cursor, baseline_y), initial, font=initial_font, fill=acronym_color)
            initial_w = text_width(initial, initial_font)
            draw.line(
                [(cursor, underline_y), (cursor + initial_w, underline_y)],
                fill=acronym_color,
                width=s(3),
            )
            cursor += initial_w
            draw.text((cursor, baseline_y), rest, font=rest_font, fill=body_color)
            cursor += text_width(rest, rest_font)

    # Soft card background.
    draw.rounded_rectangle(
        box(24, 24, width - 24, height - 24),
        radius=s(46),
        fill=(255, 255, 255, 255),
        outline=(226, 232, 240, 255),
        width=s(3),
    )
    for x, y, r, color in [
        (114, 88, 16, (196, 181, 253, 90)),
        (1680, 92, 26, (125, 211, 252, 85)),
        (1610, 315, 18, (252, 165, 165, 80)),
        (94, 330, 22, (134, 239, 172, 80)),
    ]:
        draw.ellipse(box(x - r, y - r, x + r, y + r), fill=color)

    # Panda face.
    cx, cy = 300, 213
    draw.ellipse(box(cx - 132, cy - 150, cx - 44, cy - 62), fill=(30, 41, 59, 255))
    draw.ellipse(box(cx + 44, cy - 150, cx + 132, cy - 62), fill=(30, 41, 59, 255))
    draw.ellipse(box(cx - 120, cy - 132, cx + 120, cy + 112), fill=(255, 255, 255, 255), outline=(30, 41, 59, 255), width=s(5))
    draw.ellipse(box(cx - 95, cy - 58, cx - 28, cy + 36), fill=(30, 41, 59, 255))
    draw.ellipse(box(cx + 28, cy - 58, cx + 95, cy + 36), fill=(30, 41, 59, 255))
    draw.ellipse(box(cx - 61, cy - 24, cx - 31, cy + 8), fill=(255, 255, 255, 255))
    draw.ellipse(box(cx + 31, cy - 24, cx + 61, cy + 8), fill=(255, 255, 255, 255))
    draw.ellipse(box(cx - 50, cy - 14, cx - 37, cy + 0), fill=(15, 23, 42, 255))
    draw.ellipse(box(cx + 37, cy - 14, cx + 50, cy + 0), fill=(15, 23, 42, 255))
    draw.ellipse(box(cx - 46, cy - 12, cx - 42, cy - 8), fill=(255, 255, 255, 255))
    draw.ellipse(box(cx + 42, cy - 12, cx + 46, cy - 8), fill=(255, 255, 255, 255))
    draw.ellipse(box(cx - 22, cy + 12, cx + 22, cy + 38), fill=(30, 41, 59, 255))
    draw.arc(box(cx - 43, cy + 22, cx - 2, cy + 66), start=10, end=82, fill=(30, 41, 59, 255), width=s(5))
    draw.arc(box(cx + 2, cy + 22, cx + 43, cy + 66), start=98, end=170, fill=(30, 41, 59, 255), width=s(5))
    draw.ellipse(box(cx - 104, cy + 28, cx - 67, cy + 58), fill=(253, 186, 216, 210))
    draw.ellipse(box(cx + 67, cy + 28, cx + 104, cy + 58), fill=(253, 186, 216, 210))

    # Tiny drumsticks and music notes to connect the icon to the instrument.
    draw.line([box(157, 318, 235, 268)[0:2], box(157, 318, 235, 268)[2:4]], fill=(180, 83, 9, 255), width=s(9))
    draw.line([box(443, 318, 365, 268)[0:2], box(443, 318, 365, 268)[2:4]], fill=(180, 83, 9, 255), width=s(9))
    draw.ellipse(box(145, 309, 169, 333), fill=(251, 191, 36, 255), outline=(180, 83, 9, 255), width=s(2))
    draw.ellipse(box(431, 309, 455, 333), fill=(251, 191, 36, 255), outline=(180, 83, 9, 255), width=s(2))
    note_font = _font(s(54), bold=True)
    draw.text((s(103), s(126)), "♪", font=note_font, fill=(99, 102, 241, 255))
    draw.text((s(444), s(112)), "♫", font=note_font, fill=(14, 165, 233, 255))

    # PANDA logotype and expansion.
    title_font = _font(s(92), bold=True)
    small_font = _font(s(27), bold=False)
    draw.text((s(525), s(92)), "PANDA", font=title_font, fill=(15, 23, 42, 255))
    draw_underlined_acronym(531, 193)
    draw.text((s(532), s(248)), "Contactless drum and piano performance with hand-landmark strike detection", font=small_font, fill=(71, 85, 105, 255))

    # Small friendly NPU chip badge.
    chip = box(1565, 64, 1728, 178)
    draw.rounded_rectangle(chip, radius=s(22), fill=(236, 254, 255, 255), outline=(8, 145, 178, 255), width=s(4))
    draw.text((s(1604), s(100)), "NPU", font=_font(s(32), bold=True), fill=(8, 145, 178, 255))
    for y in [87, 113, 139, 165]:
        draw.line([s(1545), s(y), s(1565), s(y)], fill=(8, 145, 178, 255), width=s(4))
        draw.line([s(1728), s(y), s(1748), s(y)], fill=(8, 145, 178, 255), width=s(4))

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    img = img.resize((width, height), resample)
    img.save(OUT / "panda_icon.png")


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
    labels = [
        "CPU\nMediaPipe",
        "CPU-baseline\nTFLite",
        "PINTO CPU\nPalm CPU + ONNX",
        "NPU-full\nPalm CPU + Hand NPU",
        "PINTO NPU\nPalm CPU + DXNN",
        "Palm NPU\ninvalid tracker",
        "NPU dual-halves\nHand NPU",
    ]
    values = [35.0, 84.4, 88.85, 50.32, 50.48, 10.88, 7.16]
    colors = ["#60a5fa", "#f59e0b", "#fb923c", "#a78bfa", "#8b5cf6", "#94a3b8", "#34d399"]

    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1f2937", linewidth=1.0)
    ax.set_ylabel("Approx. end-to-end vision latency (ms)")
    ax.set_title("Backend Latency Comparison from Prototype Measurements", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(values) + 18)
    ax.axhline(16.7, color="#ef4444", linestyle="--", linewidth=1.3, label="60 FPS frame budget (16.7 ms)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.5,
            f"{value:g} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.5,
        -0.25,
        "PINTO CPU/NPU rows are included; the Palm NPU row is fast but invalid because no palms are accepted. The dual-halves row is fast but geometrically approximate.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(OUT / "backend_latency.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def backend_tradeoff_chart() -> None:
    labels = [
        "CPU-baseline\nTFLite",
        "PINTO CPU\nPalm CPU + ONNX",
        "NPU-full\nPalm CPU + Hand NPU",
        "PINTO NPU\nPalm CPU + DXNN",
        "NPU dual-halves\nHand NPU",
        "Palm NPU\ninvalid tracker",
    ]
    latency_ms = [87.73, 88.85, 49.76, 50.48, 7.16, 10.88]
    mean_xy_error = [
        0.0,
        (0.0176 + 0.0103) / 2.0,
        (0.0068 + 0.0129) / 2.0,
        (0.0205 + 0.0115) / 2.0,
        (0.1593 + 0.1319) / 2.0,
        0.0,
    ]
    npu_active_share = [
        0.0,
        0.0,
        8.57 / 49.76 * 100.0,
        9.00 / 50.48 * 100.0,
        100.0,
        10.82 / 10.88 * 100.0,
    ]
    colors = ["#f59e0b", "#fb923c", "#a78bfa", "#8b5cf6", "#34d399", "#94a3b8"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    fig.suptitle("Backend Trade-off: Speed, Accuracy, and NPU-Active Share", fontsize=16, fontweight="bold", y=0.98)

    panels = [
        (axes[0], latency_ms, "Mean vision latency (ms)", "Lower is better", "{:.1f}", set()),
        (axes[1], mean_xy_error, "Mean normalized XY error", "Lower is better", "{:.4f}", {5}),
        (axes[2], npu_active_share, "NPU-active share proxy (%)", "Higher means more NPU-bound", "{:.0f}%", set()),
    ]

    for ax, values, ylabel, subtitle, fmt, invalid_indices in panels:
        bars = ax.bar(labels, values, color=colors, edgecolor="#1f2937", linewidth=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(axis="y", color="#e5e7eb")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        for index, (bar, value) in enumerate(zip(bars, values)):
            if index in invalid_indices:
                bar.set_alpha(0.25)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    ax.get_ylim()[1] * 0.055,
                    "invalid",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#475569",
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ax.get_ylim()[1] * 0.025,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].axhline(16.7, color="#ef4444", linestyle="--", linewidth=1.1)
    axes[0].text(0.0, 17.9, "60 FPS budget", fontsize=8, color="#ef4444")
    axes[1].set_ylim(0, 0.17)
    axes[2].set_ylim(0, 112)
    fig.text(
        0.5,
        0.01,
        "Accuracy uses 10-frame replay agreement against cpu-baseline. NPU-active share is derived from profiled NPU-backed stage time, not a raw dxtop instantaneous percent.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(bottom=0.31, top=0.84, wspace=0.28)
    fig.savefig(OUT / "backend_tradeoff.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    panda_icon()
    system_pipeline()
    strike_logic()
    latency_chart()
    backend_tradeoff_chart()
    print(f"Saved report figures to {OUT}")


if __name__ == "__main__":
    main()
