"""Generate the compact experimental-evaluation workflow figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).with_name("fig_evaluation_pipeline.png")


def add_box(ax, x, width, title, body, color):
    box = FancyBboxPatch(
        (x, 0.22),
        width,
        0.58,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        facecolor=color,
        edgecolor="#263238",
        linewidth=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        0.64,
        title,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#102027",
    )
    ax.text(
        x + width / 2,
        0.43,
        body,
        ha="center",
        va="center",
        fontsize=6.0,
        color="#263238",
        linespacing=1.18,
    )


def main():
    fig, ax = plt.subplots(figsize=(7.25, 1.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.015, 0.155, "Predictions", "26 configurations\n9 markets, 5 horizons", "#d6eaf8"),
        (0.215, 0.155, "Nine criteria", "3 losses + 2 dynamics\n4 VaR means (N, t, FHS)", "#e8f8f5"),
        (0.415, 0.155, "Sanity Gate", "$e_\\sigma\\leq0.9$\n$\\rho>0$", "#fcf3cf"),
        (0.615, 0.155, "Decision policies", "SAW + TOPSIS\n50:50 and 30:70", "#f5eef8"),
        (0.815, 0.155, "Robustness", "rank agreement\nweight/Gate sensitivity", "#fadbd8"),
    ]

    for x, width, title, body, color in boxes:
        add_box(ax, x, width, title, body, color)

    for x1, x2 in zip([0.180, 0.380, 0.580, 0.780], [0.205, 0.405, 0.605, 0.805]):
        ax.annotate(
            "",
            xy=(x2, 0.51),
            xytext=(x1, 0.51),
            arrowprops=dict(arrowstyle="-|>", color="#455a64", lw=1.2),
        )

    ax.text(
        0.5,
        0.08,
        "Metrics are computed by market-horizon block before configuration-level comparison.",
        ha="center",
        va="center",
        fontsize=7,
        color="#455a64",
    )
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
