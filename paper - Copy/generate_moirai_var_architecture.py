"""Draw the evaluated head-only MoiraiVaR--Moirai-MoE architecture."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).with_name("fig_moirai_var_architecture.png")


def box(
    ax,
    xy,
    width,
    height,
    title,
    body,
    color,
    *,
    title_size=6.0,
    body_size=5.1,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        facecolor=color,
        edgecolor="#263238",
        linewidth=0.85,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.70,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#102027",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#263238",
        linespacing=1.10,
    )


def arrow(ax, start, end, *, color="#455a64", dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.0,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def main():
    fig, ax = plt.subplots(figsize=(7.25, 2.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.010, 0.55), 0.105, 0.30, "Returns", "60-day\nwindow", "#eceff1")
    box(ax, (0.145, 0.55), 0.115, 0.30, "Patching", "left-pad to 64\n4 patches x 16", "#d6eaf8")
    box(ax, (0.290, 0.55), 0.120, 0.30, "Input map", "scale + project\ncausal mask", "#d6eaf8")
    box(
        ax,
        (0.440, 0.50),
        0.175,
        0.40,
        "Frozen Moirai-MoE",
        "pretrained small backbone\ncausal encoder + sparse\nexpert routing",
        "#e8f8f5",
        title_size=5.8,
        body_size=4.8,
    )
    box(ax, (0.645, 0.55), 0.105, 0.30, "Pooling", "mean over\n4 tokens", "#e8f8f5")
    box(
        ax,
        (0.780, 0.50),
        0.120,
        0.40,
        "Trainable MLP",
        "d -> 256 -> 5\nReLU, dropout 0.2",
        "#fcf3cf",
        body_size=4.9,
    )
    box(
        ax,
        (0.930, 0.50),
        0.060,
        0.40,
        "Output",
        "volatility\nh=1,3,5,\n10,21",
        "#f5eef8",
        title_size=5.5,
        body_size=4.7,
    )

    for start, end in (
        ((0.115, 0.70), (0.145, 0.70)),
        ((0.260, 0.70), (0.290, 0.70)),
        ((0.410, 0.70), (0.440, 0.70)),
        ((0.615, 0.70), (0.645, 0.70)),
        ((0.750, 0.70), (0.780, 0.70)),
        ((0.900, 0.70), (0.930, 0.70)),
    ):
        arrow(ax, start, end)

    box(
        ax,
        (0.205, 0.07),
        0.155,
        0.25,
        "Forecast term",
        "MSE over 5 horizons",
        "#fdebd0",
    )
    box(
        ax,
        (0.405, 0.07),
        0.190,
        0.25,
        "Joint objective",
        "L = MSE + 0.2 x pinball",
        "#fcf3cf",
    )
    box(
        ax,
        (0.645, 0.04),
        0.225,
        0.31,
        "Risk-alignment term",
        "1% Student-t VaR at h=1\npinball loss; train-only\nkurtosis sets nu",
        "#fadbd8",
        body_size=4.8,
    )

    arrow(ax, (0.360, 0.195), (0.405, 0.195))
    arrow(ax, (0.645, 0.195), (0.595, 0.195))
    arrow(ax, (0.960, 0.50), (0.770, 0.35), color="#78909c", dashed=True)
    arrow(ax, (0.930, 0.53), (0.350, 0.32), color="#78909c", dashed=True)

    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
