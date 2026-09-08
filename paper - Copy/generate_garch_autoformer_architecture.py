"""Draw the proposed GARCH-Autoformer residual-correction architecture."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).with_name("fig_garch_autoformer_architecture.png")


def box(ax, xy, width, height, title, body, color, title_size=5.9, body_size=5.1):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color,
        edgecolor="#263238",
        linewidth=0.85,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#102027",
        linespacing=1.0,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#263238",
        linespacing=1.12,
    )


def arrow(ax, start, end, color="#455a64", style="-|>", dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": 1.0,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def main():
    fig, ax = plt.subplots(figsize=(7.25, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.012, 0.36), 0.12, 0.32, "Returns", "$r_{t-L+1:t}$", "#eceff1")

    box(
        ax,
        (0.185, 0.63),
        0.15,
        0.27,
        "GJR-GARCH",
        "train-only fit\nconditional variance",
        "#d6eaf8",
    )
    box(
        ax,
        (0.39, 0.63),
        0.15,
        0.27,
        "Variance recursion",
        "$\\sigma^2_{G,t+1:t+H}$\nstructured path",
        "#d6eaf8",
    )

    box(
        ax,
        (0.185, 0.12),
        0.15,
        0.27,
        "Innovations",
        "$z_t=r_t/\\sigma_{G,t}$\nstandardized residuals",
        "#e8f8f5",
    )
    box(
        ax,
        (0.39, 0.12),
        0.15,
        0.27,
        "Autoformer encoder",
        "scale + embed + position\n2/3 decomposition layers",
        "#e8f8f5",
    )
    box(
        ax,
        (0.59, 0.12),
        0.12,
        0.27,
        "Residual head",
        "$\\Delta_{1:H}$\nvariance correction",
        "#e8f8f5",
    )

    box(
        ax,
        (0.59, 0.61),
        0.12,
        0.31,
        "Positive fusion",
        "$\\hat\\sigma_h^2=\\mathrm{softplus}$"
        "\n$(\\sigma^2_{G,h}+\\Delta_h)$",
        "#fcf3cf",
        body_size=5.0,
    )
    box(
        ax,
        (0.77, 0.54),
        0.115,
        0.38,
        "Volatility path",
        "$\\hat\\sigma_{t+1:t+H}$\n$H\\in\\{1,3,5,10,21\\}$",
        "#f5eef8",
    )
    box(
        ax,
        (0.925, 0.54),
        0.065,
        0.38,
        "Use",
        "forecast\n+\nVaR",
        "#fadbd8",
        title_size=5.8,
        body_size=5.0,
    )

    arrow(ax, (0.132, 0.58), (0.185, 0.76))
    arrow(ax, (0.132, 0.43), (0.185, 0.25))
    arrow(ax, (0.335, 0.765), (0.39, 0.765))
    arrow(ax, (0.335, 0.255), (0.39, 0.255))
    arrow(ax, (0.54, 0.255), (0.59, 0.255))
    arrow(ax, (0.54, 0.765), (0.59, 0.765))
    arrow(ax, (0.65, 0.39), (0.65, 0.61))
    arrow(ax, (0.71, 0.765), (0.77, 0.765))
    arrow(ax, (0.885, 0.735), (0.925, 0.735))

    ax.text(
        0.555,
        0.02,
        "Training: Student-t negative log-likelihood on returns; "
        "selection: forecast/dynamics and Normal, Student-t, FHS VaR.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#455a64",
    )
    arrow(ax, (0.825, 0.54), (0.70, 0.395), color="#78909c", dashed=True)

    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
