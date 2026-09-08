"""Generate the compact SAW-TOPSIS rank-comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUT = Path(__file__).with_name("fig_rank_comparison.png")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None)
    return parser.parse_args()


def resolve_run_root(path: Path | None) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    candidates = sorted(
        (PROJECT_ROOT / "output" / "mcdm_results").glob("MCDM*"),
        reverse=True,
    )
    for candidate in candidates:
        metrics = candidate / "5,5" / "MCDMInputMetrics.csv"
        if metrics.exists() and len(pd.read_csv(metrics, usecols=["model"])) == 26:
            return candidate
    raise FileNotFoundError("No complete 26-configuration MCDM run found")


def compact_name(name: str) -> str:
    replacements = {
        "MoiraiVaR - Moirai-MoE (lambda=0.2)": "MoiraiVaR-MoE",
        "MoiraiVaR - Moirai 2 (lambda=0.2)": "MoiraiVaR-2",
        "MoiraiVaR - Moirai (lambda=0.2)": "MoiraiVaR",
        "GARCH-Autoformer (Tier 1)": "GARCH-Autoformer T1",
        "GARCH-Autoformer (Tier 2)": "GARCH-Autoformer T2",
    }
    return replacements.get(str(name), str(name))


def load_panel(run_root: Path, folder: str, title: str):
    path = run_root / folder / "CombinedMCDMRanking.csv"
    frame = pd.read_csv(path).sort_values(
        ["avg_mcdm_rank_rank", "avg_mcdm_rank", "saw_rank", "topsis_rank", "model_id"]
    )
    rows = [
        (compact_name(row.display_name), int(row.saw_rank), int(row.topsis_rank))
        for row in frame.head(5).itertuples(index=False)
    ]
    return title, rows


def draw_panel(ax, title, rows):
    names = [r[0] for r in rows]
    saw = [r[1] for r in rows]
    topsis = [r[2] for r in rows]
    y = list(range(len(rows)))

    for yi, s_rank, t_rank in zip(y, saw, topsis):
        ax.plot([s_rank, t_rank], [yi, yi], color="#90a4ae", lw=1.5, zorder=1)
    ax.scatter(saw, y, s=34, marker="o", color="#1565c0", label="SAW", zorder=2)
    ax.scatter(topsis, y, s=34, marker="s", color="#c62828", label="TOPSIS", zorder=2)

    ax.set_yticks(y, names, fontsize=7.2)
    ax.invert_yaxis()
    max_rank = max(max(saw), max(topsis))
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_xticks(range(1, max_rank + 1))
    ax.grid(axis="x", color="#cfd8dc", lw=0.55, alpha=0.8)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.set_xlabel("Rank (lower is better)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    args = parse_args()
    run_root = resolve_run_root(args.run_root)
    panels = [
        load_panel(run_root, "5,5", "50:50  Accuracy-Risk"),
        load_panel(run_root, "3,7", "30:70  Risk Priority"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    for ax, (title, rows) in zip(axes, panels):
        draw_panel(ax, title, rows)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.20, top=0.82, wspace=0.58)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{OUT} <- {run_root}")


if __name__ == "__main__":
    main()
