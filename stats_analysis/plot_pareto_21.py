"""Plot all eligible MCDM configurations on the accuracy--risk plane.

This reads the saved SAW ranking output; it does not retrain models or
recompute predictions.  Both axes are benefit scores, so a configuration is
Pareto dominated when another point is at least as high on both axes and
strictly higher on one.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "output" / "mcdm_results" / "MCDM20260729001458" / "5,5" / "SAWRanking.csv"
OUTPUT = ROOT / "Springer_Conference_Proceedings_Template_Updated_2022_01_12" / "fig_accuracy_risk_pareto.png"


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame[["accuracy_score", "risk_score"]].to_numpy(dtype=float)
    keep = []
    for i, point in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if (other >= point).all() and (other > point).any():
                dominated = True
                break
        keep.append(not dominated)
    return pd.Series(keep, index=frame.index)


def short_label(value: str, max_length: int = 28) -> str:
    text = str(value).replace("MoiraiVaR - ", "VaR-")
    text = text.replace(" (lambda=0.2)", "-l0.2")
    text = text.replace(" (Tier 1)", "-T1").replace(" (Tier 2)", "-T2").replace(" (Tier 3)", "-T3")
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def main() -> None:
    frame = pd.read_csv(INPUT)
    required = {"display_name", "accuracy_score", "risk_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["accuracy_score", "risk_score"]).copy()
    frame["pareto"] = pareto_mask(frame)
    frame = frame.sort_values(["pareto", "accuracy_score"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    colors = {True: "#c0392b", False: "#4c78a8"}
    ax.scatter(
        frame.loc[~frame["pareto"], "accuracy_score"],
        frame.loc[~frame["pareto"], "risk_score"],
        s=72,
        color=colors[False],
        alpha=0.80,
        edgecolor="white",
        linewidth=0.7,
        label="Dominated (eligible)",
    )
    ax.scatter(
        frame.loc[frame["pareto"], "accuracy_score"],
        frame.loc[frame["pareto"], "risk_score"],
        s=105,
        color=colors[True],
        marker="D",
        alpha=0.95,
        edgecolor="black",
        linewidth=0.7,
        label="Pareto non-dominated",
    )

    for number, (_, row) in enumerate(frame.iterrows(), start=1):
        ax.annotate(
            f"{number}: {short_label(row['display_name'])}",
            (row["accuracy_score"], row["risk_score"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.4,
            alpha=0.90,
        )

    ax.set_title("Accuracy--Risk Pareto Map for All 21 Eligible Configurations")
    ax.set_xlabel("Forecast Accuracy Component Score (higher is better)")
    ax.set_ylabel("Risk Calibration Component Score (higher is better)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")
    print(f"Configurations: {len(frame)}; Pareto non-dominated: {int(frame['pareto'].sum())}")


if __name__ == "__main__":
    main()
