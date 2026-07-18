from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VAR_CASES = ("var_5pct", "var_1pct")
VAR_METHODS = ("normal", "student_t", "fhs")
FORECAST_METRICS = ("mse", "mae", "qlike")
SIGNIFICANCE_LEVEL = 0.05
MODEL_COLUMNS = ["branch", "tier", "model"]
GROUP_COLUMNS = ["branch", "tier", "model", "dataset", "horizon"]


def model_id_frame(df: pd.DataFrame) -> pd.Series:
    tier = df["tier"].fillna("").astype(str)
    return df["branch"].astype(str) + "|" + tier + "|" + df["model"].astype(str)


def quantile_loss(returns: np.ndarray, var_threshold: np.ndarray, alpha: float) -> np.ndarray:
    indicator = (returns < var_threshold).astype(float)
    return (alpha - indicator) * (returns - var_threshold)


def add_rank_columns(df: pd.DataFrame, lower_is_better: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for metric in lower_is_better:
        if metric in out.columns:
            out[f"{metric}_rank"] = out[metric].rank(method="min", ascending=True).astype("Int64")
    return out


def add_average_forecast_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rank_columns = [f"{metric}_rank" for metric in FORECAST_METRICS]
    missing_columns = [column for column in rank_columns if column not in out.columns]
    if missing_columns:
        out["avg_forecast_rank"] = pd.NA
        out["avg_forecast_rank_rank"] = pd.NA
        return out

    out["avg_forecast_rank"] = out[rank_columns].mean(axis=1, skipna=False)
    out["avg_forecast_rank_rank"] = (
        out["avg_forecast_rank"].rank(method="min", ascending=True).astype("Int64")
    )
    return out


def build_forecast_ranking(stats_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for var_case in VAR_CASES:
        path = stats_root / var_case / "stats_by_model.csv"
        df = pd.read_csv(path)
        df.insert(0, "var_case", var_case)
        df["model_id"] = model_id_frame(df)
        ranked = add_rank_columns(df, FORECAST_METRICS)
        frames.append(add_average_forecast_rank(ranked))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["var_case", "avg_forecast_rank", "qlike_rank", "model_id"], na_position="last")
        .reset_index(drop=True)
    )


def build_var_ranking(stats_root: Path, var_case: str) -> pd.DataFrame:
    df = pd.read_csv(stats_root / var_case / "stats_by_model.csv")
    df.insert(0, "var_case", var_case)
    df["model_id"] = model_id_frame(df)
    lower_metrics = ("violation_rate", "kupiec_lr", "lr_ind", "qlike")
    ranked = add_rank_columns(df, lower_metrics)
    ranked["pass_rate_rank"] = ranked["pass_rate"].rank(method="min", ascending=False).astype("Int64")
    return ranked.sort_values(["pass_rate_rank", "qlike_rank"], na_position="last").reset_index(drop=True)


def build_accuracy_risk_tradeoff(stats_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for var_case in VAR_CASES:
        df = pd.read_csv(stats_root / var_case / "stats_by_model.csv")
        df.insert(0, "var_case", var_case)
        df["model_id"] = model_id_frame(df)
        df["qlike_rank"] = df["qlike"].rank(method="min", ascending=True).astype(int)
        df["mse_rank"] = df["mse"].rank(method="min", ascending=True).astype(int)
        df["mae_rank"] = df["mae"].rank(method="min", ascending=True).astype(int)
        df["pass_rate_rank"] = df["pass_rate"].rank(method="min", ascending=False).astype(int)
        df["risk_accuracy_gap"] = df["pass_rate_rank"] - df["qlike_rank"]
        rows.extend(df.to_dict("records"))
    cols = [
        "var_case",
        "branch",
        "tier",
        "model",
        "model_id",
        "mse",
        "mae",
        "qlike",
        "pass_rate",
        "violation_rate",
        "qlike_rank",
        "mse_rank",
        "mae_rank",
        "pass_rate_rank",
        "risk_accuracy_gap",
    ]
    out = pd.DataFrame(rows)
    return out.loc[:, cols].sort_values(["var_case", "risk_accuracy_gap", "qlike_rank"]).reset_index(drop=True)


def summarize_var_predictions(path: Path, var_case: str, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    records: list[dict] = []

    for group_key, group in df.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        base = dict(zip(GROUP_COLUMNS, group_key))
        returns = pd.to_numeric(group["log_return"], errors="coerce").to_numpy(dtype=float)

        for method in VAR_METHODS:
            var_col = f"{method}_var"
            if var_col not in group.columns:
                continue

            var_values = pd.to_numeric(group[var_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(returns) & np.isfinite(var_values)
            if not np.any(valid):
                continue

            violations = returns[valid] < var_values[valid]
            qloss = quantile_loss(returns[valid], var_values[valid], alpha)
            records.append(
                {
                    **base,
                    "var_case": var_case,
                    "alpha": alpha,
                    "var_method": method,
                    "n": int(valid.sum()),
                    "violation_rate": float(violations.mean()),
                    "abs_violation_error": float(abs(violations.mean() - alpha)),
                    "quantile_loss": float(np.mean(qloss)),
                }
            )

    detailed = pd.DataFrame.from_records(records)
    detailed["model_id"] = model_id_frame(detailed)
    aggregate = (
        detailed.groupby(["var_case", "alpha", "var_method", *MODEL_COLUMNS, "model_id"], dropna=False)
        .agg(
            cases=("model_id", "size"),
            total_n=("n", "sum"),
            violation_rate=("violation_rate", "mean"),
            abs_violation_error=("abs_violation_error", "mean"),
            quantile_loss=("quantile_loss", "mean"),
        )
        .reset_index()
    )
    aggregate["abs_violation_error_rank"] = aggregate.groupby(["var_case", "var_method"])["abs_violation_error"].rank(
        method="min", ascending=True
    ).astype("Int64")
    aggregate["quantile_loss_rank"] = aggregate.groupby(["var_case", "var_method"])["quantile_loss"].rank(
        method="min", ascending=True
    ).astype("Int64")
    return aggregate.sort_values(["var_case", "var_method", "quantile_loss_rank"]).reset_index(drop=True)


def build_var_method_comparison(stats_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for var_case in VAR_CASES:
        alpha = 0.05 if var_case == "var_5pct" else 0.01
        frames.append(summarize_var_predictions(stats_root / var_case / "var_predictions.csv", var_case, alpha))
    return pd.concat(frames, ignore_index=True)


def build_friedman_summary(stats_root: Path) -> pd.DataFrame:
    df = pd.read_csv(stats_root / "statistical_tests" / "friedman_results.csv")
    df["significant_0.05"] = df["friedman_p"] < SIGNIFICANCE_LEVEL
    return df.sort_values(["test_family", "var_case", "var_method", "metric"], na_position="first").reset_index(drop=True)


def build_nemenyi_significant_pairs(stats_root: Path) -> pd.DataFrame:
    df = pd.read_csv(stats_root / "statistical_tests" / "nemenyi_pairwise_results.csv")
    out = df[df["nemenyi_p"] < SIGNIFICANCE_LEVEL].copy()
    return out.sort_values(["test_family", "var_case", "var_method", "metric", "nemenyi_p"]).reset_index(drop=True)


def build_dm_significant_pairs_summary(stats_root: Path) -> pd.DataFrame:
    df = pd.read_csv(stats_root / "statistical_tests" / "dm_var_pairwise_results.csv")
    sig = df[df["dm_p"] < SIGNIFICANCE_LEVEL].copy()
    if sig.empty:
        return pd.DataFrame(
            columns=[
                "var_case",
                "var_method",
                "dataset",
                "horizon",
                "significant_pairs",
                "model_a_wins",
                "model_b_wins",
                "avg_abs_dm_stat",
                "min_dm_p",
            ]
        )

    sig["model_a_wins"] = sig["mean_loss_diff_a_minus_b"] < 0
    sig["model_b_wins"] = sig["mean_loss_diff_a_minus_b"] > 0
    summary = (
        sig.groupby(["var_case", "var_method", "dataset", "horizon"], dropna=False)
        .agg(
            significant_pairs=("dm_p", "size"),
            model_a_wins=("model_a_wins", "sum"),
            model_b_wins=("model_b_wins", "sum"),
            avg_abs_dm_stat=("dm_stat", lambda s: float(np.mean(np.abs(s)))),
            min_dm_p=("dm_p", "min"),
        )
        .reset_index()
    )
    return summary.sort_values(["var_case", "var_method", "significant_pairs"], ascending=[True, True, False])


def main() -> None:
    stats_root = PROJECT_ROOT / "output" / "stats_analysis"
    output_dir = stats_root / "research_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "forecast_ranking.csv": build_forecast_ranking(stats_root),
        "var_5pct_ranking.csv": build_var_ranking(stats_root, "var_5pct"),
        "var_1pct_ranking.csv": build_var_ranking(stats_root, "var_1pct"),
        "var_method_comparison.csv": build_var_method_comparison(stats_root),
        "accuracy_risk_tradeoff.csv": build_accuracy_risk_tradeoff(stats_root),
        "friedman_summary.csv": build_friedman_summary(stats_root),
        "nemenyi_significant_pairs.csv": build_nemenyi_significant_pairs(stats_root),
        "dm_significant_pairs_summary.csv": build_dm_significant_pairs_summary(stats_root),
    }

    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
