from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stats_analysis.risk import VarBacktester


VAR_CASES = {
    "var_5pct": 0.05,
    "var_1pct": 0.01,
}
VAR_METHODS = ("normal", "student_t", "fhs")
FORECAST_METRICS = ("mse", "mae", "qlike")
VAR_METRICS = ("abs_violation_error", "quantile_loss", "kupiec_lr", "lr_ind")
GROUP_COLUMNS = ["branch", "tier", "model", "dataset", "horizon"]
MODEL_COLUMNS = ["branch", "tier", "model"]


def model_id_frame(df: pd.DataFrame) -> pd.Series:
    tier = df["tier"].fillna("").astype(str)
    return df["branch"].astype(str) + "|" + tier + "|" + df["model"].astype(str)


def quantile_loss(returns: np.ndarray, var_threshold: np.ndarray, alpha: float) -> np.ndarray:
    indicator = (returns < var_threshold).astype(float)
    return (alpha - indicator) * (returns - var_threshold)


def aggregate_var_method_stats(var_predictions: pd.DataFrame, alpha: float) -> pd.DataFrame:
    records: list[dict] = []
    backtester = VarBacktester(alpha=alpha)

    for group_key, group in var_predictions.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        base_record = dict(zip(GROUP_COLUMNS, group_key))
        returns = pd.to_numeric(group["log_return"], errors="coerce").to_numpy(dtype=float)

        for method in VAR_METHODS:
            var_col = f"{method}_var"
            if var_col not in group.columns:
                continue

            var_values = pd.to_numeric(group[var_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(returns) & np.isfinite(var_values)
            if not np.any(valid):
                records.append(
                    {
                        **base_record,
                        "var_method": method,
                        "n_risk": 0,
                        "violation_count": 0,
                        "violation_rate": np.nan,
                        "abs_violation_error": np.nan,
                        "quantile_loss": np.nan,
                        "kupiec_lr": np.nan,
                        "kupiec_p": np.nan,
                        "lr_ind": np.nan,
                        "lr_ind_p": np.nan,
                    }
                )
                continue

            violations = returns[valid] < var_values[valid]
            violation_rate, kupiec_lr, kupiec_p = backtester.kupiec_test(violations)
            lr_ind, lr_ind_p = backtester.christoffersen_independence_test(violations)
            qloss = quantile_loss(returns[valid], var_values[valid], alpha)

            records.append(
                {
                    **base_record,
                    "var_method": method,
                    "n_risk": int(valid.sum()),
                    "violation_count": int(violations.sum()),
                    "violation_rate": float(violation_rate),
                    "abs_violation_error": float(abs(violation_rate - alpha)),
                    "quantile_loss": float(np.mean(qloss)),
                    "kupiec_lr": float(kupiec_lr),
                    "kupiec_p": float(kupiec_p),
                    "lr_ind": float(lr_ind),
                    "lr_ind_p": float(lr_ind_p),
                }
            )

    out = pd.DataFrame.from_records(records)
    out["model_id"] = model_id_frame(out)
    out["block_id"] = out["dataset"].astype(str) + "|h" + out["horizon"].astype(str)
    return out


def build_metric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    matrix = df.pivot_table(index="block_id", columns="model_id", values=metric, aggfunc="mean")
    return matrix.dropna(axis=0, how="any").sort_index(axis=0).sort_index(axis=1)


def friedman_and_nemenyi(df: pd.DataFrame, metric: str, context: dict[str, object]) -> tuple[dict, pd.DataFrame]:
    matrix = build_metric_matrix(df, metric)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        summary = {
            **context,
            "metric": metric,
            "blocks": int(matrix.shape[0]),
            "models": int(matrix.shape[1]),
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
        }
        return summary, pd.DataFrame()

    values = [matrix[col].to_numpy(dtype=float) for col in matrix.columns]
    friedman_stat, friedman_p = stats.friedmanchisquare(*values)
    ranks = matrix.rank(axis=1, method="average", ascending=True)
    avg_ranks = ranks.mean(axis=0)
    n_blocks = matrix.shape[0]
    n_models = matrix.shape[1]
    se = np.sqrt(n_models * (n_models + 1) / (6.0 * n_blocks))

    nemenyi_records: list[dict] = []
    for model_a, model_b in combinations(matrix.columns, 2):
        rank_diff = abs(float(avg_ranks[model_a] - avg_ranks[model_b]))
        q_stat = rank_diff / se if se > 0 else np.nan
        p_value = stats.studentized_range.sf(q_stat * np.sqrt(2.0), n_models, np.inf)
        nemenyi_records.append(
            {
                **context,
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "avg_rank_a": float(avg_ranks[model_a]),
                "avg_rank_b": float(avg_ranks[model_b]),
                "rank_diff": rank_diff,
                "nemenyi_q": float(q_stat),
                "nemenyi_p": float(p_value),
            }
        )

    summary = {
        **context,
        "metric": metric,
        "blocks": int(n_blocks),
        "models": int(n_models),
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
    }
    return summary, pd.DataFrame.from_records(nemenyi_records)


def newey_west_variance(values: np.ndarray, lag: int) -> float:
    centered = values - np.mean(values)
    n = centered.size
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * gamma
    return max(lrv, 0.0)


def diebold_mariano(differential: np.ndarray, lag: int) -> tuple[float, float]:
    d = differential[np.isfinite(differential)]
    n = d.size
    if n < 3:
        return np.nan, np.nan

    lrv = newey_west_variance(d, lag=lag)
    if lrv <= 0:
        return np.nan, np.nan

    dm_stat = float(np.mean(d) / np.sqrt(lrv / n))
    p_value = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    return dm_stat, p_value


def run_dm_for_var_predictions(var_predictions: pd.DataFrame, alpha: float, var_case: str) -> pd.DataFrame:
    df = var_predictions.copy()
    df["model_id"] = model_id_frame(df)
    records: list[dict] = []

    for method in VAR_METHODS:
        var_col = f"{method}_var"
        if var_col not in df.columns:
            continue

        work = df.loc[:, ["dataset", "horizon", "time", "model_id", "log_return", var_col]].copy()
        work["log_return"] = pd.to_numeric(work["log_return"], errors="coerce")
        work[var_col] = pd.to_numeric(work[var_col], errors="coerce")
        work = work.dropna(subset=["log_return", var_col])
        if work.empty:
            continue

        work["loss"] = quantile_loss(
            work["log_return"].to_numpy(dtype=float),
            work[var_col].to_numpy(dtype=float),
            alpha,
        )

        for (dataset, horizon), group in work.groupby(["dataset", "horizon"], dropna=False, sort=True):
            pivot = group.pivot_table(index="time", columns="model_id", values="loss", aggfunc="mean")
            pivot = pivot.dropna(axis=1, how="all").sort_index(axis=1)
            lag = max(int(horizon) - 1, 0)

            for model_a, model_b in combinations(pivot.columns, 2):
                pair = pivot[[model_a, model_b]].dropna()
                if pair.shape[0] < 3:
                    continue

                differential = pair[model_a].to_numpy(dtype=float) - pair[model_b].to_numpy(dtype=float)
                dm_stat, dm_p = diebold_mariano(differential, lag=lag)
                records.append(
                    {
                        "var_case": var_case,
                        "alpha": alpha,
                        "var_method": method,
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "model_a": model_a,
                        "model_b": model_b,
                        "n": int(pair.shape[0]),
                        "mean_loss_a": float(pair[model_a].mean()),
                        "mean_loss_b": float(pair[model_b].mean()),
                        "mean_loss_diff_a_minus_b": float(np.mean(differential)),
                        "dm_stat": dm_stat,
                        "dm_p": dm_p,
                    }
                )

    return pd.DataFrame.from_records(records)


def run_forecast_tests(case_dir: Path, var_case: str, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = pd.read_csv(case_dir / "stats_by_dataset_horizon.csv")
    detailed["model_id"] = model_id_frame(detailed)
    detailed["block_id"] = detailed["dataset"].astype(str) + "|h" + detailed["horizon"].astype(str)

    friedman_records: list[dict] = []
    nemenyi_frames: list[pd.DataFrame] = []
    for metric in FORECAST_METRICS:
        summary, nemenyi = friedman_and_nemenyi(
            detailed,
            metric,
            {"test_family": "forecast", "var_case": var_case, "alpha": alpha},
        )
        friedman_records.append(summary)
        if not nemenyi.empty:
            nemenyi_frames.append(nemenyi)

    return pd.DataFrame.from_records(friedman_records), pd.concat(nemenyi_frames, ignore_index=True)


def run_var_tests(var_predictions: pd.DataFrame, var_case: str, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    var_stats = aggregate_var_method_stats(var_predictions, alpha)

    friedman_records: list[dict] = []
    nemenyi_frames: list[pd.DataFrame] = []
    for method in VAR_METHODS:
        method_stats = var_stats[var_stats["var_method"] == method]
        for metric in VAR_METRICS:
            summary, nemenyi = friedman_and_nemenyi(
                method_stats,
                metric,
                {
                    "test_family": "var",
                    "var_case": var_case,
                    "alpha": alpha,
                    "var_method": method,
                },
            )
            friedman_records.append(summary)
            if not nemenyi.empty:
                nemenyi_frames.append(nemenyi)

    return pd.DataFrame.from_records(friedman_records), pd.concat(nemenyi_frames, ignore_index=True)


def main() -> None:
    output_dir = PROJECT_ROOT / "output" / "stats_analysis" / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    friedman_frames: list[pd.DataFrame] = []
    nemenyi_frames: list[pd.DataFrame] = []
    dm_frames: list[pd.DataFrame] = []

    for var_case, alpha in VAR_CASES.items():
        case_dir = PROJECT_ROOT / "output" / "stats_analysis" / var_case
        print(f"Running statistical tests for {var_case}...")

        forecast_friedman, forecast_nemenyi = run_forecast_tests(case_dir, var_case, alpha)
        friedman_frames.append(forecast_friedman)
        nemenyi_frames.append(forecast_nemenyi)

        var_predictions = pd.read_csv(case_dir / "var_predictions.csv", low_memory=False)
        var_friedman, var_nemenyi = run_var_tests(var_predictions, var_case, alpha)
        dm_results = run_dm_for_var_predictions(var_predictions, alpha, var_case)

        friedman_frames.append(var_friedman)
        nemenyi_frames.append(var_nemenyi)
        dm_frames.append(dm_results)

    friedman = pd.concat(friedman_frames, ignore_index=True)
    nemenyi = pd.concat(nemenyi_frames, ignore_index=True)
    dm = pd.concat(dm_frames, ignore_index=True)

    friedman.to_csv(output_dir / "friedman_results.csv", index=False)
    nemenyi.to_csv(output_dir / "nemenyi_pairwise_results.csv", index=False)
    dm.to_csv(output_dir / "dm_var_pairwise_results.csv", index=False)

    print(f"Saved: {output_dir / 'friedman_results.csv'}")
    print(f"Saved: {output_dir / 'nemenyi_pairwise_results.csv'}")
    print(f"Saved: {output_dir / 'dm_var_pairwise_results.csv'}")


if __name__ == "__main__":
    main()
