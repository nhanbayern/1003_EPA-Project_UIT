from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "merged_predictions" / "merged_all_predictions_24_7.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "stats_analysis" / "stationarity"


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not math.isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("Split ratios must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check stationarity of true volatility by dataset, horizon, and temporal split "
            "using ADF and KPSS tests."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-observations", type=int, default=30)
    return parser.parse_args()


def load_volatility(input_csv: Path) -> pd.DataFrame:
    required = ["dataset", "time", "horizon", "true_volatility"]
    df = pd.read_csv(input_csv, usecols=required, parse_dates=["time"], low_memory=False)
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    df["true_volatility"] = pd.to_numeric(df["true_volatility"], errors="coerce")
    df = df.dropna(subset=["dataset", "time", "horizon", "true_volatility"])

    # The merged prediction file repeats true volatility for each model. Keep one observation
    # per dataset-horizon-date before running time-series tests.
    df = (
        df.sort_values(["dataset", "horizon", "time"])
        .drop_duplicates(["dataset", "horizon", "time"], keep="first")
        .reset_index(drop=True)
    )
    return df


def assign_temporal_splits(df: pd.DataFrame, split_config: SplitConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset, group in df.groupby("dataset", sort=True):
        dates = pd.Series(sorted(group["time"].dropna().unique()))
        n_dates = len(dates)
        train_end = int(math.floor(n_dates * split_config.train))
        val_end = int(math.floor(n_dates * (split_config.train + split_config.val)))

        split_by_time = pd.DataFrame(
            {
                "time": dates,
                "split": np.select(
                    [np.arange(n_dates) < train_end, np.arange(n_dates) < val_end],
                    ["train", "val"],
                    default="test",
                ),
            }
        )
        merged = group.merge(split_by_time, on="time", how="left")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def safe_adf(values: np.ndarray) -> tuple[float | None, float | None, str | None]:
    try:
        stat, pvalue, *_ = adfuller(values, autolag="AIC")
        return float(stat), float(pvalue), None
    except Exception as exc:  # pragma: no cover - diagnostic script
        return None, None, str(exc)


def safe_kpss(values: np.ndarray) -> tuple[float | None, float | None, str | None]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = kpss(values, regression="c", nlags="auto")
        return float(stat), float(pvalue), None
    except Exception as exc:  # pragma: no cover - diagnostic script
        return None, None, str(exc)


def classify_stationarity(
    adf_pvalue: float | None,
    kpss_pvalue: float | None,
    alpha: float,
) -> str:
    adf_stationary = adf_pvalue is not None and adf_pvalue < alpha
    kpss_stationary = kpss_pvalue is not None and kpss_pvalue >= alpha
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and not kpss_stationary:
        return "non_stationary"
    return "mixed"


def summarize_series(
    dataset: str,
    horizon: int,
    split: str,
    values: pd.Series,
    alpha: float,
    min_observations: int,
) -> dict[str, object]:
    clean = values.dropna().astype(float).to_numpy()
    row: dict[str, object] = {
        "dataset": dataset,
        "horizon": horizon,
        "split": split,
        "n_obs": int(len(clean)),
        "mean": float(np.mean(clean)) if len(clean) else np.nan,
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else np.nan,
        "min": float(np.min(clean)) if len(clean) else np.nan,
        "median": float(np.median(clean)) if len(clean) else np.nan,
        "max": float(np.max(clean)) if len(clean) else np.nan,
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "kpss_stat": np.nan,
        "kpss_pvalue": np.nan,
        "stationarity_decision": "insufficient_data",
        "adf_error": "",
        "kpss_error": "",
    }
    if len(clean) < min_observations or np.nanstd(clean) == 0:
        return row

    adf_stat, adf_pvalue, adf_error = safe_adf(clean)
    kpss_stat, kpss_pvalue, kpss_error = safe_kpss(clean)
    row.update(
        {
            "adf_stat": adf_stat if adf_stat is not None else np.nan,
            "adf_pvalue": adf_pvalue if adf_pvalue is not None else np.nan,
            "kpss_stat": kpss_stat if kpss_stat is not None else np.nan,
            "kpss_pvalue": kpss_pvalue if kpss_pvalue is not None else np.nan,
            "stationarity_decision": classify_stationarity(adf_pvalue, kpss_pvalue, alpha),
            "adf_error": adf_error or "",
            "kpss_error": kpss_error or "",
        }
    )
    return row


def run_stationarity_checks(
    df: pd.DataFrame,
    alpha: float,
    min_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["dataset", "horizon"], sort=True)
    for (dataset, horizon), group in grouped:
        rows.append(
            summarize_series(
                str(dataset),
                int(horizon),
                "full",
                group["true_volatility"],
                alpha,
                min_observations,
            )
        )
        for split in ["train", "val", "test"]:
            split_group = group.loc[group["split"] == split, "true_volatility"]
            rows.append(
                summarize_series(
                    str(dataset),
                    int(horizon),
                    split,
                    split_group,
                    alpha,
                    min_observations,
                )
            )
    return pd.DataFrame(rows)


def write_summary_md(
    results: pd.DataFrame,
    output_md: Path,
    input_csv: Path,
    alpha: float,
    split_config: SplitConfig,
) -> None:
    decision_counts = (
        results.groupby(["split", "stationarity_decision"]).size().unstack(fill_value=0).reset_index()
    )
    by_dataset = (
        results.groupby(["dataset", "split", "stationarity_decision"]).size().unstack(fill_value=0).reset_index()
    )
    full_rows = results[results["split"] == "full"]
    stationary_rate = (
        full_rows["stationarity_decision"].eq("stationary").mean() * 100 if len(full_rows) else np.nan
    )

    lines = [
        "# Kiểm tra tính dừng của chuỗi volatility",
        "",
        f"**Input:** `{input_csv}`  ",
        f"**Alpha:** `{alpha}`  ",
        f"**Split thời gian:** train `{split_config.train:.0%}`, val `{split_config.val:.0%}`, test `{split_config.test:.0%}`",
        "",
        "## Phương pháp",
        "",
        "Script kiểm tra `true_volatility` sau khi loại các dòng lặp theo `dataset/horizon/time`, vì file merged predictions lặp lại cùng volatility thật cho nhiều mô hình. Do không có cột split gốc trong CSV, các giai đoạn `train`, `val`, `test` được suy ra theo thứ tự thời gian trong từng dataset. Mỗi kiểm định được chạy riêng theo `dataset × horizon × split`, đồng thời có thêm split `full` cho toàn bộ giai đoạn quan sát.",
        "",
        "Hai kiểm định được dùng cùng lúc: ADF có giả thuyết không là chuỗi có unit root, tức không dừng; KPSS có giả thuyết không là chuỗi dừng quanh mức trung bình. Pipeline chỉ kết luận `stationary` khi ADF có `p < alpha` và KPSS có `p >= alpha`. Nếu hai kiểm định mâu thuẫn, kết quả được ghi là `mixed` để tránh kết luận quá mạnh.",
        "",
        "## Tóm tắt toàn bộ",
        "",
        f"Tỷ lệ `dataset × horizon` được kết luận stationary trên full sample: `{stationary_rate:.2f}%`.",
        "",
        decision_counts.to_markdown(index=False),
        "",
        "## Tóm tắt theo dataset",
        "",
        by_dataset.to_markdown(index=False),
        "",
        "## File output",
        "",
        f"- CSV chi tiết: `{output_md.with_name('volatility_stationarity_by_dataset_horizon_split.csv')}`",
        f"- Metadata: `{output_md.with_name('stationarity_run_metadata.json')}`",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    split_config = SplitConfig(args.train_ratio, args.val_ratio, args.test_ratio)
    split_config.validate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"stationarity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_volatility(args.input_csv)
    df = assign_temporal_splits(df, split_config)
    results = run_stationarity_checks(df, args.alpha, args.min_observations)

    output_csv = output_dir / "volatility_stationarity_by_dataset_horizon_split.csv"
    output_md = output_dir / "volatility_stationarity_summary.md"
    metadata_json = output_dir / "stationarity_run_metadata.json"

    results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    write_summary_md(results, output_md, args.input_csv, args.alpha, split_config)
    metadata_json.write_text(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "rows_after_dedup": int(len(df)),
                "datasets": sorted(df["dataset"].unique().tolist()),
                "horizons": sorted(int(x) for x in df["horizon"].dropna().unique()),
                "alpha": args.alpha,
                "split_config": {
                    "train": split_config.train,
                    "val": split_config.val,
                    "test": split_config.test,
                },
                "min_observations": args.min_observations,
                "output_csv": str(output_csv),
                "output_md": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved stationarity results to: {output_dir}")
    print(results.groupby(["split", "stationarity_decision"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
