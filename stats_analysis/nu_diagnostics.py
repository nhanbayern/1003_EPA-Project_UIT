from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis


FIXED_SPLITS: dict[str, tuple[int, int, int]] = {
    "VN30_INDEX": (1971, 1096, 925),
    "VN_INDEX": (1964, 1103, 925),
    "DAX_40": (1983, 1104, 972),
    "EURONEXT_100": (2005, 1115, 979),
    "IBEX_35": (1815, 1362, 923),
    "KOSPI_INDEX": (1944, 1109, 881),
    "SMI": (1987, 1135, 901),
    "SNP500": (1988, 1109, 927),
    "NIKKEI_225": (1677, 1174, 1062),
}


@dataclass(frozen=True)
class NuDiagnosticsConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    dataset_dir: Path | None = None
    output_csv: Path | None = None
    min_nu: float = 2.1
    max_nu: float = 30.0
    return_scale: float = 100.0

    def resolved_dataset_dir(self) -> Path:
        if self.dataset_dir is not None:
            return Path(self.dataset_dir)
        return self.project_root / "dataset"

    def resolved_output_csv(self) -> Path:
        if self.output_csv is not None:
            return Path(self.output_csv)
        return self.project_root / "output" / "stats_analysis" / "student_t_nu_by_dataset.csv"


def normalize_dataset_key(name: str) -> str:
    return "".join(ch for ch in str(name).upper().replace(".CSV", "") if ch.isalnum())


def calculate_kurtosis_nu(
    returns,
    min_nu: float = 2.1,
    max_nu: float = 30.0,
) -> tuple[float, float, int]:
    arr = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if arr.size < 5:
        return float("nan"), float("nan"), int(arr.size)

    excess_kurtosis = float(kurtosis(arr, fisher=True, nan_policy="omit"))
    if not np.isfinite(excess_kurtosis):
        return float("nan"), excess_kurtosis, int(arr.size)
    if excess_kurtosis <= 0:
        return float(max_nu), excess_kurtosis, int(arr.size)

    nu = 4.0 + 6.0 / excess_kurtosis
    return float(np.clip(nu, min_nu, max_nu)), excess_kurtosis, int(arr.size)


def fit_student_t_nu(
    returns,
    min_nu: float = 2.1,
    max_nu: float = 30.0,
) -> tuple[float, float, float, int]:
    arr = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if arr.size < 5:
        return float("nan"), float("nan"), float("nan"), int(arr.size)

    try:
        nu, loc, scale = stats.t.fit(arr)
    except Exception:
        return float("nan"), float("nan"), float("nan"), int(arr.size)

    if not np.isfinite(nu):
        return float("nan"), float(loc), float(scale), int(arr.size)
    return float(np.clip(nu, min_nu, max_nu)), float(loc), float(scale), int(arr.size)


def load_close_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    columns = {col.strip().lower(): col for col in df.columns}
    if "close" not in columns:
        raise ValueError(f"Missing close column in {csv_path}")

    close = pd.to_numeric(df[columns["close"]], errors="coerce")
    if "time" in columns:
        index = pd.to_datetime(df[columns["time"]], errors="coerce")
    elif "date" in columns:
        index = pd.to_datetime(df[columns["date"]], errors="coerce")
    else:
        index = pd.RangeIndex(len(close))

    series = pd.Series(close.to_numpy(), index=index).dropna().astype(float)
    if series.empty:
        raise ValueError(f"No valid close prices in {csv_path}")
    return series.sort_index()


def compute_log_returns(close: pd.Series, scale: float = 100.0) -> pd.Series:
    returns = np.log(close / close.shift(1)) * float(scale)
    return pd.Series(returns, index=close.index)


def discover_dataset_files(dataset_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for csv_path in Path(dataset_dir).glob("*.csv"):
        files[normalize_dataset_key(csv_path.stem)] = csv_path
    return files


def split_returns(
    returns: pd.Series,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, pd.Series]:
    train = returns.iloc[:train_count]
    validation = returns.iloc[train_count : train_count + validation_count]
    test = returns.iloc[train_count + validation_count : train_count + validation_count + test_count]
    train_validation = returns.iloc[: train_count + validation_count]
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "train_validation": train_validation,
    }


def build_nu_diagnostics(config: NuDiagnosticsConfig | None = None) -> pd.DataFrame:
    cfg = config or NuDiagnosticsConfig()
    dataset_files = discover_dataset_files(cfg.resolved_dataset_dir())
    rows: list[dict[str, object]] = []

    for dataset, (train_count, validation_count, test_count) in FIXED_SPLITS.items():
        csv_path = dataset_files.get(normalize_dataset_key(dataset))
        if csv_path is None:
            rows.append(
                {
                    "dataset": dataset,
                    "file": "",
                    "status": "missing_csv",
                    "train_count": train_count,
                    "validation_count": validation_count,
                    "test_count": test_count,
                }
            )
            continue

        close = load_close_series(csv_path)
        returns = compute_log_returns(close, scale=cfg.return_scale)
        segments = split_returns(returns, train_count, validation_count, test_count)

        row: dict[str, object] = {
            "dataset": dataset,
            "file": csv_path.name,
            "status": "ok",
            "train_count": train_count,
            "validation_count": validation_count,
            "test_count": test_count,
            "raw_rows": int(close.size),
            "return_scale": float(cfg.return_scale),
        }

        for segment_name, segment_returns in segments.items():
            k_nu, excess_k, n_obs = calculate_kurtosis_nu(
                segment_returns,
                min_nu=cfg.min_nu,
                max_nu=cfg.max_nu,
            )
            fit_nu, fit_loc, fit_scale, fit_n = fit_student_t_nu(
                segment_returns,
                min_nu=cfg.min_nu,
                max_nu=cfg.max_nu,
            )
            row[f"n_{segment_name}"] = n_obs
            row[f"excess_kurtosis_{segment_name}"] = excess_k
            row[f"kurtosis_nu_{segment_name}"] = k_nu
            row[f"fit_nu_{segment_name}"] = fit_nu
            row[f"fit_loc_{segment_name}"] = fit_loc
            row[f"fit_scale_{segment_name}"] = fit_scale
            row[f"fit_n_{segment_name}"] = fit_n

        rows.append(row)

    return pd.DataFrame(rows)


def save_nu_diagnostics(config: NuDiagnosticsConfig | None = None) -> Path:
    cfg = config or NuDiagnosticsConfig()
    diagnostics = build_nu_diagnostics(cfg)
    output_csv = cfg.resolved_output_csv()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    output_csv = save_nu_diagnostics()
    print(f"Saved Student-t nu diagnostics to {output_csv}")


if __name__ == "__main__":
    main()
