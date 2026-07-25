from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import StatsAnalysisConfig


@dataclass(frozen=True)
class LoadReport:
    input_rows: int
    missing_return: int
    missing_true_volatility: int
    missing_predict_volatility: int


class PredictionDataLoader:
    """Load, validate, and normalize merged prediction data."""

    REQUIRED_COLUMNS = {
        "dataset",
        "branch",
        "tier",
        "model",
        "time",
        "horizon",
        "log_return",
        "true_volatility",
        "predict_volatility",
    }

    def __init__(self, config: StatsAnalysisConfig) -> None:
        self.config = config

    def load(self) -> tuple[pd.DataFrame, LoadReport]:
        input_csv = self.config.resolved_input_csv()
        df = pd.read_csv(input_csv, low_memory=False)
        self._validate_schema(df)

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["tier"] = df["tier"].fillna("").astype(str)
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")

        report = LoadReport(
            input_rows=int(len(df)),
            missing_return=int(df["log_return"].isna().sum()),
            missing_true_volatility=int(df["true_volatility"].isna().sum()),
            missing_predict_volatility=int(df["predict_volatility"].isna().sum()),
        )
        return df, report

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
