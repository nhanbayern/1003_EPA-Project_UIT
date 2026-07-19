from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionPlotConfig:
    """Configuration for prediction time-series diagnostics."""

    project_root: Path = Path(__file__).resolve().parents[1]
    input_csv: Path | None = None
    output_dir: Path | None = None
    rolling_window: int = 20
    figure_dpi: int = 160

    def resolved_input_csv(self) -> Path:
        if self.input_csv is not None:
            return Path(self.input_csv)
        return self.project_root / "output" / "merged_all_predictions.csv"

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return self.project_root / "output" / "stats_analysis" / "prediction_timeseries"


class PredictionTimeSeriesPlotter:
    """Load prediction rows and produce model-level diagnostic time-series plots."""

    REQUIRED_COLUMNS = {
        "dataset",
        "branch",
        "tier",
        "model",
        "time",
        "horizon",
        "true_volatility",
        "predict_volatility",
    }

    def __init__(self, config: PredictionPlotConfig | None = None) -> None:
        self.config = config or PredictionPlotConfig()
        self.output_dir = self.config.resolved_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.resolved_input_csv(), low_memory=False)
        self._validate_schema(df)

        out = df.copy()
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out["tier"] = out["tier"].fillna("").astype(str).replace({"": "No Tier", "None": "No Tier"})
        out["model_label"] = (
            out["branch"].astype(str) + " / " + out["tier"].astype(str) + " / " + out["model"].astype(str)
        )
        out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("Int64")

        true = pd.to_numeric(out["true_volatility"], errors="coerce")
        pred = pd.to_numeric(out["predict_volatility"], errors="coerce")
        out["true_volatility"] = true
        out["predict_volatility"] = pred
        out["abs_error"] = (true - pred).abs()
        out["relative_error"] = out["abs_error"] / true.abs().clip(lower=1e-8)
        out["squared_error"] = (true - pred) ** 2
        return out.sort_values(["dataset", "horizon", "model_label", "time"]).reset_index(drop=True)

    def subset(
        self,
        df: pd.DataFrame,
        dataset: str,
        horizon: int,
        model_label: str | None = None,
    ) -> pd.DataFrame:
        mask = df["dataset"].eq(dataset) & df["horizon"].eq(horizon)
        if model_label is not None:
            mask &= df["model_label"].eq(model_label)
        return df.loc[mask].sort_values("time").copy()

    def available_model_labels(self, df: pd.DataFrame, dataset: str | None = None, horizon: int | None = None) -> list[str]:
        data = df
        if dataset is not None:
            data = data[data["dataset"].eq(dataset)]
        if horizon is not None:
            data = data[data["horizon"].eq(horizon)]
        return sorted(data["model_label"].dropna().unique().tolist())

    def outlier_table(self, df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        columns = [
            "dataset",
            "horizon",
            "model_label",
            "time",
            "true_volatility",
            "predict_volatility",
            "abs_error",
            "relative_error",
        ]
        valid = df.dropna(subset=["predict_volatility", "abs_error"]).copy()
        return valid.sort_values(["abs_error", "predict_volatility"], ascending=False).loc[:, columns].head(n)

    def plot_single_model(
        self,
        df: pd.DataFrame,
        dataset: str,
        horizon: int,
        model_label: str,
        *,
        log_y: bool = False,
        save: bool = True,
    ) -> Path | None:
        data = self.subset(df, dataset=dataset, horizon=horizon, model_label=model_label)
        data = data.dropna(subset=["time", "true_volatility", "predict_volatility"])
        if data.empty:
            raise ValueError(f"No valid rows for {dataset=} {horizon=} {model_label=}")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(data["time"], data["true_volatility"], label="True volatility", linewidth=1.4, color="#1f4e79")
        ax.plot(data["time"], data["predict_volatility"], label="Predicted volatility", linewidth=1.1, color="#b04a3f", alpha=0.85)

        rolling_error = data["abs_error"].rolling(self.config.rolling_window, min_periods=1).mean()
        ax2 = ax.twinx()
        ax2.plot(data["time"], rolling_error, label=f"Rolling abs error ({self.config.rolling_window})", color="#5f6f52", alpha=0.45)
        ax2.set_ylabel("Rolling absolute error")

        ax.set_title(f"{dataset} | h={horizon} | {model_label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        if log_y:
            ax.set_yscale("log")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()

        if not save:
            plt.show()
            return None

        path = self.output_dir / f"{self._safe_name(dataset)}_h{horizon}_{self._safe_name(model_label)}_timeseries{'_logy' if log_y else ''}.png"
        self._save(fig, path)
        plt.close(fig)
        return path

    def plot_overlay(
        self,
        df: pd.DataFrame,
        dataset: str,
        horizon: int,
        model_labels: list[str] | None = None,
        *,
        log_y: bool = False,
        save: bool = True,
    ) -> Path | None:
        data = self.subset(df, dataset=dataset, horizon=horizon)
        data = data.dropna(subset=["time", "true_volatility", "predict_volatility"])
        if model_labels is not None:
            data = data[data["model_label"].isin(model_labels)]
        if data.empty:
            raise ValueError(f"No valid rows for overlay {dataset=} {horizon=}")

        fig, ax = plt.subplots(figsize=(14, 6))
        true_series = data.groupby("time", as_index=False)["true_volatility"].median()
        ax.plot(true_series["time"], true_series["true_volatility"], label="True volatility", linewidth=2.0, color="black")

        for label, group in data.groupby("model_label", sort=True):
            ax.plot(group["time"], group["predict_volatility"], label=label, linewidth=0.85, alpha=0.65)

        ax.set_title(f"Prediction overlay | {dataset} | h={horizon}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()

        if not save:
            plt.show()
            return None

        suffix = "selected" if model_labels else "all_models"
        path = self.output_dir / f"{self._safe_name(dataset)}_h{horizon}_{suffix}_overlay{'_logy' if log_y else ''}.png"
        self._save(fig, path)
        plt.close(fig)
        return path

    def plot_small_multiples(
        self,
        df: pd.DataFrame,
        dataset: str,
        horizon: int,
        *,
        max_models: int | None = None,
        log_y: bool = False,
        save: bool = True,
    ) -> Path | None:
        data = self.subset(df, dataset=dataset, horizon=horizon)
        data = data.dropna(subset=["time", "true_volatility", "predict_volatility"])
        labels = sorted(data["model_label"].dropna().unique().tolist())
        if max_models is not None:
            labels = labels[:max_models]
        if not labels:
            raise ValueError(f"No model labels for {dataset=} {horizon=}")

        ncols = 2
        nrows = int(np.ceil(len(labels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4, 3.1 * nrows)), sharex=False)
        axes_arr = np.asarray(axes).reshape(-1)

        for ax, label in zip(axes_arr, labels):
            group = data[data["model_label"].eq(label)]
            ax.plot(group["time"], group["true_volatility"], color="#1f4e79", linewidth=1.1, label="True")
            ax.plot(group["time"], group["predict_volatility"], color="#b04a3f", linewidth=0.9, alpha=0.85, label="Pred")
            ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.22)
            if log_y:
                ax.set_yscale("log")
        for ax in axes_arr[len(labels):]:
            ax.axis("off")

        axes_arr[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Small multiples | {dataset} | h={horizon}", y=0.995)

        if not save:
            plt.show()
            return None

        path = self.output_dir / f"{self._safe_name(dataset)}_h{horizon}_small_multiples{'_logy' if log_y else ''}.png"
        self._save(fig, path)
        plt.close(fig)
        return path

    def export_all_single_model_plots(self, df: pd.DataFrame, *, log_y: bool = False) -> list[Path]:
        paths: list[Path] = []
        groups = df[["dataset", "horizon", "model_label"]].drop_duplicates().sort_values(["dataset", "horizon", "model_label"])
        for row in groups.itertuples(index=False):
            try:
                path = self.plot_single_model(df, row.dataset, int(row.horizon), row.model_label, log_y=log_y, save=True)
                if path is not None:
                    paths.append(path)
            except ValueError:
                continue
        return paths

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    def _save(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=self.config.figure_dpi, bbox_inches="tight")

    def _safe_name(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
        return cleaned[:140] or "unknown"

