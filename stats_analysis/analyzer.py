from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import StatsAnalysisConfig
from .data_loader import LoadReport, PredictionDataLoader
from .metrics import ForecastMetrics
from .risk import (
    FilteredHistoricalVaRMethod,
    NormalVaRMethod,
    StudentTNuEstimator,
    StudentTVaRMethod,
    VaRMethod,
    VarBacktester,
)


@dataclass(frozen=True)
class AnalysisResult:
    detailed: pd.DataFrame
    aggregate: pd.DataFrame
    pass_cases: pd.DataFrame
    summary: pd.DataFrame
    var_predictions: pd.DataFrame


class StatsAnalysisPipeline:
    """Coordinate loading, metrics, risk backtesting, aggregation, and CSV export."""

    GROUP_COLUMNS = ["branch", "tier", "model", "dataset", "horizon"]
    MODEL_COLUMNS = ["branch", "tier", "model"]

    def __init__(self, config: StatsAnalysisConfig | None = None) -> None:
        self.config = config or StatsAnalysisConfig()
        self.loader = PredictionDataLoader(self.config)
        self.forecast_metrics = ForecastMetrics(epsilon=self.config.epsilon)
        self.nu_estimator = StudentTNuEstimator(min_nu=self.config.student_t_min_nu)
        self.backtester = VarBacktester(
            alpha=self.config.alpha,
            pvalue_threshold=self.config.pvalue_threshold,
            epsilon=self.config.epsilon,
            mu=self.config.mu,
        )
        self.var_methods: tuple[VaRMethod, ...] = (
            NormalVaRMethod(alpha=self.config.alpha, epsilon=self.config.epsilon, mu=self.config.mu),
            StudentTVaRMethod(alpha=self.config.alpha, epsilon=self.config.epsilon, mu=self.config.mu),
            FilteredHistoricalVaRMethod(
                alpha=self.config.alpha,
                epsilon=self.config.epsilon,
                mu=self.config.mu,
                rolling_window=self.config.rolling_window,
                min_history=self.config.min_history,
            ),
        )

    def run(self, save: bool = True) -> AnalysisResult:
        df, load_report = self.loader.load()
        nu_by_dataset = self._estimate_nu_by_dataset(df)
        detailed = self._build_detailed_stats(df, nu_by_dataset)
        pass_cases = self._build_pass_cases(detailed)
        aggregate = self._build_aggregate_stats(detailed)
        summary = self._build_summary(df, load_report, detailed, aggregate)
        var_predictions = self._build_var_predictions(df, nu_by_dataset)

        result = AnalysisResult(
            detailed=detailed,
            aggregate=aggregate,
            pass_cases=pass_cases,
            summary=summary,
            var_predictions=var_predictions,
        )
        if save:
            self.save(result)
        return result

    def save(self, result: AnalysisResult) -> dict[str, Path]:
        output_dir = self.config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "detailed": output_dir / self.config.detailed_filename,
            "aggregate": output_dir / self.config.aggregate_filename,
            "pass_cases": output_dir / self.config.pass_cases_filename,
            "summary": output_dir / self.config.summary_filename,
            "var_predictions": output_dir / self.config.var_predictions_filename,
        }

        result.detailed.to_csv(paths["detailed"], index=False)
        result.aggregate.to_csv(paths["aggregate"], index=False)
        result.pass_cases.to_csv(paths["pass_cases"], index=False)
        result.summary.to_csv(paths["summary"], index=False)
        result.var_predictions.to_csv(paths["var_predictions"], index=False)
        return paths

    def _estimate_nu_by_dataset(self, df: pd.DataFrame) -> dict[str, float]:
        return {
            str(dataset): self.nu_estimator.estimate(group["log_return"])
            for dataset, group in df.groupby("dataset", dropna=False)
        }

    def _build_detailed_stats(self, df: pd.DataFrame, nu_by_dataset: dict[str, float]) -> pd.DataFrame:
        records: list[dict] = []

        for group_key, group in df.groupby(self.GROUP_COLUMNS, dropna=False, sort=True):
            branch, tier, model, dataset, horizon = group_key
            group = group.sort_values("time")

            forecast = self.forecast_metrics.calculate(
                group["true_volatility"],
                group["predict_volatility"],
            )
            risk = self.backtester.calculate(
                group["log_return"],
                group["predict_volatility"],
                nu=nu_by_dataset.get(str(dataset), float("nan")),
            )

            records.append(
                {
                    "branch": branch,
                    "tier": tier,
                    "model": model,
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "rows": int(len(group)),
                    **forecast,
                    **risk,
                }
            )

        return pd.DataFrame.from_records(records)

    def _build_pass_cases(self, detailed: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "branch",
            "tier",
            "model",
            "dataset",
            "horizon",
            "n_risk",
            "violation_count",
            "violation_rate",
            "kupiec_lr",
            "kupiec_p",
            "lr_ind",
            "lr_ind_p",
            "kupiec_pass",
            "independence_pass",
            "backtest_pass",
        ]
        return detailed.loc[:, columns].copy()

    def _build_var_predictions(self, df: pd.DataFrame, nu_by_dataset: dict[str, float]) -> pd.DataFrame:
        columns = [
            "dataset",
            "branch",
            "tier",
            "model",
            "time",
            "horizon",
            "log_return",
            "predict_volatility",
        ]
        output = df.loc[:, columns].copy()

        for method in self.var_methods:
            output[f"{method.name}_var"] = float("nan")
            output[f"{method.name}_vio"] = pd.Series(pd.NA, index=output.index, dtype="boolean")

        for group_key, group in df.groupby(self.GROUP_COLUMNS, dropna=False, sort=True):
            _, _, _, dataset, _ = group_key
            group = group.sort_values("time")
            nu = nu_by_dataset.get(str(dataset), float("nan"))

            for method in self.var_methods:
                var_values = method.calculate(
                    group["log_return"],
                    group["predict_volatility"],
                    nu=nu,
                )
                var_col = f"{method.name}_var"
                vio_col = f"{method.name}_vio"
                output.loc[group.index, var_col] = var_values

                returns_arr = pd.to_numeric(group["log_return"], errors="coerce").to_numpy(dtype=float)
                valid = pd.notna(var_values) & pd.notna(returns_arr)
                vio = pd.Series(pd.NA, index=group.index, dtype="boolean")
                vio.loc[group.index[valid]] = returns_arr[valid] < var_values[valid]
                output.loc[group.index, vio_col] = vio

        return output

    def _build_aggregate_stats(self, detailed: pd.DataFrame) -> pd.DataFrame:
        grouped = detailed.groupby(self.MODEL_COLUMNS, dropna=False, sort=True)
        aggregate = grouped.agg(
            cases=("model", "size"),
            valid_forecast_cases=("n_forecast", lambda s: int((s > 0).sum())),
            valid_risk_cases=("n_risk", lambda s: int((s > 0).sum())),
            total_rows=("rows", "sum"),
            total_forecast_rows=("n_forecast", "sum"),
            total_risk_rows=("n_risk", "sum"),
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            qlike=("qlike", "mean"),
            violation_rate=("violation_rate", "mean"),
            kupiec_lr=("kupiec_lr", "mean"),
            kupiec_p=("kupiec_p", "mean"),
            lr_ind=("lr_ind", "mean"),
            lr_ind_p=("lr_ind_p", "mean"),
            passed_cases=("backtest_pass", "sum"),
        ).reset_index()

        aggregate["pass_rate"] = aggregate["passed_cases"] / aggregate["valid_risk_cases"].where(
            aggregate["valid_risk_cases"] > 0
        )
        return aggregate.sort_values(["pass_rate", "qlike"], ascending=[False, True]).reset_index(drop=True)

    def _build_summary(
        self,
        df: pd.DataFrame,
        load_report: LoadReport,
        detailed: pd.DataFrame,
        aggregate: pd.DataFrame,
    ) -> pd.DataFrame:
        risk_invalid_rows = int(df[["log_return", "predict_volatility"]].isna().any(axis=1).sum())
        forecast_invalid_rows = int(df[["true_volatility", "predict_volatility"]].isna().any(axis=1).sum())

        rows = {
            "input_csv": str(self.config.resolved_input_csv()),
            "output_dir": str(self.config.resolved_output_dir()),
            "input_rows": load_report.input_rows,
            "datasets": int(df["dataset"].nunique(dropna=True)),
            "models": int(df["model"].nunique(dropna=True)),
            "horizons": int(df["horizon"].nunique(dropna=True)),
            "detailed_cases": int(len(detailed)),
            "aggregate_models": int(len(aggregate)),
            "missing_return": load_report.missing_return,
            "missing_true_volatility": load_report.missing_true_volatility,
            "missing_predict_volatility": load_report.missing_predict_volatility,
            "forecast_invalid_rows": forecast_invalid_rows,
            "risk_invalid_rows": risk_invalid_rows,
            "alpha": self.config.alpha,
            "pvalue_threshold": self.config.pvalue_threshold,
            "epsilon": self.config.epsilon,
            "student_t_min_nu": self.config.student_t_min_nu,
            "mu": self.config.mu,
        }
        return pd.DataFrame([rows])
