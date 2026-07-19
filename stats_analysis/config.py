from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StatsAnalysisConfig:
    """Runtime configuration for the volatility statistics pipeline."""

    project_root: Path = Path(__file__).resolve().parents[1]
    input_csv: Path | None = None
    output_dir: Path | None = None
    alpha: float = 0.05
    pvalue_threshold: float = 0.05
    epsilon: float = 1e-8
    student_t_min_nu: float = 2.1
    mu: float = 0.0
    rolling_window: int = 250
    min_history: int = 250

    detailed_filename: str = "stats_by_dataset_horizon.csv"
    aggregate_filename: str = "stats_by_model.csv"
    pass_cases_filename: str = "backtest_pass_cases.csv"
    summary_filename: str = "stats_run_summary.csv"
    var_predictions_filename: str = "var_predictions.csv"

    def resolved_input_csv(self) -> Path:
        if self.input_csv is not None:
            return Path(self.input_csv)
        return self.project_root / "output" / "merged_all_predictions.csv"

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return self.project_root / "output" / "stats_analysis"
