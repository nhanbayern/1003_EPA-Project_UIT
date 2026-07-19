from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stats_analysis import StatsAnalysisConfig, StatsAnalysisPipeline


def run_case(name: str, config: StatsAnalysisConfig) -> None:
    print(f"Running {name} analysis...")
    StatsAnalysisPipeline(config).run(save=True)
    output_dir = config.resolved_output_dir()
    print(f"Finished {name}: {output_dir}")
    print(f"  detailed: {output_dir / config.detailed_filename}")
    print(f"  aggregate: {output_dir / config.aggregate_filename}")
    print(f"  pass_cases: {output_dir / config.pass_cases_filename}")
    print(f"  summary: {output_dir / config.summary_filename}")
    print(f"  var_predictions: {output_dir / config.var_predictions_filename}")


def main() -> None:
    run_case(
        "VaR 5%",
        StatsAnalysisConfig(
            alpha=0.05,
            output_dir=PROJECT_ROOT / "output" / "stats_analysis" / "var_5pct",
        ),
    )
    run_case(
        "VaR 1%",
        StatsAnalysisConfig(
            alpha=0.01,
            output_dir=PROJECT_ROOT / "output" / "stats_analysis" / "var_1pct",
        ),
    )


if __name__ == "__main__":
    main()
