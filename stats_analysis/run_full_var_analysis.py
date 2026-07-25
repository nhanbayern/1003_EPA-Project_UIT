from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stats_analysis import StatsAnalysisConfig, StatsAnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VaR 5% and VaR 1% analyses from a merged prediction CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Merged prediction CSV to analyze. If omitted, the latest "
            "output/merged_predictions/merged_all_predictions*.csv file is used."
        ),
    )
    return parser.parse_args()


def resolve_input_csv(input_csv: Path | None) -> Path:
    if input_csv is not None:
        path = input_csv if input_csv.is_absolute() else PROJECT_ROOT / input_csv
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
        return path

    merged_dir = PROJECT_ROOT / "output" / "merged_predictions"
    candidates = sorted(
        merged_dir.glob("merged_all_predictions*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    legacy_path = PROJECT_ROOT / "output" / "merged_all_predictions.csv"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        "Missing merged prediction CSV. Expected --input-csv, "
        "output/merged_predictions/merged_all_predictions*.csv, or "
        f"{legacy_path}."
    )


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
    args = parse_args()
    input_csv = resolve_input_csv(args.input_csv)
    print(f"Using input CSV: {input_csv}")

    run_case(
        "VaR 5%",
        StatsAnalysisConfig(
            input_csv=input_csv,
            alpha=0.05,
            output_dir=PROJECT_ROOT / "output" / "stats_analysis" / "var_5pct",
        ),
    )
    run_case(
        "VaR 1%",
        StatsAnalysisConfig(
            input_csv=input_csv,
            alpha=0.01,
            output_dir=PROJECT_ROOT / "output" / "stats_analysis" / "var_1pct",
        ),
    )


if __name__ == "__main__":
    main()
