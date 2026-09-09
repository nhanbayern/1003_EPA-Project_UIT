from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stats_analysis import StatsAnalysisConfig, StatsAnalysisPipeline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VaR 5% and VaR 1% analyses from a merged prediction CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Versioned merged prediction CSV produced by merge_canonical_predictions.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory for this analysis run; it must not already exist.",
    )
    return parser.parse_args()


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
    input_csv = args.input_csv if args.input_csv.is_absolute() else PROJECT_ROOT / args.input_csv
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    if not input_csv.is_file():
        raise FileNotFoundError(f"Missing --input-csv: {input_csv}")
    if output_root.exists():
        raise FileExistsError(f"--output-root already exists: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_csv": str(input_csv.resolve()),
                "input_csv_sha256": sha256(input_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Using input CSV: {input_csv}")
    print(f"Analysis run root: {output_root}")

    run_case(
        "VaR 5%",
        StatsAnalysisConfig(
            input_csv=input_csv,
            alpha=0.05,
            output_dir=output_root / "var_5pct",
        ),
    )
    run_case(
        "VaR 1%",
        StatsAnalysisConfig(
            input_csv=input_csv,
            alpha=0.01,
            output_dir=output_root / "var_1pct",
        ),
    )


if __name__ == "__main__":
    main()
