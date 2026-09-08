"""Reproduce MCDM sensitivity results for three-method averaged VaR criteria.

The input decision matrix must contain model-level risk diagnostics for
Normal, Student-t, and filtered historical simulation (FHS).  The four MCDM
risk criteria are the arithmetic means of the corresponding method-level
metrics at 1% and 5%; VaR thresholds are not averaged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stats_analysis.run_mcdm_evaluation import (  # noqa: E402
    Criterion,
    combined_ranking,
    saw_ranking,
    topsis_ranking,
)


ACCURACY_CRITERIA = (
    ("mse", "cost"),
    ("mae", "cost"),
    ("qlike", "cost"),
    ("volatility_std_ratio_error", "cost"),
    ("tracking_correlation_error", "cost"),
)
RISK_CRITERIA = (
    ("var_1pct_pass_rate", "benefit"),
    ("var_1pct_abs_violation_error", "cost"),
    ("var_5pct_pass_rate", "benefit"),
    ("var_5pct_abs_violation_error", "cost"),
)
EXPECTED_METHODS = ("normal", "student_t", "fhs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="MCDM run directory containing 5,5 and 3,7 subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("averaged_var_results"),
    )
    return parser.parse_args()


def resolve_run_root(path: Path | None) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    candidates = sorted(
        (PROJECT_ROOT / "output" / "mcdm_results").glob("MCDM*"),
        reverse=True,
    )
    for candidate in candidates:
        input_path = candidate / "5,5" / "MCDMInputMetrics.csv"
        if input_path.exists() and len(pd.read_csv(input_path, usecols=["model"])) == 26:
            return candidate
    raise FileNotFoundError("No complete 26-configuration MCDM run found")


def validate_three_method_matrix(matrix: pd.DataFrame) -> None:
    required = []
    for var_case in ("var_1pct", "var_5pct"):
        for method in EXPECTED_METHODS:
            required.extend(
                [
                    f"{var_case}_{method}_pass_rate",
                    f"{var_case}_{method}_abs_violation_error",
                ]
            )
    missing = sorted(set(required) - set(matrix.columns))
    if missing:
        raise ValueError(f"Decision matrix lacks three-method metrics: {missing}")

    for var_case in ("var_1pct", "var_5pct"):
        for metric in ("pass_rate", "abs_violation_error"):
            method_columns = [
                f"{var_case}_{method}_{metric}" for method in EXPECTED_METHODS
            ]
            rebuilt = matrix[method_columns].mean(axis=1)
            if not np.allclose(
                rebuilt,
                matrix[f"{var_case}_{metric}"],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{var_case}_{metric} is not the mean of {EXPECTED_METHODS}"
                )


def criteria_for_accuracy_weight(accuracy_weight: float) -> tuple[Criterion, ...]:
    risk_weight = 1.0 - accuracy_weight
    return tuple(
        Criterion(name, direction, accuracy_weight / len(ACCURACY_CRITERIA))
        for name, direction in ACCURACY_CRITERIA
    ) + tuple(
        Criterion(name, direction, risk_weight / len(RISK_CRITERIA))
        for name, direction in RISK_CRITERIA
    )


def eligible_matrix(
    matrix: pd.DataFrame,
    criteria: tuple[Criterion, ...],
    correlation_threshold: float,
) -> pd.DataFrame:
    names = [criterion.name for criterion in criteria]
    complete = np.isfinite(
        matrix[names].apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    std_error = pd.to_numeric(
        matrix["volatility_std_ratio_error"], errors="coerce"
    )
    correlation = pd.to_numeric(
        matrix["tracking_correlation"], errors="coerce"
    )
    return matrix.loc[
        complete & (std_error <= 0.9) & (correlation > correlation_threshold)
    ].copy()


def rank_matrix(
    matrix: pd.DataFrame,
    criteria: tuple[Criterion, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    saw = saw_ranking(matrix, criteria)
    topsis = topsis_ranking(matrix, criteria)
    combined = combined_ranking(saw, topsis)
    return saw, topsis, combined


def winner_record(
    combined: pd.DataFrame,
    **metadata: float | int | str,
) -> dict:
    winner = combined.sort_values(
        ["avg_mcdm_rank_rank", "avg_mcdm_rank", "saw_rank", "topsis_rank", "model_id"]
    ).iloc[0]
    return {
        **metadata,
        "winner": winner["display_name"],
        "winner_model_id": winner["model_id"],
        "saw_rank": int(winner["saw_rank"]),
        "topsis_rank": int(winner["topsis_rank"]),
        "consensus_rank": int(winner["avg_mcdm_rank_rank"]),
    }


def main() -> None:
    args = parse_args()
    run_root = resolve_run_root(args.run_root)
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(run_root / "5,5" / "MCDMInputMetrics.csv")
    validate_three_method_matrix(matrix)

    agreement_records = []
    for folder, accuracy_weight in (("5,5", 0.5), ("3,7", 0.3)):
        combined = pd.read_csv(run_root / folder / "CombinedMCDMRanking.csv")
        rho, p_value = stats.spearmanr(
            combined["saw_rank"], combined["topsis_rank"]
        )
        agreement_records.append(
            {
                "policy": folder,
                "accuracy_weight": accuracy_weight,
                "risk_weight": 1.0 - accuracy_weight,
                "eligible_models": len(combined),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
            }
        )

    weight_records = []
    for accuracy_weight in np.arange(0.3, 0.71, 0.1):
        criteria = criteria_for_accuracy_weight(float(accuracy_weight))
        eligible = eligible_matrix(matrix, criteria, correlation_threshold=0.0)
        _, _, combined = rank_matrix(eligible, criteria)
        record = winner_record(
            combined,
            accuracy_weight=round(float(accuracy_weight), 2),
            risk_weight=round(float(1.0 - accuracy_weight), 2),
            eligible_models=len(eligible),
        )
        record["spearman_rho"] = float(
            stats.spearmanr(
                combined["saw_rank"], combined["topsis_rank"]
            ).statistic
        )
        weight_records.append(record)

    gate_records = []
    balanced = criteria_for_accuracy_weight(0.5)
    for threshold in (0.0, 0.05, 0.10):
        eligible = eligible_matrix(matrix, balanced, correlation_threshold=threshold)
        _, _, combined = rank_matrix(eligible, balanced)
        gate_records.append(
            winner_record(
                combined,
                correlation_threshold=threshold,
                eligible_models=len(eligible),
            )
        )

    definition_records = []
    for definition in ("average", *EXPECTED_METHODS):
        definition_matrix = matrix.copy()
        if definition != "average":
            for var_case in ("var_1pct", "var_5pct"):
                for metric in ("pass_rate", "abs_violation_error"):
                    definition_matrix[f"{var_case}_{metric}"] = definition_matrix[
                        f"{var_case}_{definition}_{metric}"
                    ]

        for policy, accuracy_weight in (("5,5", 0.5), ("3,7", 0.3)):
            criteria = criteria_for_accuracy_weight(accuracy_weight)
            eligible = eligible_matrix(
                definition_matrix, criteria, correlation_threshold=0.0
            )
            _, _, combined = rank_matrix(eligible, criteria)
            for row in combined.itertuples(index=False):
                definition_records.append(
                    {
                        "risk_definition": definition,
                        "policy": policy,
                        "display_name": row.display_name,
                        "model_id": row.model_id,
                        "saw_rank": int(row.saw_rank),
                        "topsis_rank": int(row.topsis_rank),
                        "consensus_rank": int(row.avg_mcdm_rank_rank),
                        "average_rank": float(row.avg_mcdm_rank),
                    }
                )

    pd.DataFrame(agreement_records).to_csv(
        output_dir / "rank_agreement.csv", index=False
    )
    pd.DataFrame(weight_records).to_csv(
        output_dir / "weight_sensitivity.csv", index=False
    )
    pd.DataFrame(gate_records).to_csv(
        output_dir / "gate_sensitivity.csv", index=False
    )
    pd.DataFrame(definition_records).to_csv(
        output_dir / "risk_definition_rankings.csv", index=False
    )

    print(f"Run root: {run_root}")
    print(pd.DataFrame(agreement_records).to_string(index=False))
    print(pd.DataFrame(weight_records).to_string(index=False))
    print(pd.DataFrame(gate_records).to_string(index=False))
    definition_frame = pd.DataFrame(definition_records)
    print(
        definition_frame[
            definition_frame["consensus_rank"] == 1
        ].to_string(index=False)
    )
    print(f"Saved sensitivity outputs in {output_dir}")


if __name__ == "__main__":
    main()
