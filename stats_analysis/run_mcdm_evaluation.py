from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_COLUMNS = ["branch", "tier", "model"]


@dataclass(frozen=True)
class Criterion:
    name: str
    direction: str
    weight: float

    @property
    def is_benefit(self) -> bool:
        if self.direction not in {"benefit", "cost"}:
            raise ValueError(f"Invalid direction for {self.name}: {self.direction}")
        return self.direction == "benefit"


@dataclass(frozen=True)
class McdmScenario:
    name: str
    folder: str
    label: str
    description: str
    criteria: tuple[Criterion, ...]


CRITERIA_5_5 = (
    Criterion("mse", "cost", 0.1),
    Criterion("mae", "cost", 0.1),
    Criterion("qlike", "cost", 0.1),
    Criterion("volatility_std_ratio_error", "cost", 0.1),
    Criterion("tracking_correlation_error", "cost", 0.1),
    Criterion("var_1pct_pass_rate", "benefit", 0.125),
    Criterion("var_1pct_abs_violation_error", "cost", 0.125),
    Criterion("var_5pct_pass_rate", "benefit", 0.125),
    Criterion("var_5pct_abs_violation_error", "cost", 0.125),
)


CRITERIA_3_7 = (
    Criterion("mse", "cost", 0.06),
    Criterion("mae", "cost", 0.06),
    Criterion("qlike", "cost", 0.06),
    Criterion("volatility_std_ratio_error", "cost", 0.06),
    Criterion("tracking_correlation_error", "cost", 0.06),
    Criterion("var_1pct_pass_rate", "benefit", 0.175),
    Criterion("var_1pct_abs_violation_error", "cost", 0.175),
    Criterion("var_5pct_pass_rate", "benefit", 0.175),
    Criterion("var_5pct_abs_violation_error", "cost", 0.175),
)


MCDM_SCENARIOS = (
    McdmScenario(
        name="criteria_5_5",
        folder="5,5",
        label="Accuracy:Risk = 50:50",
        description="Accuracy 50%, risk 50%; inside risk, pass_rate 50% and abs_violation_error 50%.",
        criteria=CRITERIA_5_5,
    ),
    McdmScenario(
        name="criteria_3_7",
        folder="3,7",
        label="Accuracy:Risk = 30:70",
        description="Accuracy 30%, risk 70%; inside each block, criteria are evenly weighted.",
        criteria=CRITERIA_3_7,
    ),
)


VAR_CASES = {
    "var_1pct": 0.01,
    "var_5pct": 0.05,
}


METRIC_LABELS = {
    "mse": "Mean Squared Error (MSE)",
    "mae": "Mean Absolute Error (MAE)",
    "qlike": "Quasi-Likelihood Loss (QLIKE)",
    "volatility_std_ratio_error": "Volatility Standard-Deviation Ratio Error",
    "tracking_correlation": "Volatility Tracking Correlation",
    "tracking_correlation_error": "Volatility Tracking Correlation Error",
    "var_1pct_pass_rate": "VaR 1% Backtesting Pass Rate",
    "var_1pct_violation_rate": "VaR 1% Violation Rate",
    "var_1pct_abs_violation_error": "VaR 1% Absolute Violation Error",
    "var_5pct_pass_rate": "VaR 5% Backtesting Pass Rate",
    "var_5pct_violation_rate": "VaR 5% Violation Rate",
    "var_5pct_abs_violation_error": "VaR 5% Absolute Violation Error",
    "saw_score": "SAW Composite Score",
    "topsis_score": "TOPSIS Closeness Coefficient",
    "accuracy_score": "Forecast Accuracy Component",
    "risk_score": "Risk Calibration Component",
}


PREDICTION_CSV_CANDIDATES = (
    PROJECT_ROOT / "output" / "merged_predictions" / "merged_all_predictions_24_7.csv",
    PROJECT_ROOT / "output" / "merged_predictions" / "merged_all_predictions.csv",
    PROJECT_ROOT / "output" / "merged_all_predictions.csv",
)
MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY = 0.9
MIN_TRACKING_CORRELATION_FOR_ELIGIBILITY = 0.0


def model_id(df: pd.DataFrame) -> pd.Series:
    tier = df["tier"].fillna("").astype(str)
    return df["branch"].astype(str) + "|" + tier + "|" + df["model"].astype(str)


def _clean_tier(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text in {"", "nan", "NaN", "None", "No Tier"}:
        return ""
    return text


def display_model_name(branch: object, tier: object, model: object, *, include_tier: bool = True) -> str:
    branch_text = "" if pd.isna(branch) else str(branch)
    tier_text = _clean_tier(tier)
    model_text = "" if pd.isna(model) else str(model)

    model_map = {
        "HybridGARCHAutoformer": "GARCH-Autoformer",
        "WaveletAutoformer": "Wavelet-Autoformer",
        "moirai": "Moirai",
        "moirai2": "Moirai 2",
        "moirai_moe": "Moirai-MoE",
    }
    branch_map = {
        "Moirai_VAR": "MoiraiVaR",
        "Moirai_VAR_FT": "MoiraiVaR-FT",
        "Moirai": "",
        "modified_autoformer": "",
        "modify_autoformer": "",
        "Transformers": "",
        "GARCH": "",
    }

    clean_model = model_map.get(model_text, model_text)
    clean_branch = branch_map.get(branch_text, branch_text.replace("_", " "))

    if branch_text in {"modified_autoformer", "modify_autoformer"}:
        base = clean_model
    elif branch_text.startswith("Moirai_VAR"):
        base = f"{clean_branch} - {clean_model}"
    elif clean_branch and clean_branch != clean_model:
        base = f"{clean_branch} {clean_model}"
    else:
        base = clean_model

    if include_tier and tier_text:
        tier_label = tier_text.replace("Tier_1_Miniaturized", "Tier 1").replace(
            "Tier_2_Standard", "Tier 2"
        ).replace("Tier_3_Large", "Tier 3").replace("lambda_0.2", "lambda=0.2")
        return f"{base} ({tier_label})"
    return base


def _pascal_token(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    return "".join(word[:1].upper() + word[1:] for word in words)


def compact_model_tier_name(branch: object, tier: object, model: object) -> str:
    base = display_model_name(branch, tier, model, include_tier=False)
    tier_text = _clean_tier(tier)
    label = _pascal_token(base)
    if not tier_text:
        return label

    tier_match = re.search(r"Tier[_\s-]*(\d+)", tier_text, flags=re.IGNORECASE)
    if tier_match:
        return f"{label}Tier{tier_match.group(1)}"

    lambda_match = re.search(r"lambda[_\s=-]*(\d+(?:\.\d+)?)", tier_text, flags=re.IGNORECASE)
    if lambda_match:
        lambda_value = lambda_match.group(1).replace(".", "")
        return f"{label}Lambda{lambda_value}"

    return f"{label}{_pascal_token(tier_text)}"


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["display_name"] = [
        display_model_name(row.branch, row.tier, row.model, include_tier=True)
        for row in out.itertuples(index=False)
    ]
    out["display_group"] = [
        display_model_name(row.branch, row.tier, row.model, include_tier=False)
        for row in out.itertuples(index=False)
    ]
    out["display_model_tier"] = [
        compact_model_tier_name(row.branch, row.tier, row.model)
        for row in out.itertuples(index=False)
    ]
    return out


def metric_label(name: str) -> str:
    return METRIC_LABELS.get(str(name), str(name).replace("_", " ").title())


def validate_weights(criteria: tuple[Criterion, ...]) -> None:
    total_weight = sum(c.weight for c in criteria)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"MCDM weights must sum to 1.0, got {total_weight}")


def criteria_frame(scenario: McdmScenario) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": scenario.name,
                "folder": scenario.folder,
                "scenario_label": scenario.label,
                "description": scenario.description,
                "criterion": criterion.name,
                "metric_label": metric_label(criterion.name),
                "direction": criterion.direction,
                "weight": criterion.weight,
            }
            for criterion in scenario.criteria
        ]
    )


def load_case(stats_root: Path, var_case: str, alpha: float) -> pd.DataFrame:
    path = stats_root / var_case / "stats_by_model.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing stats file: {path}")

    df = pd.read_csv(path)
    required = {*MODEL_COLUMNS, "mse", "mae", "qlike", "pass_rate", "violation_rate"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df.loc[:, [*MODEL_COLUMNS, "mse", "mae", "qlike", "pass_rate", "violation_rate"]].copy()
    out[f"{var_case}_pass_rate"] = pd.to_numeric(out["pass_rate"], errors="coerce")
    out[f"{var_case}_violation_rate"] = pd.to_numeric(out["violation_rate"], errors="coerce")
    out[f"{var_case}_abs_violation_error"] = (out[f"{var_case}_violation_rate"] - alpha).abs()
    return out.drop(columns=["pass_rate", "violation_rate"])


def resolve_prediction_csv() -> Path:
    for path in PREDICTION_CSV_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in PREDICTION_CSV_CANDIDATES)
    raise FileNotFoundError(f"Missing prediction CSV. Checked:\n{candidates}")


def _tracking_stats(group: pd.DataFrame) -> dict[str, float | int]:
    true_values = pd.to_numeric(group["true_volatility"], errors="coerce")
    pred_values = pd.to_numeric(group["predict_volatility"], errors="coerce")
    valid = pd.DataFrame({"true": true_values, "pred": pred_values}).dropna()

    if valid.empty:
        return {
            "tracking_rows": 0,
            "true_volatility_std": np.nan,
            "predict_volatility_std": np.nan,
            "volatility_std_ratio": np.nan,
            "volatility_std_ratio_error": np.nan,
            "tracking_correlation": np.nan,
            "tracking_correlation_error": np.nan,
        }

    true_std = float(valid["true"].std(ddof=0))
    pred_std = float(valid["pred"].std(ddof=0))
    if np.isfinite(true_std) and true_std > 0:
        std_ratio = pred_std / true_std
        std_ratio_error = abs(std_ratio - 1.0)
    else:
        std_ratio = np.nan
        std_ratio_error = np.nan

    if np.isfinite(true_std) and true_std > 1e-12 and np.isfinite(pred_std) and pred_std > 1e-12:
        tracking_corr = float(valid["true"].corr(valid["pred"]))
    else:
        tracking_corr = 0.0
    if not np.isfinite(tracking_corr):
        tracking_corr = 0.0
    tracking_corr = float(np.clip(tracking_corr, -1.0, 1.0))
    tracking_corr_error = 1.0 - max(tracking_corr, 0.0)

    return {
        "tracking_rows": int(len(valid)),
        "true_volatility_std": true_std,
        "predict_volatility_std": pred_std,
        "volatility_std_ratio": std_ratio,
        "volatility_std_ratio_error": std_ratio_error,
        "tracking_correlation": tracking_corr,
        "tracking_correlation_error": tracking_corr_error,
    }


def build_tracking_metrics(prediction_csv: Path) -> pd.DataFrame:
    case_columns = [*MODEL_COLUMNS, "dataset", "horizon"]
    required = [*case_columns, "true_volatility", "predict_volatility"]
    df = pd.read_csv(prediction_csv, usecols=required, low_memory=False)
    df["branch"] = df["branch"].replace({"modify_autoformer": "modified_autoformer"})

    case_records: list[dict] = []
    for group_key, group in df.groupby(case_columns, dropna=False, sort=True):
        branch, tier, model, dataset, horizon = group_key
        case_records.append(
            {
                "branch": branch,
                "tier": tier,
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                **_tracking_stats(group),
            }
        )

    cases = pd.DataFrame.from_records(case_records)
    aggregate = (
        cases.groupby(MODEL_COLUMNS, dropna=False, sort=True)
        .agg(
            tracking_cases=("model", "size"),
            valid_tracking_cases=("tracking_rows", lambda s: int((s > 0).sum())),
            tracking_rows=("tracking_rows", "sum"),
            true_volatility_std=("true_volatility_std", "mean"),
            predict_volatility_std=("predict_volatility_std", "mean"),
            volatility_std_ratio=("volatility_std_ratio", "mean"),
            volatility_std_ratio_error=("volatility_std_ratio_error", "mean"),
            tracking_correlation=("tracking_correlation", "mean"),
            tracking_correlation_error=("tracking_correlation_error", "mean"),
        )
        .reset_index()
    )
    return aggregate


def build_decision_matrix(stats_root: Path) -> pd.DataFrame:
    cases = [load_case(stats_root, var_case, alpha) for var_case, alpha in VAR_CASES.items()]

    matrix = cases[0]
    for case in cases[1:]:
        matrix = matrix.merge(case, on=[*MODEL_COLUMNS, "mse", "mae", "qlike"], how="inner")

    tracking = build_tracking_metrics(resolve_prediction_csv())
    matrix = matrix.merge(tracking, on=MODEL_COLUMNS, how="left")
    matrix["model_id"] = model_id(matrix)
    return add_display_columns(matrix)


def minmax_score(series: pd.Series, benefit: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index)

    min_value = finite.min()
    max_value = finite.max()
    if np.isclose(max_value, min_value):
        return pd.Series(1.0, index=series.index)

    if benefit:
        return (values - min_value) / (max_value - min_value)
    return (max_value - values) / (max_value - min_value)


def saw_ranking(matrix: pd.DataFrame, criteria: tuple[Criterion, ...]) -> pd.DataFrame:
    out = matrix.copy()
    score_columns = []
    weighted_columns = []
    risk_metric_names = {
        "var_1pct_pass_rate",
        "var_1pct_abs_violation_error",
        "var_5pct_pass_rate",
        "var_5pct_abs_violation_error",
    }
    accuracy_weighted_columns = []
    risk_weighted_columns = []

    for criterion in criteria:
        score_column = f"{criterion.name}_saw_score"
        weighted_column = f"{criterion.name}_weighted_score"
        out[score_column] = minmax_score(out[criterion.name], benefit=criterion.is_benefit)
        out[weighted_column] = out[score_column] * criterion.weight
        score_columns.append(score_column)
        weighted_columns.append(weighted_column)
        if criterion.name in risk_metric_names:
            risk_weighted_columns.append(weighted_column)
        else:
            accuracy_weighted_columns.append(weighted_column)

    out["accuracy_score"] = out[accuracy_weighted_columns].sum(axis=1)
    out["risk_score"] = out[risk_weighted_columns].sum(axis=1)
    out["saw_score"] = out[weighted_columns].sum(axis=1)
    out["saw_rank"] = out["saw_score"].rank(method="min", ascending=False).astype("Int64")

    columns = [
        "saw_rank",
        *MODEL_COLUMNS,
        "model_id",
        "display_name",
        "display_group",
        "accuracy_score",
        "risk_score",
        "saw_score",
        *[c.name for c in criteria],
        *score_columns,
    ]
    return out.loc[:, columns].sort_values(["saw_rank", "model_id"]).reset_index(drop=True)


def topsis_ranking(matrix: pd.DataFrame, criteria: tuple[Criterion, ...]) -> pd.DataFrame:
    out = matrix.copy()
    criterion_names = [c.name for c in criteria]
    raw = out.loc[:, criterion_names].apply(pd.to_numeric, errors="coerce")

    if raw.isna().any().any():
        missing = raw.columns[raw.isna().any()].tolist()
        raise ValueError(f"TOPSIS cannot run with missing values in criteria: {missing}")

    normalized = raw.copy()
    for criterion in criteria:
        denom = float(np.sqrt(np.square(raw[criterion.name]).sum()))
        if np.isclose(denom, 0.0):
            normalized[criterion.name] = 0.0
        else:
            normalized[criterion.name] = raw[criterion.name] / denom
        normalized[criterion.name] *= criterion.weight

    ideal_best = {}
    ideal_worst = {}
    for criterion in criteria:
        values = normalized[criterion.name]
        if criterion.is_benefit:
            ideal_best[criterion.name] = values.max()
            ideal_worst[criterion.name] = values.min()
        else:
            ideal_best[criterion.name] = values.min()
            ideal_worst[criterion.name] = values.max()

    best = pd.Series(ideal_best)
    worst = pd.Series(ideal_worst)
    out["distance_to_ideal"] = np.sqrt(np.square(normalized - best).sum(axis=1))
    out["distance_to_worst"] = np.sqrt(np.square(normalized - worst).sum(axis=1))
    denominator = out["distance_to_ideal"] + out["distance_to_worst"]
    out["topsis_score"] = out["distance_to_worst"] / denominator.replace(0, np.nan)
    out["topsis_rank"] = out["topsis_score"].rank(method="min", ascending=False).astype("Int64")

    columns = [
        "topsis_rank",
        *MODEL_COLUMNS,
        "model_id",
        "display_name",
        "display_group",
        "distance_to_ideal",
        "distance_to_worst",
        "topsis_score",
        *criterion_names,
    ]
    return out.loc[:, columns].sort_values(["topsis_rank", "model_id"]).reset_index(drop=True)


def combined_ranking(saw: pd.DataFrame, topsis: pd.DataFrame) -> pd.DataFrame:
    key_columns = [*MODEL_COLUMNS, "model_id", "display_name", "display_group"]
    out = saw.loc[:, [*key_columns, "accuracy_score", "risk_score", "saw_score", "saw_rank"]].merge(
        topsis.loc[:, [*key_columns, "topsis_score", "topsis_rank"]],
        on=key_columns,
        how="inner",
    )
    out["avg_mcdm_rank"] = out[["saw_rank", "topsis_rank"]].mean(axis=1)
    out["avg_mcdm_rank_rank"] = out["avg_mcdm_rank"].rank(method="min", ascending=True).astype("Int64")
    return out.sort_values(["avg_mcdm_rank_rank", "avg_mcdm_rank", "model_id"]).reset_index(drop=True)


def split_eligible_models(
    matrix: pd.DataFrame,
    criteria: tuple[Criterion, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    criterion_names = [c.name for c in criteria]
    numeric = matrix.loc[:, criterion_names].apply(pd.to_numeric, errors="coerce")
    complete_criteria_mask = np.isfinite(numeric).all(axis=1)
    std_error = pd.to_numeric(matrix["volatility_std_ratio_error"], errors="coerce")
    tracking_corr = pd.to_numeric(matrix["tracking_correlation"], errors="coerce")
    sanity_mask = (std_error <= MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY) & (
        tracking_corr > MIN_TRACKING_CORRELATION_FOR_ELIGIBILITY
    )
    eligible_mask = complete_criteria_mask & sanity_mask

    eligible = matrix.loc[eligible_mask].copy()
    excluded = matrix.loc[~eligible_mask].copy()
    if excluded.empty:
        excluded["excluded_reason"] = pd.Series(dtype=str)
        return eligible, excluded

    reasons = []
    for idx, row in numeric.loc[~eligible_mask].iterrows():
        row_reasons = []
        missing = [column for column, value in row.items() if not np.isfinite(value)]
        if missing:
            row_reasons.append("missing criteria: " + ", ".join(missing))

        current_std_error = std_error.loc[idx]
        current_corr = tracking_corr.loc[idx]
        if np.isfinite(current_std_error) and current_std_error > MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY:
            row_reasons.append(
                f"failed forecast sanity gate: volatility_std_ratio_error={current_std_error:.6f} "
                f"> {MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY:.6f}"
            )
        if np.isfinite(current_corr) and current_corr <= MIN_TRACKING_CORRELATION_FOR_ELIGIBILITY:
            row_reasons.append(
                f"failed forecast sanity gate: tracking_correlation={current_corr:.6f} "
                f"<= {MIN_TRACKING_CORRELATION_FOR_ELIGIBILITY:.6f}"
            )
        if not row_reasons:
            row_reasons.append("failed forecast sanity gate")
        reasons.append("; ".join(row_reasons))
    excluded["excluded_reason"] = reasons
    return eligible, excluded


def short_label(value: str, max_length: int = 42) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def aggregate_for_plots(df: pd.DataFrame, *, rank_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"branch", "tier", "model", "model_id", "display_name", "display_group", "display_model_tier"}
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    agg_spec = {col: (col, "mean") for col in numeric_cols}
    out = df.groupby("display_group", dropna=False, sort=True).agg(**agg_spec).reset_index()
    out["tier_count"] = df.groupby("display_group", dropna=False)["tier"].nunique().to_numpy()
    for rank_column in rank_columns:
        if rank_column in out.columns:
            out[f"{rank_column}_group_rank"] = out[rank_column].rank(method="min", ascending=True).astype("Int64")
    return out


def model_color_map(labels: pd.Series | list[str]) -> dict[str, tuple[float, float, float, float]]:
    unique_labels = sorted(pd.Series(labels).dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20")
    return {label: cmap(index % cmap.N) for index, label in enumerate(unique_labels)}


def add_model_color_legend(
    ax,
    color_map: dict[str, tuple[float, float, float, float]],
    *,
    title: str = "Model",
    max_items: int | None = None,
) -> None:
    items = list(color_map.items())
    if max_items is not None:
        items = items[:max_items]
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in items]
    ax.legend(
        handles=handles,
        title=title,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )


def save_bar_plot(
    df: pd.DataFrame,
    *,
    score_column: str,
    rank_column: str,
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    plot_df = aggregate_for_plots(df, rank_columns=(rank_column,))
    group_rank_column = f"{rank_column}_group_rank"
    plot_df = plot_df.sort_values(group_rank_column, ascending=True).head(top_n).copy()
    plot_df = plot_df.sort_values(score_column, ascending=True)
    labels = plot_df["display_group"].map(short_label)
    colors = model_color_map(plot_df["display_group"])
    bar_colors = plot_df["display_group"].map(colors)

    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.38 * len(plot_df))))
    ax.barh(labels, pd.to_numeric(plot_df[score_column], errors="coerce"), color=bar_colors)
    ax.set_title(title)
    ax.set_xlabel(metric_label(score_column))
    ax.grid(axis="x", alpha=0.25)
    add_model_color_legend(ax, colors)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_risk_plot(saw: pd.DataFrame, combined: pd.DataFrame, output_path: Path) -> None:
    plot_df = saw.merge(
        combined.loc[:, ["model_id", "avg_mcdm_rank_rank"]],
        on="model_id",
        how="left",
    ).copy()
    plot_df = aggregate_for_plots(plot_df, rank_columns=("avg_mcdm_rank_rank",))
    colors = model_color_map(plot_df["display_group"])

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for _, row in plot_df.iterrows():
        ax.scatter(
            pd.to_numeric(row["accuracy_score"], errors="coerce"),
            pd.to_numeric(row["risk_score"], errors="coerce"),
            s=90,
            alpha=0.82,
            color=colors[str(row["display_group"])],
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_title("Forecast Accuracy vs Risk Calibration")
    ax.set_xlabel("Forecast Accuracy Component Score")
    ax.set_ylabel("Risk Calibration Component Score")
    ax.grid(alpha=0.25)
    add_model_color_legend(ax, colors)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_weight_plot(criteria: tuple[Criterion, ...], output_path: Path) -> None:
    weights = pd.DataFrame(
        [
            {
                "criterion": c.name,
                "metric_label": metric_label(c.name),
                "weight": c.weight,
                "direction": c.direction,
            }
            for c in criteria
        ]
    ).sort_values("weight", ascending=True)
    colors = weights["direction"].map({"benefit": "#2d7d46", "cost": "#8f4b3b"}).fillna("#555555")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(weights["metric_label"], weights["weight"], color=colors)
    ax.set_title("MCDM Criteria Weights")
    ax.set_xlabel("Criterion Weight")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rank_comparison_plot(combined: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    plot_df = aggregate_for_plots(combined, rank_columns=("avg_mcdm_rank",))
    plot_df = plot_df.sort_values("avg_mcdm_rank", ascending=True).head(top_n).copy()
    plot_df = plot_df.iloc[::-1]
    y = np.arange(len(plot_df))
    colors = model_color_map(plot_df["display_group"])

    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.38 * len(plot_df))))
    for index, row in enumerate(plot_df.itertuples(index=False)):
        color = colors[str(row.display_group)]
        saw_rank = float(row.saw_rank)
        topsis_rank = float(row.topsis_rank)
        ax.plot([saw_rank, topsis_rank], [index, index], color=color, linewidth=2.2, alpha=0.75)
        ax.scatter(saw_rank, index, marker="o", s=64, color=color, edgecolor="white", linewidth=0.8)
        ax.scatter(topsis_rank, index, marker="s", s=64, color=color, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_group"].map(short_label))
    ax.set_title("SAW and TOPSIS Rank Comparison")
    ax.set_xlabel("Rank (Lower Is Better)")
    ax.grid(axis="x", alpha=0.25)
    model_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in colors.items()]
    method_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=7, label="SAW Rank"),
        Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=7, label="TOPSIS Rank"),
    ]
    first_legend = ax.legend(
        handles=method_handles,
        title="Ranking Method",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=model_handles,
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.01, 0.42),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tracking_penalty_plot(matrix: pd.DataFrame, combined: pd.DataFrame, output_path: Path, top_n: int = 18) -> None:
    plot_df = matrix.merge(
        combined.loc[:, ["model_id", "avg_mcdm_rank_rank"]],
        on="model_id",
        how="left",
    )
    plot_df = aggregate_for_plots(plot_df, rank_columns=("avg_mcdm_rank_rank",))
    plot_df = plot_df.sort_values("avg_mcdm_rank_rank", ascending=True).head(top_n).copy()
    plot_df = plot_df.iloc[::-1]
    y = np.arange(len(plot_df))
    colors = model_color_map(plot_df["display_group"])
    bar_colors = plot_df["display_group"].map(colors)

    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.38 * len(plot_df))))
    ax.barh(
        y - 0.18,
        pd.to_numeric(plot_df["volatility_std_ratio_error"], errors="coerce"),
        height=0.36,
        label="Std-Ratio Error",
        color=bar_colors,
        alpha=0.95,
    )
    ax.barh(
        y + 0.18,
        pd.to_numeric(plot_df["tracking_correlation_error"], errors="coerce"),
        height=0.36,
        label="Correlation Error",
        color=bar_colors,
        alpha=0.45,
        hatch="//",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_group"].map(short_label))
    ax.set_title("Volatility Dynamics Penalty for Top Ranked Models")
    ax.set_xlabel("Penalty Value (Lower Is Better)")
    ax.grid(axis="x", alpha=0.25)
    metric_handles = [
        Patch(facecolor="gray", edgecolor="gray", alpha=0.95, label="Std-Ratio Error"),
        Patch(facecolor="gray", edgecolor="gray", alpha=0.45, hatch="//", label="Correlation Error"),
    ]
    first_legend = ax.legend(
        handles=metric_handles,
        title="Penalty Metric",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(first_legend)
    add_model_color_legend(ax, colors)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_std_ratio_error_by_model_tier_plot(matrix: pd.DataFrame, output_path: Path) -> None:
    plot_df = matrix.copy()
    plot_df["volatility_std_ratio_error"] = pd.to_numeric(
        plot_df["volatility_std_ratio_error"],
        errors="coerce",
    )
    plot_df = plot_df.dropna(subset=["volatility_std_ratio_error"]).sort_values(
        "volatility_std_ratio_error",
        ascending=True,
    )
    colors = model_color_map(plot_df["display_group"])
    bar_colors = plot_df["display_group"].map(colors)
    labels = plot_df["display_model_tier"].map(lambda value: short_label(value, max_length=36))
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(14, max(6.5, 0.32 * len(plot_df))))
    ax.barh(y, plot_df["volatility_std_ratio_error"], color=bar_colors, alpha=0.92)
    ax.axvline(
        MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY,
        color="#b3261e",
        linestyle="--",
        linewidth=1.8,
        label=f"Eligibility Threshold = {MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY:g}",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Volatility Std Ratio Error by Model Tier")
    ax.set_xlabel("Volatility Std Ratio Error (Lower Is Better)")
    ax.grid(axis="x", alpha=0.25)

    threshold_handle = Line2D(
        [0],
        [0],
        color="#b3261e",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold = {MAX_STD_RATIO_ERROR_FOR_ELIGIBILITY:g}",
    )
    model_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in colors.items()]
    first_legend = ax.legend(
        handles=[threshold_handle],
        title="Sanity Gate",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=model_handles,
        title="Model Family",
        loc="center left",
        bbox_to_anchor=(1.01, 0.45),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def garch_autoformer_dominance(combined: pd.DataFrame, scenario: McdmScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = aggregate_for_plots(combined, rank_columns=("avg_mcdm_rank",))
    baseline_rows = grouped[grouped["display_group"].eq("GARCH-Autoformer")]
    if baseline_rows.empty:
        empty_summary = pd.DataFrame(
            [
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "baseline_model": "GARCH-Autoformer",
                    "metric": metric,
                    "baseline_score": np.nan,
                    "total_remaining_models": 0,
                    "h1_count": 0,
                    "h1_rate_pct": np.nan,
                    "mean_other_score": np.nan,
                    "mean_relative_difference_pct": np.nan,
                    "note": "GARCH-Autoformer is not available in eligible MCDM ranking",
                }
                for metric in ("saw_score", "topsis_score")
            ]
        )
        return empty_summary, pd.DataFrame()

    baseline = baseline_rows.iloc[0]
    others = grouped[~grouped["display_group"].eq("GARCH-Autoformer")].copy()
    pairwise_records: list[dict] = []
    summary_records: list[dict] = []

    for score_column, label in (
        ("saw_score", "SAW Composite Score"),
        ("topsis_score", "TOPSIS Closeness Coefficient"),
    ):
        baseline_score = float(baseline[score_column])
        metric_records = []
        for _, row in others.iterrows():
            other_score = float(row[score_column])
            absolute_difference = baseline_score - other_score
            relative_difference_pct = (
                absolute_difference / abs(other_score) * 100.0 if not np.isclose(other_score, 0.0) else np.nan
            )
            h1 = bool(baseline_score > other_score)
            record = {
                "scenario": scenario.name,
                "scenario_label": scenario.label,
                "baseline_model": "GARCH-Autoformer",
                "comparison_model": row["display_group"],
                "metric": label,
                "metric_column": score_column,
                "baseline_score": baseline_score,
                "comparison_score": other_score,
                "absolute_difference": absolute_difference,
                "relative_difference_pct": relative_difference_pct,
                "h1_garch_autoformer_outperforms": h1,
            }
            metric_records.append(record)
            pairwise_records.append(record)

        metric_frame = pd.DataFrame.from_records(metric_records)
        total = int(len(metric_frame))
        h1_count = int(metric_frame["h1_garch_autoformer_outperforms"].sum()) if total else 0
        h1_rate = h1_count / total if total else np.nan
        null_win_rate = 0.5
        if total:
            binomial_p_value = float(stats.binomtest(h1_count, total, null_win_rate, alternative="greater").pvalue)
            z_stat = (h1_rate - null_win_rate) / np.sqrt(null_win_rate * (1.0 - null_win_rate) / total)
            z_p_value = float(stats.norm.sf(z_stat))
        else:
            binomial_p_value = np.nan
            z_stat = np.nan
            z_p_value = np.nan
        summary_records.append(
            {
                "scenario": scenario.name,
                "scenario_label": scenario.label,
                "baseline_model": "GARCH-Autoformer",
                "metric": label,
                "metric_column": score_column,
                "baseline_score": baseline_score,
                "total_remaining_models": total,
                "h1_count": h1_count,
                "h1_rate_pct": h1_rate * 100.0 if total else np.nan,
                "null_win_rate": null_win_rate,
                "binomial_p_value_greater": binomial_p_value,
                "one_proportion_z_stat": z_stat,
                "one_proportion_z_p_value_greater": z_p_value,
                "mean_other_score": float(metric_frame["comparison_score"].mean()) if total else np.nan,
                "mean_relative_difference_pct": float(metric_frame["relative_difference_pct"].mean()) if total else np.nan,
                "note": (
                    "H1 is counted when GARCH-Autoformer score is greater than the comparison model score. "
                    "The binomial p-value is the preferred exact one-sided test against a 50% win-rate null; "
                    "the z p-value is the normal-approximation one-proportion z-test."
                ),
            }
        )

    return pd.DataFrame.from_records(summary_records), pd.DataFrame.from_records(pairwise_records)


def save_analysis_plots(
    output_dir: Path,
    scenario: McdmScenario,
    matrix: pd.DataFrame,
    saw: pd.DataFrame,
    topsis: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:
    save_bar_plot(
        saw,
        score_column="saw_score",
        rank_column="saw_rank",
        title=f"Top Models by SAW Composite Score ({scenario.label})",
        output_path=output_dir / "SAWTopModels.png",
    )
    save_bar_plot(
        topsis,
        score_column="topsis_score",
        rank_column="topsis_rank",
        title=f"Top Models by TOPSIS Closeness Coefficient ({scenario.label})",
        output_path=output_dir / "TOPSISTopModels.png",
    )
    save_accuracy_risk_plot(saw, combined, output_dir / "AccuracyRiskTradeoff.png")
    save_weight_plot(scenario.criteria, output_dir / "CriteriaWeights.png")
    save_rank_comparison_plot(combined, output_dir / "SAWTOPSISRankComparison.png")
    save_tracking_penalty_plot(matrix, combined, output_dir / "VolatilityDynamicsPenalty.png")
    save_std_ratio_error_by_model_tier_plot(matrix, output_dir / "VolatilityStdRatioErrorByModelTier.png")


def run_scenario(matrix: pd.DataFrame, scenario: McdmScenario, output_dir: Path) -> None:
    validate_weights(scenario.criteria)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_matrix, excluded = split_eligible_models(matrix, scenario.criteria)
    if eligible_matrix.empty:
        raise ValueError(f"No model has complete criteria for MCDM scoring in {scenario.name}")

    saw = saw_ranking(eligible_matrix, scenario.criteria)
    topsis = topsis_ranking(eligible_matrix, scenario.criteria)
    combined = combined_ranking(saw, topsis)
    garch_dominance_summary, garch_dominance_pairwise = garch_autoformer_dominance(combined, scenario)

    outputs = {
        "CriteriaWeights.csv": criteria_frame(scenario),
        "MCDMInputMetrics.csv": matrix,
        "ExcludedModels.csv": excluded,
        "SAWRanking.csv": saw,
        "TOPSISRanking.csv": topsis,
        "CombinedMCDMRanking.csv": combined,
        "GARCHAutoformerDominanceSummary.csv": garch_dominance_summary,
        "GARCHAutoformerPairwiseDominance.csv": garch_dominance_pairwise,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {path} ({len(frame)} rows)")

    save_analysis_plots(output_dir, scenario, matrix, saw, topsis, combined)
    print(f"Saved analysis plots in {output_dir}")


def main() -> None:
    for scenario in MCDM_SCENARIOS:
        validate_weights(scenario.criteria)

    stats_root = PROJECT_ROOT / "output" / "stats_analysis"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_root = PROJECT_ROOT / "output" / "mcdm_results" / f"MCDM{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    matrix = build_decision_matrix(stats_root)
    print(f"MCDM run root: {run_root}")

    for scenario in MCDM_SCENARIOS:
        scenario_dir = run_root / scenario.folder
        print(f"Running {scenario.name}: {scenario.description}")
        run_scenario(matrix, scenario, scenario_dir)


if __name__ == "__main__":
    main()
