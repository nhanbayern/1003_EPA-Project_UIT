"""Evaluate lambda sweep against a validation-tuned scalar-rescaling control.

This addresses Reviewer 1's Major Weakness 1 & 5:
1. Validates the null hypothesis: post-hoc scalar rescaling (sigma_hat * c)
   where c is tuned on validation returns to minimize VaR pinball loss.
2. Traces both the continuous rescaling frontier (c-frontier) and the trained
   lambda-frontier on the accuracy--tail risk plane (MSE vs VaR pinball loss).
3. Uses the mathematically correct standardized Student-t quantile:
   q_{alpha, nu} = t_{alpha, nu} * sqrt((nu - 2) / nu)
   and computes pinball loss on realized returns r_{t+1} vs VaR threshold.

Usage:
  python evaluate_lambda_sweep.py --input-dir output/modal_moirai_var_loss --output-dir output/lambda_ablation
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_NU_BY_DATASET: dict[str, float] = {
    "dax40": 6.279,
    "euronext100": 5.650,
    "ibex35": 4.864,
    "kospiindex": 5.228,
    "nikkei225": 5.249,
    "smi": 4.896,
    "snp500": 5.258,
    "sp500": 5.258,
    "vn30index": 6.788,
    "vnindex": 6.848,
}


def _norm_key(name: object) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def get_standardized_t_quantile(nu: float, alpha: float = 0.01) -> float:
    """Return the standardized Student-t alpha-quantile.
    
    For a variable standardized to variance 1, the quantile is:
      q_alpha = t_alpha * sqrt((nu - 2) / nu) < 0
    """
    if nu <= 2.0:
        raise ValueError(f"Student-t degrees of freedom must be > 2, got {nu}")
    scale_factor = math.sqrt((nu - 2.0) / nu)
    return float(stats.t.ppf(alpha, df=nu)) * scale_factor


def var_pinball_loss(returns: np.ndarray, var_threshold: np.ndarray, alpha: float = 0.01) -> float:
    """Compute asymmetric tick / pinball loss on realized return vs VaR threshold.
    
    u_t = r_t - VaR_t
    loss = (alpha - 1{r_t < VaR_t}) * (r_t - VaR_t) >= 0
    """
    indicator = (returns < var_threshold).astype(float)
    return float(np.mean((alpha - indicator) * (returns - var_threshold)))


def empirical_violation_rate(returns: np.ndarray, var_threshold: np.ndarray) -> float:
    """Fraction of returns breaching the Value-at-Risk threshold."""
    return float(np.mean(returns < var_threshold))


def evaluate_lambda_sweep(input_dir: Path, output_dir: Path, alpha: float = 0.01) -> None:
    """Evaluate lambda sweep and compare against scalar-rescaling control."""
    files = sorted(input_dir.rglob("*_predictions.csv"))
    frames = [pd.read_csv(f) for f in files if "validation" in f.name or "_lambda_" in f.name]
    if not frames:
        raise FileNotFoundError(f"No lambda prediction CSVs found in {input_dir}")

    df = pd.concat(frames, ignore_index=True)
    required = {"dataset", "model", "lambda_var", "horizon", "log_return", "true_volatility", "predict_volatility", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}; ensure prediction frames carry {required}")

    df = df[np.isfinite(df["predict_volatility"]) & np.isfinite(df["true_volatility"]) & np.isfinite(df["log_return"])].copy()

    records = []
    frontier_curves = []
    group_cols = ["dataset", "model", "horizon"]

    for key, group in df.groupby(group_cols, sort=True):
        val = group[group["split"] == "validation"]
        test = group[group["split"] == "test"]

        base_val = val[val["lambda_var"] == 0.0]
        base_test = test[test["lambda_var"] == 0.0]
        if base_val.empty or base_test.empty:
            continue

        dataset_name = str(group["dataset"].iloc[0])
        if "nu" in group.columns and pd.notna(group["nu"].iloc[0]) and group["nu"].iloc[0] > 2.0:
            nu = float(group["nu"].iloc[0])
        else:
            nu = DEFAULT_NU_BY_DATASET.get(_norm_key(dataset_name), 5.0)

        q_alpha = get_standardized_t_quantile(nu, alpha)

        r_val = base_val["log_return"].to_numpy(float)
        p_val = base_val["predict_volatility"].to_numpy(float)

        # 1. Validation tuning for scalar factor c: c * sigma_hat
        c_grid = np.linspace(0.4, 3.0, 521)
        val_losses = [var_pinball_loss(r_val, c * p_val * q_alpha, alpha) for c in c_grid]
        best_c = float(c_grid[np.argmin(val_losses)])

        r_test_base = base_test["log_return"].to_numpy(float)
        y_test_base = base_test["true_volatility"].to_numpy(float)
        p_test_base = base_test["predict_volatility"].to_numpy(float)

        # 2. Rescaled baseline on test set
        rescaled_var_test = best_c * p_test_base * q_alpha
        records.append({
            **dict(zip(group_cols, key)),
            "method": "rescaled_lambda_0",
            "lambda_var": np.nan,
            "scale_factor": best_c,
            "nu": nu,
            "mse": float(np.mean((y_test_base - best_c * p_test_base) ** 2)),
            "pinball": var_pinball_loss(r_test_base, rescaled_var_test, alpha),
            "violation_rate": empirical_violation_rate(r_test_base, rescaled_var_test),
        })

        # 3. Trained models across lambda values on test set
        for lam, part in test.groupby("lambda_var"):
            r_t = part["log_return"].to_numpy(float)
            y_t = part["true_volatility"].to_numpy(float)
            p_t = part["predict_volatility"].to_numpy(float)
            trained_var_test = p_t * q_alpha
            records.append({
                **dict(zip(group_cols, key)),
                "method": "trained_lambda",
                "lambda_var": float(lam),
                "scale_factor": 1.0,
                "nu": nu,
                "mse": float(np.mean((y_t - p_t) ** 2)),
                "pinball": var_pinball_loss(r_t, trained_var_test, alpha),
                "violation_rate": empirical_violation_rate(r_t, trained_var_test),
            })

        # 4. Continuous c-frontier on test set (for Pareto trade-off curve comparison)
        c_curve_grid = np.linspace(0.6, 2.4, 73)
        for c in c_curve_grid:
            c_var = c * p_test_base * q_alpha
            frontier_curves.append({
                **dict(zip(group_cols, key)),
                "curve_type": "c_rescaling",
                "param_value": float(c),
                "mse": float(np.mean((y_test_base - c * p_test_base) ** 2)),
                "pinball": var_pinball_loss(r_test_base, c_var, alpha),
                "violation_rate": empirical_violation_rate(r_test_base, c_var),
            })

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No matching validation/test lambda=0 groups found to compute scalar rescaling")

    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "lambda_vs_rescaling_by_case.csv", index=False)

    summary = (
        out.groupby(["method", "lambda_var"], dropna=False)
        .agg(
            cases=("mse", "size"),
            mean_scale=("scale_factor", "mean"),
            mse=("mse", "mean"),
            pinball=("pinball", "mean"),
            violation_rate=("violation_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "lambda_vs_rescaling_summary.csv", index=False)

    # 5. Publication figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: Test Pinball Loss & Violation Rate across Lambda vs Rescaled Control
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    trained = summary[summary["method"] == "trained_lambda"].sort_values("lambda_var")
    control = summary[summary["method"] == "rescaled_lambda_0"].iloc[0]

    ax1.plot(trained["lambda_var"], trained["pinball"], "o-", color="#1f77b4", linewidth=2, markersize=6, label=r"Trained $\lambda$-objective")
    ax1.axhline(control["pinball"], color="#d62728", linestyle="--", linewidth=1.8, label=f"Rescaled $\lambda=0$ ($c^*={control['mean_scale']:.2f}$)")
    ax1.set_xlabel(r"Risk penalty weight $\lambda_{\mathrm{VaR}}$", fontsize=11)
    ax1.set_ylabel(f"Test VaR {int(alpha*100)}% Pinball Loss", fontsize=11)
    ax1.set_title("Tail Risk Quantile Loss vs Rescaling Control", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=True)

    ax2.plot(trained["lambda_var"], trained["violation_rate"] * 100.0, "s-", color="#2ca02c", linewidth=2, markersize=6, label=r"Trained $\lambda$-objective")
    ax2.axhline(control["violation_rate"] * 100.0, color="#d62728", linestyle="--", linewidth=1.8, label=f"Rescaled $\lambda=0$ ($c^*={control['mean_scale']:.2f}$)")
    ax2.axhline(alpha * 100.0, color="black", linestyle=":", linewidth=1.5, label=f"Nominal $\\alpha={int(alpha*100)}\\%$")
    ax2.set_xlabel(r"Risk penalty weight $\lambda_{\mathrm{VaR}}$", fontsize=11)
    ax2.set_ylabel("Empirical Violation Rate (%)", fontsize=11)
    ax2.set_title("Calibration Violation Rate vs Nominal Rate", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "lambda_vs_rescaling_pinball.png", dpi=300)
    plt.close(fig)

    # Figure 2: The Core Reviewer 1 Accuracy--Risk Frontier (MSE vs VaR Loss)
    if frontier_curves:
        df_curves = pd.DataFrame(frontier_curves)
        c_frontier = df_curves.groupby("param_value").agg(mse=("mse", "mean"), pinball=("pinball", "mean")).reset_index().sort_values("mse")

        fig2, ax = plt.subplots(figsize=(8, 5.5))
        # Plot continuous c-frontier curve
        ax.plot(c_frontier["mse"], c_frontier["pinball"], "--", color="#7f7f7f", linewidth=2.0, label=r"Scalar Rescaling Frontier ($c \times \hat{\sigma}_{\lambda=0}$)")
        # Plot discrete lambda points
        ax.plot(trained["mse"], trained["pinball"], "o-", color="#1f77b4", linewidth=2.2, markersize=8, label=r"MoiraiVaR Frontier ($\lambda \in \{0, 0.05, 0.1, 0.2, 0.5, 1\}$)")

        # Annotate each lambda point
        for _, row in trained.iterrows():
            ax.annotate(
                f"$\\lambda={row['lambda_var']:g}$",
                (row["mse"], row["pinball"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                fontweight="medium",
            )

        # Annotate optimal rescaled point c*
        ax.plot(control["mse"], control["pinball"], "*", color="#d62728", markersize=13, label=f"Validation-tuned $c^*={control['mean_scale']:.2f}$")

        ax.set_xlabel("Multi-Horizon Volatility MSE (lower is better)", fontsize=11)
        ax.set_ylabel(f"Test VaR {int(alpha*100)}% Pinball Loss (lower is better)", fontsize=11)
        ax.set_title(r"Accuracy--Risk Trade-off: $\lambda$-Frontier vs $c$-Rescaling Frontier", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(frameon=True, loc="upper right")
        fig2.tight_layout()
        fig2.savefig(output_dir / "accuracy_risk_frontier_lambda_vs_c.png", dpi=300)
        plt.close(fig2)

        # 6. Per-model Frontier Plots (Reviewer 1 explicitly requested single-backbone frontier, e.g. Moirai 2)
        summary_by_model = (
            out.groupby(["model", "method", "lambda_var"], dropna=False)
            .agg(
                cases=("mse", "size"),
                mean_scale=("scale_factor", "mean"),
                mse=("mse", "mean"),
                pinball=("pinball", "mean"),
                violation_rate=("violation_rate", "mean"),
            )
            .reset_index()
        )
        summary_by_model.to_csv(output_dir / "lambda_vs_rescaling_by_model.csv", index=False)

        for model_name in sorted(out["model"].unique()):
            sub_curves = df_curves[df_curves["model"] == model_name]
            sub_trained = summary_by_model[(summary_by_model["model"] == model_name) & (summary_by_model["method"] == "trained_lambda")].sort_values("lambda_var")
            sub_ctrl_rows = summary_by_model[(summary_by_model["model"] == model_name) & (summary_by_model["method"] == "rescaled_lambda_0")]
            if sub_curves.empty or sub_trained.empty or sub_ctrl_rows.empty:
                continue

            sub_ctrl = sub_ctrl_rows.iloc[0]
            sub_c_frontier = sub_curves.groupby("param_value").agg(mse=("mse", "mean"), pinball=("pinball", "mean")).reset_index().sort_values("mse")

            fig_m, ax_m = plt.subplots(figsize=(8, 5.5))
            ax_m.plot(sub_c_frontier["mse"], sub_c_frontier["pinball"], "--", color="#7f7f7f", linewidth=2.0, label=r"Scalar Rescaling Frontier ($c \times \hat{\sigma}_{\lambda=0}$)")
            ax_m.plot(sub_trained["mse"], sub_trained["pinball"], "o-", color="#1f77b4", linewidth=2.2, markersize=8, label=rf"MoiraiVaR {model_name} Frontier")

            for _, row in sub_trained.iterrows():
                ax_m.annotate(
                    f"$\\lambda={row['lambda_var']:g}$",
                    (row["mse"], row["pinball"]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="medium",
                )

            ax_m.plot(sub_ctrl["mse"], sub_ctrl["pinball"], "*", color="#d62728", markersize=13, label=f"Validation-tuned $c^*={sub_ctrl['mean_scale']:.2f}$")
            ax_m.set_xlabel("Multi-Horizon Volatility MSE (lower is better)", fontsize=11)
            ax_m.set_ylabel(f"Test VaR {int(alpha*100)}% Pinball Loss (lower is better)", fontsize=11)
            ax_m.set_title(rf"Accuracy--Risk Frontier: {model_name} ($\lambda$-Frontier vs $c$-Rescaling)", fontsize=12, fontweight="bold")
            ax_m.grid(alpha=0.3)
            ax_m.legend(frameon=True, loc="upper right")
            fig_m.tight_layout()
            
            fig_m.savefig(output_dir / f"accuracy_risk_frontier_{model_name}.png", dpi=300)
            if model_name == "moirai2":
                iceba_fig = Path("ICEBA-paper/fig_lambda_vs_rescaling_frontier.png")
                iceba_fig.parent.mkdir(parents=True, exist_ok=True)
                fig_m.savefig(iceba_fig, dpi=300)
            plt.close(fig_m)

    print(f"Successfully generated lambda ablation results in: {output_dir}")
    print(summary.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.01)
    args = p.parse_args()
    evaluate_lambda_sweep(args.input_dir, args.output_dir, args.alpha)


if __name__ == "__main__":
    main()
