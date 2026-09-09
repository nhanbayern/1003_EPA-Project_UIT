"""Evaluate lambda sweep and a validation-tuned scalar-rescaling control.

Usage: python evaluate_lambda_sweep.py --input-dir output/modal_moirai_var_loss --output-dir output/lambda_ablation
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pinball(y: np.ndarray, q: np.ndarray, alpha: float = .01) -> float:
    e = y - q
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1) * e)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=.01)
    args = p.parse_args()
    files = sorted(args.input_dir.rglob("*_predictions.csv"))
    frames = [pd.read_csv(f) for f in files if "validation" in f.name or "_lambda_" in f.name]
    if not frames:
        raise FileNotFoundError("No lambda prediction CSVs found")
    df = pd.concat(frames, ignore_index=True)
    required = {"dataset", "model", "lambda_var", "horizon", "true_volatility", "predict_volatility", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}; rerun the updated Modal runner")
    df = df[np.isfinite(df["predict_volatility"]) & np.isfinite(df["true_volatility"])].copy()
    records = []
    group_cols = ["dataset", "model", "horizon"]
    for key, group in df.groupby(group_cols, sort=True):
        val = group[group["split"] == "validation"]
        test = group[group["split"] == "test"]
        base_val = val[val.lambda_var == 0]
        base_test = test[test.lambda_var == 0]
        if base_val.empty or base_test.empty:
            continue
        c_grid = np.linspace(.1, 3.0, 291)
        yv = base_val.true_volatility.to_numpy(float)
        pv = base_val.predict_volatility.to_numpy(float)
        c = float(c_grid[np.argmin([pinball(yv, factor * pv, args.alpha) for factor in c_grid])])
        yt = base_test.true_volatility.to_numpy(float)
        pt = base_test.predict_volatility.to_numpy(float)
        records.append({**dict(zip(group_cols, key)), "method": "rescaled_lambda_0", "lambda_var": np.nan,
                        "scale_factor": c, "mse": float(np.mean((yt - c * pt) ** 2)),
                        "pinball": pinball(yt, c * pt, args.alpha)})
        for lam, part in test.groupby("lambda_var"):
            y = part.true_volatility.to_numpy(float); pred = part.predict_volatility.to_numpy(float)
            records.append({**dict(zip(group_cols, key)), "method": "trained_lambda", "lambda_var": float(lam),
                            "scale_factor": 1.0, "mse": float(np.mean((y - pred) ** 2)),
                            "pinball": pinball(y, pred, args.alpha)})
    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No matching validation/test lambda=0 groups")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out.to_csv(args.output_dir / "lambda_vs_rescaling_by_case.csv", index=False)
    summary = out.groupby(["method", "lambda_var"], dropna=False).agg(cases=("mse", "size"), mse=("mse", "mean"), pinball=("pinball", "mean")).reset_index()
    summary.to_csv(args.output_dir / "lambda_vs_rescaling_summary.csv", index=False)
    import matplotlib.pyplot as plt
    plot = summary[summary.method == "trained_lambda"].sort_values("lambda_var")
    plt.figure(figsize=(7, 4)); plt.plot(plot.lambda_var, plot.pinball, "o-", label="trained lambda")
    control = summary[summary.method == "rescaled_lambda_0"]
    plt.axhline(control.pinball.iloc[0], color="black", linestyle="--", label="rescaled lambda=0")
    plt.xlabel("lambda_var"); plt.ylabel("test pinball loss"); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "lambda_vs_rescaling_pinball.png", dpi=180); plt.close()


if __name__ == "__main__":
    main()
