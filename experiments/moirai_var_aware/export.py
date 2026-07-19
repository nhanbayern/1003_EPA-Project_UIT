from __future__ import annotations

import pandas as pd

from .config import HORIZONS


def build_prediction_frame(index_name, model_type, lambda_var, test_ds, preds, targets, tuning_mode: str = "head") -> pd.DataFrame:
    rows = []
    full_ds = test_ds.dataset
    branch = "Moirai_VAR" if tuning_mode == "head" else "Moirai_VAR_FT"
    tier = f"lambda_{lambda_var:g}" if tuning_mode == "head" else f"{tuning_mode}_lambda_{lambda_var:g}"
    for k, idx in enumerate(test_ds.indices):
        sample = full_ds.samples[idx]
        for h_idx, horizon in enumerate(HORIZONS):
            rows.append(
                {
                    "dataset": index_name,
                    "branch": branch,
                    "tier": tier,
                    "model": model_type,
                    "time": sample["time"],
                    "horizon": horizon,
                    "log_return": sample["log_return"],
                    "true_volatility": targets[k, h_idx],
                    "predict_volatility": preds[k, h_idx],
                }
            )
    return pd.DataFrame(rows)
