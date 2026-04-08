from __future__ import annotations

import numpy as np


def _to_numpy(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=float)
    return arr


def compute_mse(y_true, y_pred):
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    n = min(len(true), len(pred))
    if n == 0:
        return np.nan
    return float(np.mean((true[:n] - pred[:n]) ** 2))


def compute_mae(y_true, y_pred):
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    n = min(len(true), len(pred))
    if n == 0:
        return np.nan
    return float(np.mean(np.abs(true[:n] - pred[:n])))


def compute_metrics(y_true, y_pred):
    return {
        "MAE": compute_mae(y_true, y_pred),
        "MSE": compute_mse(y_true, y_pred),
    }
