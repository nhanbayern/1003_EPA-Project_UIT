from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=float)
    return arr


@dataclass(frozen=True)
class ForecastMetrics:
    epsilon: float = 1e-8

    def mse(self, y_true, y_pred) -> float:
        true, pred = self._aligned_valid_arrays(y_true, y_pred)
        if true.size == 0:
            return float("nan")
        return float(np.mean((true - pred) ** 2))

    def mae(self, y_true, y_pred) -> float:
        true, pred = self._aligned_valid_arrays(y_true, y_pred)
        if true.size == 0:
            return float("nan")
        return float(np.mean(np.abs(true - pred)))

    def qlike(self, y_true, y_pred) -> float:
        true, pred = self._aligned_valid_arrays(y_true, y_pred)
        if true.size == 0:
            return float("nan")

        true_safe = np.maximum(true, self.epsilon)
        pred_safe = np.maximum(pred, self.epsilon)
        ratio = true_safe / pred_safe
        return float(np.mean(ratio - np.log(ratio) - 1.0))

    def calculate(self, y_true, y_pred) -> dict[str, float | int]:
        true, pred = self._aligned_valid_arrays(y_true, y_pred)
        return {
            "n_forecast": int(true.size),
            "mse": self.mse(true, pred),
            "mae": self.mae(true, pred),
            "qlike": self.qlike(true, pred),
        }

    def _aligned_valid_arrays(self, y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
        true = _to_float_array(y_true)
        pred = _to_float_array(y_pred)
        n = min(true.size, pred.size)
        if n == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        true = true[:n]
        pred = pred[:n]
        mask = np.isfinite(true) & np.isfinite(pred) & (true > 0) & (pred > 0)
        return true[mask], pred[mask]

