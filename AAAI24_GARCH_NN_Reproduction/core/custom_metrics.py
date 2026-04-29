from __future__ import annotations

import numpy as np
from scipy import stats


def _to_numpy(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=float)
    return arr


def _clip_prob(prob, eps=1e-12):
    return float(np.clip(float(prob), eps, 1.0 - eps))


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


def compute_qlike(y_true, y_pred):
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    n = min(len(true), len(pred))
    if n == 0:
        return np.nan

    true_var = np.maximum(true[:n] ** 2, 1e-8)
    pred_var = np.maximum(pred[:n] ** 2, 1e-8)
    return float(np.mean(np.log(pred_var) + true_var / pred_var))


def compute_kupiec_test(violations, confidence_level=0.95):
    violations = np.asarray(violations, dtype=bool)
    n = int(violations.size)
    if n == 0:
        return np.nan, np.nan, np.nan

    n_viol = int(violations.sum())
    violation_rate = n_viol / n
    expected_rate = 1.0 - confidence_level

    p_hat = _clip_prob(violation_rate)
    p_exp = _clip_prob(expected_rate)

    log_l_unrestricted = (n - n_viol) * np.log(1 - p_hat) + n_viol * np.log(p_hat)
    log_l_restricted = (n - n_viol) * np.log(1 - p_exp) + n_viol * np.log(p_exp)
    lr_stat = max(0.0, -2.0 * (log_l_restricted - log_l_unrestricted))
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, 1))
    return float(violation_rate), float(lr_stat), p_value


def compute_christoffersen_independence(violations):
    violations = np.asarray(violations, dtype=bool).astype(int)
    if violations.size < 2:
        return np.nan, np.nan

    n00 = int(np.sum((violations[:-1] == 0) & (violations[1:] == 0)))
    n01 = int(np.sum((violations[:-1] == 0) & (violations[1:] == 1)))
    n10 = int(np.sum((violations[:-1] == 1) & (violations[1:] == 0)))
    n11 = int(np.sum((violations[:-1] == 1) & (violations[1:] == 1)))

    # Jeffreys smoothing keeps LR_Ind finite in degenerate cases (all-0 or all-1).
    pseudo = 0.5
    n00_s = n00 + pseudo
    n01_s = n01 + pseudo
    n10_s = n10 + pseudo
    n11_s = n11 + pseudo

    p01 = _clip_prob(n01_s / (n00_s + n01_s))
    p11 = _clip_prob(n11_s / (n10_s + n11_s))
    p = _clip_prob((n01_s + n11_s) / (n00_s + n01_s + n10_s + n11_s))

    log_l1 = (
        n00 * np.log(1 - p01)
        + n01 * np.log(p01)
        + n10 * np.log(1 - p11)
        + n11 * np.log(p11)
    )

    log_l0 = (
        (n00 + n10) * np.log(1 - p)
        + (n01 + n11) * np.log(p)
    )

    lr_ind = max(0.0, -2.0 * (log_l0 - log_l1))
    p_ind = float(1.0 - stats.chi2.cdf(lr_ind, 1))
    return float(lr_ind), p_ind


def compute_var_metrics(returns_eval, predicted_vol, nu, confidence_level=0.95):
    returns_eval = _to_numpy(returns_eval)
    predicted_vol = _to_numpy(predicted_vol)
    n = min(len(returns_eval), len(predicted_vol))
    if n == 0:
        return {
            "Violation_Rate": np.nan,
            "Kupiec_LR": np.nan,
            "Kupiec_p": np.nan,
            "LR_Ind": np.nan,
        }

    returns_eval = returns_eval[:n]
    predicted_vol = np.maximum(predicted_vol[:n], 1e-8)
    t_alpha = stats.t.ppf(confidence_level, df=float(nu))
    var_forecast = t_alpha * predicted_vol * np.sqrt((float(nu) - 2.0) / float(nu))
    violations = returns_eval < -var_forecast

    violation_rate, kupiec_lr, kupiec_p = compute_kupiec_test(violations, confidence_level=confidence_level)
    lr_ind, _ = compute_christoffersen_independence(violations)
    return {
        "Violation_Rate": violation_rate,
        "Kupiec_LR": kupiec_lr,
        "Kupiec_p": kupiec_p,
        "LR_Ind": lr_ind,
    }


def compute_metrics(y_true, y_pred, returns_eval=None, nu=None, confidence_level=0.95):
    metrics = {
        "MAE": compute_mae(y_true, y_pred),
        "MSE": compute_mse(y_true, y_pred),
        "QLIKE": compute_qlike(y_true, y_pred),
        "Violation_Rate": np.nan,
        "Kupiec_LR": np.nan,
        "Kupiec_p": np.nan,
        "LR_Ind": np.nan,
    }

    if returns_eval is not None and nu is not None:
        metrics.update(
            compute_var_metrics(
                returns_eval=returns_eval,
                predicted_vol=y_pred,
                nu=nu,
                confidence_level=confidence_level,
            )
        )

    return metrics
