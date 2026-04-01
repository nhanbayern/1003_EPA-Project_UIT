
import numpy as np
from scipy import stats as scipy_stats


def rolling_std_vol(returns, window_size):
    window_size = int(max(1, window_size))
    arr = np.asarray(returns, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)

    for t in range(window_size - 1, arr.shape[0]):
        w = arr[t - window_size + 1:t + 1]
        if np.all(np.isfinite(w)):
            out[t] = np.std(w, ddof=0)

    return out


def compute_mse_qlike(realized_vol, predicted_vol):
    realized_vol = np.asarray(realized_vol, dtype=float)
    predicted_vol = np.asarray(predicted_vol, dtype=float)
    if realized_vol.size == 0 or predicted_vol.size == 0:
        return np.nan, np.nan

    realized_var = realized_vol**2
    predicted_var = np.maximum(predicted_vol**2, 1e-8)
    mse = np.mean((realized_vol - predicted_vol) ** 2)
    qlike = np.mean(np.log(predicted_var) + realized_var / predicted_var)
    return mse, qlike

def kupiec_test(violations, confidence_level=0.95):
    n = len(violations)
    n_viol = violations.sum()

    if n == 0:
        return 0.0, np.nan, np.nan

    viol_rate = n_viol / n
    expected_rate = 1 - confidence_level

    if n_viol == 0 or n_viol == n:
        lr_stat, p_value = (np.nan, np.nan)
    else:
        log_likelihood_unrestricted = (n - n_viol) * np.log(1 - viol_rate) + n_viol * np.log(viol_rate)
        log_likelihood_restricted = (n - n_viol) * np.log(1 - expected_rate) + n_viol * np.log(expected_rate)
        lr_stat = -2 * (log_likelihood_restricted - log_likelihood_unrestricted)
        p_value = 1 - scipy_stats.chi2.cdf(lr_stat, 1)

    return viol_rate, lr_stat, p_value

def evaluate_all(returns_eval, vol_forecast, nu, confidence_level=0.95, seq_len=60):
    realized_vol = np.abs(returns_eval)
    mse, qlike = compute_mse_qlike(realized_vol, vol_forecast)

    t_alpha = scipy_stats.t.ppf(confidence_level, df=nu)
    var_forecast = t_alpha * vol_forecast * np.sqrt((nu - 2) / nu)

    violations = returns_eval < -var_forecast
    viol_rate, kupiec_lr, kupiec_p = kupiec_test(violations, 1 - confidence_level)

    return {
        "MSE": mse,
        "QLIKE": qlike,
        "Violation_Rate": viol_rate,
        "Kupiec_LR": kupiec_lr,
        "Kupiec_p": kupiec_p,
    }
