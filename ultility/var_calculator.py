

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RollingVaRCalculator:
  
    
    def __init__(self, confidence_level=0.95, min_history=30, refit_frequency=1):
 
        self.confidence_level = confidence_level
        self.min_history = min_history
        self.refit_frequency = refit_frequency
        self.alpha = 1 - confidence_level
        
    def _fit_nu_robust(self, residuals):
    
        if len(residuals) < self.min_history:
            return 4.0  # Default when insufficient data
        
        try:
            # Fit Student-t distribution
            nu, loc, scale = stats.t.fit(residuals)
            
            # Bound nu to avoid extreme values
            nu = np.clip(nu, 2.5, 30.0)
            
            return nu
        except Exception:
            return 4.0
    
    def compute_var_expanding_window(self, returns, volatilities):
        T = len(returns)
        var_estimates = np.full(T, np.nan)

        for t in range(self.min_history, T):
            # 🔥 KHÔNG dùng residual ML
            past_returns = returns[:t]

            # normalize bằng rolling std (robust hơn)
            std = np.std(past_returns)
            z = past_returns / (std + 1e-8)

            # fit Student-t trên distribution thật
            nu, _, _ = stats.t.fit(z)
            nu = np.clip(nu, 2.5, 10.0)

            # quantile đúng (LEFT tail)
            q = stats.t.ppf(self.alpha, df=nu)  # âm

            # VaR
            var_estimates[t] = volatilities[t] * abs(q)

        return var_estimates
    
    def compute_var_rolling_window(self, returns, volatilities, 
                                   window_size=60, confidence_level=None):

        if confidence_level is None:
            confidence_level = self.confidence_level

        T = len(returns)
        var_estimates = np.full(T, np.nan)

        # Compute standardized residuals
        residuals = returns / np.maximum(volatilities, 1e-8)

        # Rolling window VaR (Filtered Historical Simulation)
        for t in range(window_size, T):
            # Use rolling window [t-window_size, t-1] (NO LEAKAGE)
            window_start = max(0, t - window_size)
            window_residuals = residuals[window_start:t]

            # Empirical left-tail quantile of residuals
            q = np.quantile(window_residuals, self.alpha)

            # VaR_t as positive magnitude
            var_estimates[t] = -volatilities[t] * q

        return var_estimates


def compute_kupiec_test(returns, var_estimates, confidence_level=0.95):

    # Mask out NaNs that occur before min_history
    valid_mask = (~np.isnan(returns)) & (~np.isnan(var_estimates))
    returns_valid = returns[valid_mask]
    var_valid = var_estimates[valid_mask]

    if returns_valid.size == 0:
        return np.nan, np.nan, np.nan

    # A violation occurs when loss exceeds VaR (negative returns)
    violations = returns_valid < -var_valid

    n = len(violations)
    n_violations = violations.sum()

    expected_violations = (1 - confidence_level) * n

    violation_rate = n_violations / n if n > 0 else np.nan

    # Handle edge cases
    if n == 0 or n_violations == 0 or n_violations == n:
        return violation_rate, np.nan, np.nan

    # Likelihood ratio test
    p_observed = n_violations / n
    p_expected = 1 - confidence_level

    lr = -2 * (
        (n - n_violations) * np.log(1 - p_expected) + 
        n_violations * np.log(p_expected) - 
        (n - n_violations) * np.log(1 - p_observed) - 
        n_violations * np.log(p_observed)
    )

    kupiec_p = 1 - stats.chi2.cdf(lr, 1)

    return violation_rate, lr, kupiec_p


def compute_traffic_light_test(returns, var_estimates, confidence_level=0.95, 
                               window_size=250):
 
    # Mask NaNs
    valid_mask = (~np.isnan(returns)) & (~np.isnan(var_estimates))
    returns_valid = returns[valid_mask]
    var_valid = var_estimates[valid_mask]

    if returns_valid.size == 0:
        return 'Red', np.nan

    violations = (returns_valid < -var_valid).astype(int)
    
    # Get last window_size violations
    recent_violations = violations[-window_size:]
    cumulative_violations = recent_violations.sum()
    
    # Expected violations
    expected = (1 - confidence_level) * min(window_size, len(violations))
    
    # Traffic light thresholds (Basel III)
    if cumulative_violations <= expected * 1.0:  # Up to expected
        traffic_light = 'Green'
    elif cumulative_violations <= expected * 1.5:
        traffic_light = 'Yellow'
    else:
        traffic_light = 'Red'
    
    return traffic_light, cumulative_violations


def compute_christoffersen_independence(returns, var_estimates, confidence_level=0.95):
    valid_mask = (~np.isnan(returns)) & (~np.isnan(var_estimates))
    returns_valid = returns[valid_mask]
    var_valid = var_estimates[valid_mask]
    if returns_valid.size < 2:
        return np.nan, np.nan
    violations = (returns_valid < -var_valid).astype(int)
    n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
    n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
    n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
    n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
    n0 = n00 + n01
    n1 = n10 + n11
    if n0 == 0 or n1 == 0:
        return np.nan, np.nan
    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / (n0 + n1)
    log_l1 = 0
    if p01 > 0:
        log_l1 += n01 * np.log(p01)
    if p01 < 1:
        log_l1 += n00 * np.log(1 - p01)
    if p11 > 0:
        log_l1 += n11 * np.log(p11)
    if p11 < 1:
        log_l1 += n10 * np.log(1 - p11)
    log_l0 = 0
    if p > 0:
        log_l0 += (n01 + n11) * np.log(p)
    if p < 1:
        log_l0 += (n00 + n10) * np.log(1 - p)
    lr_ind = -2 * (log_l0 - log_l1)
    p_ind = 1 - stats.chi2.cdf(lr_ind, 1)
    return lr_ind, p_ind
