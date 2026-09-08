from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats


def _clip_prob(value: float, eps: float = 1e-12) -> float:
    return float(np.clip(float(value), eps, 1.0 - eps))


@dataclass(frozen=True)
class StudentTNuEstimator:
    min_nu: float = 2.1

    def estimate(self, returns) -> float:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return float(max(4.0, self.min_nu))

        try:
            nu, _, _ = stats.t.fit(arr)
            if not np.isfinite(nu):
                return float(max(4.0, self.min_nu))
            return float(max(nu, self.min_nu))
        except Exception:
            return float(max(4.0, self.min_nu))


@dataclass(frozen=True)
class VaRMethod(ABC):
    alpha: float = 0.05
    epsilon: float = 1e-8
    mu: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def calculate(self, returns, predicted_volatility, nu: float | None = None) -> np.ndarray:
        raise NotImplementedError

    def _valid_volatility(self, predicted_volatility) -> tuple[np.ndarray, np.ndarray]:
        vol = np.asarray(predicted_volatility, dtype=float)
        mask = np.isfinite(vol) & (vol > 0)
        return vol, mask


@dataclass(frozen=True)
class NormalVaRMethod(VaRMethod):
    @property
    def name(self) -> str:
        return "normal"

    def calculate(self, returns, predicted_volatility, nu: float | None = None) -> np.ndarray:
        vol, mask = self._valid_volatility(predicted_volatility)
        out = np.full(vol.shape[0], np.nan, dtype=float)
        q_alpha = stats.norm.ppf(self.alpha)
        out[mask] = self.mu + np.maximum(vol[mask], self.epsilon) * q_alpha
        return out


@dataclass(frozen=True)
class StudentTVaRMethod(VaRMethod):
    @property
    def name(self) -> str:
        return "student_t"

    def calculate(self, returns, predicted_volatility, nu: float | None = None) -> np.ndarray:
        vol, mask = self._valid_volatility(predicted_volatility)
        out = np.full(vol.shape[0], np.nan, dtype=float)
        if nu is None or not np.isfinite(float(nu)):
            return out

        # Apply variance scale factor for standardized Student-t (M3 fix)
        scale_factor = np.sqrt((float(nu) - 2.0) / float(nu)) if float(nu) > 2.0 else 1.0
        q_alpha = stats.t.ppf(self.alpha, df=float(nu)) * scale_factor
        out[mask] = self.mu + np.maximum(vol[mask], self.epsilon) * q_alpha
        return out


@dataclass(frozen=True)
class FilteredHistoricalVaRMethod(VaRMethod):
    rolling_window: int = 250
    min_history: int = 250

    @property
    def name(self) -> str:
        return "fhs"

    def calculate(self, returns, predicted_volatility, nu: float | None = None) -> np.ndarray:
        returns_arr = np.asarray(returns, dtype=float)
        vol, vol_mask = self._valid_volatility(predicted_volatility)
        n = min(returns_arr.size, vol.size)
        returns_arr = returns_arr[:n]
        vol = vol[:n]
        vol_mask = vol_mask[:n]

        out = np.full(n, np.nan, dtype=float)
        residuals = np.full(n, np.nan, dtype=float)
        valid_residuals = np.isfinite(returns_arr) & vol_mask
        residuals[valid_residuals] = returns_arr[valid_residuals] / np.maximum(vol[valid_residuals], self.epsilon)

        # Shift by one to keep the VaR at t strictly based on past residuals.
        q_alpha = (
            pd.Series(residuals)
            .shift(1)
            .rolling(window=self.rolling_window, min_periods=self.min_history)
            .quantile(self.alpha)
            .to_numpy(dtype=float)
        )
        valid_var = vol_mask & np.isfinite(q_alpha)
        out[valid_var] = self.mu + np.maximum(vol[valid_var], self.epsilon) * q_alpha[valid_var]

        return out


@dataclass(frozen=True)
class EVTFilteredVaRMethod(VaRMethod):
    rolling_window: int = 250
    min_history: int = 250
    tail_probability: float = 0.10
    min_exceedances: int = 20

    @property
    def name(self) -> str:
        return "evt"

    def calculate(self, returns, predicted_volatility, nu: float | None = None) -> np.ndarray:
        returns_arr = np.asarray(returns, dtype=float)
        vol, vol_mask = self._valid_volatility(predicted_volatility)
        n = min(returns_arr.size, vol.size)
        returns_arr = returns_arr[:n]
        vol = vol[:n]
        vol_mask = vol_mask[:n]

        out = np.full(n, np.nan, dtype=float)
        residuals = np.full(n, np.nan, dtype=float)
        valid_residuals = np.isfinite(returns_arr) & vol_mask
        residuals[valid_residuals] = returns_arr[valid_residuals] / np.maximum(vol[valid_residuals], self.epsilon)

        for t in range(n):
            if not vol_mask[t]:
                continue

            start = max(0, t - self.rolling_window)
            history = residuals[start:t]
            history = history[np.isfinite(history)]
            if history.size < self.min_history:
                continue

            q_tail = np.quantile(history, self.tail_probability)
            tail_losses = -history[history <= q_tail]
            threshold_loss = -q_tail
            exceedances = tail_losses - threshold_loss
            exceedances = exceedances[np.isfinite(exceedances) & (exceedances > 0)]
            if exceedances.size < self.min_exceedances:
                continue

            try:
                shape, _, scale = stats.genpareto.fit(exceedances, floc=0.0)
                if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                    continue

                tail_fraction = exceedances.size / history.size
                target_tail_prob = self.alpha / tail_fraction
                if not 0 < target_tail_prob < 1:
                    continue

                excess_quantile = stats.genpareto.ppf(1.0 - target_tail_prob, c=shape, loc=0.0, scale=scale)
                if not np.isfinite(excess_quantile):
                    continue

                residual_quantile = -(threshold_loss + excess_quantile)
                out[t] = self.mu + np.maximum(vol[t], self.epsilon) * residual_quantile
            except Exception:
                continue

        return out


@dataclass(frozen=True)
class VarBacktester:
    alpha: float = 0.05
    pvalue_threshold: float = 0.05
    epsilon: float = 1e-8
    mu: float = 0.0

    def calculate(self, returns, predicted_volatility, nu: float) -> dict[str, float | int | bool]:
        returns_arr, vol_arr = self._aligned_valid_arrays(returns, predicted_volatility)
        if returns_arr.size == 0:
            return self._empty_result()

        var_threshold = self.var_threshold(vol_arr, nu)
        violations = returns_arr < var_threshold
        violation_rate, kupiec_lr, kupiec_p = self.kupiec_test(violations)
        lr_ind, lr_ind_p = self.christoffersen_independence_test(violations)

        kupiec_pass = bool(np.isfinite(kupiec_p) and kupiec_p > self.pvalue_threshold)
        independence_pass = bool(np.isfinite(lr_ind_p) and lr_ind_p > self.pvalue_threshold)

        return {
            "n_risk": int(returns_arr.size),
            "nu": float(nu),
            "violation_count": int(np.sum(violations)),
            "violation_rate": violation_rate,
            "kupiec_lr": kupiec_lr,
            "kupiec_p": kupiec_p,
            "lr_ind": lr_ind,
            "lr_ind_p": lr_ind_p,
            "kupiec_pass": kupiec_pass,
            "independence_pass": independence_pass,
            "backtest_pass": bool(kupiec_pass and independence_pass),
        }

    def var_threshold(self, predicted_volatility: np.ndarray, nu: float) -> np.ndarray:
        vol = np.maximum(np.asarray(predicted_volatility, dtype=float), self.epsilon)
        q_alpha = stats.t.ppf(self.alpha, df=float(nu))
        return self.mu + vol * q_alpha

    def kupiec_test(self, violations) -> tuple[float, float, float]:
        v = np.asarray(violations, dtype=bool)
        n = int(v.size)
        if n == 0:
            return float("nan"), float("nan"), float("nan")

        x = int(v.sum())
        violation_rate = x / n
        p_hat = _clip_prob(violation_rate)
        p_exp = _clip_prob(self.alpha)

        log_l_restricted = (n - x) * np.log(1.0 - p_exp) + x * np.log(p_exp)
        log_l_unrestricted = (n - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        lr_stat = max(0.0, -2.0 * (log_l_restricted - log_l_unrestricted))
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, 1))
        return float(violation_rate), float(lr_stat), p_value

    def christoffersen_independence_test(self, violations) -> tuple[float, float]:
        v = np.asarray(violations, dtype=bool).astype(int)
        if v.size < 2:
            return float("nan"), float("nan")

        n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
        n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
        n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
        n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

        pseudo = 0.5
        p01 = _clip_prob((n01 + pseudo) / (n00 + n01 + 2.0 * pseudo))
        p11 = _clip_prob((n11 + pseudo) / (n10 + n11 + 2.0 * pseudo))
        p = _clip_prob((n01 + n11 + pseudo) / (n00 + n01 + n10 + n11 + 2.0 * pseudo))

        log_l_unrestricted = (
            n00 * np.log(1.0 - p01)
            + n01 * np.log(p01)
            + n10 * np.log(1.0 - p11)
            + n11 * np.log(p11)
        )
        log_l_restricted = (n00 + n10) * np.log(1.0 - p) + (n01 + n11) * np.log(p)

        lr_stat = max(0.0, -2.0 * (log_l_restricted - log_l_unrestricted))
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, 1))
        return float(lr_stat), p_value

    def _aligned_valid_arrays(self, returns, predicted_volatility) -> tuple[np.ndarray, np.ndarray]:
        returns_arr = np.asarray(returns, dtype=float)
        vol_arr = np.asarray(predicted_volatility, dtype=float)
        n = min(returns_arr.size, vol_arr.size)
        if n == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        returns_arr = returns_arr[:n]
        vol_arr = vol_arr[:n]
        mask = np.isfinite(returns_arr) & np.isfinite(vol_arr) & (vol_arr > 0)
        return returns_arr[mask], vol_arr[mask]

    def _empty_result(self) -> dict[str, float | int | bool]:
        return {
            "n_risk": 0,
            "nu": float("nan"),
            "violation_count": 0,
            "violation_rate": float("nan"),
            "kupiec_lr": float("nan"),
            "kupiec_p": float("nan"),
            "lr_ind": float("nan"),
            "lr_ind_p": float("nan"),
            "kupiec_pass": False,
            "independence_pass": False,
            "backtest_pass": False,
        }
