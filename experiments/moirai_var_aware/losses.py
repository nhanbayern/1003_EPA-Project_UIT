from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


def _normal_ppf(alpha: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    ).icdf(torch.tensor(alpha, device=device, dtype=dtype))


def _student_t_ppf(alpha: float, nu: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from scipy import stats

    if nu <= 2.0:
        raise ValueError("Student-t standard deviation requires nu > 2")
    scale_factor = math.sqrt((nu - 2.0) / nu)
    return torch.tensor(float(stats.t.ppf(alpha, df=nu)) * scale_factor, device=device, dtype=dtype)


@dataclass(frozen=True)
class QuantileVaRLossConfig:
    alpha: float = 0.01
    distribution: str = "student_t"
    nu: float = 4.0
    mu: float = 0.0
    epsilon: float = 1e-8


class QuantileVaRLoss(nn.Module):
    """Quantile/tick loss for VaR inferred from predicted volatility."""

    def __init__(
        self,
        alpha: float = 0.01,
        distribution: str = "student_t",
        nu: float = 4.0,
        mu: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if distribution not in {"normal", "student_t"}:
            raise ValueError("distribution must be 'normal' or 'student_t'")
        self.config = QuantileVaRLossConfig(alpha, distribution, nu, mu, epsilon)

    def forward(self, predicted_volatility: torch.Tensor, realized_return: torch.Tensor) -> torch.Tensor:
        vol = torch.clamp(predicted_volatility, min=self.config.epsilon)
        returns = realized_return.to(device=vol.device, dtype=vol.dtype)
        if self.config.distribution == "normal":
            q_alpha = _normal_ppf(self.config.alpha, device=vol.device, dtype=vol.dtype)
        else:
            q_alpha = _student_t_ppf(self.config.alpha, self.config.nu, device=vol.device, dtype=vol.dtype)
        var_threshold = self.config.mu + vol * q_alpha
        indicator = (returns < var_threshold).to(vol.dtype)
        return ((self.config.alpha - indicator) * (returns - var_threshold)).mean()


class VarAwareVolatilityLoss(nn.Module):
    """Joint MSE volatility loss and VaR quantile loss."""

    def __init__(
        self,
        alpha: float = 0.01,
        lambda_var: float = 0.2,
        distribution: str = "student_t",
        nu: float = 4.0,
        mu: float = 0.0,
        var_horizon_index: int = 0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if lambda_var < 0.0:
            raise ValueError("lambda_var must be non-negative")
        self.lambda_var = float(lambda_var)
        self.var_horizon_index = int(var_horizon_index)
        self.volatility_loss = nn.MSELoss()
        self.var_loss = QuantileVaRLoss(alpha, distribution, nu, mu, epsilon)

    def forward(self, predicted_volatility, true_volatility, realized_return):
        vol_loss = self.volatility_loss(predicted_volatility, true_volatility)
        q_loss = self.var_loss(predicted_volatility[:, self.var_horizon_index], realized_return)
        total = vol_loss + self.lambda_var * q_loss
        return total, {
            "total_loss": total.detach(),
            "volatility_loss": vol_loss.detach(),
            "var_quantile_loss": q_loss.detach(),
        }
