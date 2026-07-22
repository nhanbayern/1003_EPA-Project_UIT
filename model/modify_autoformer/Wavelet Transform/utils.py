import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis

def calculate_fixed_nu(returns):
    """
    Calculate fixed degrees of freedom (nu) for Student-t distribution
    based on excess kurtosis of the training data.
    """
    k = kurtosis(returns, fisher=True, nan_policy='omit')
    if k <= 0:
        return 30.0  # Near-normal distribution
    nu = 4.0 + 6.0 / k
    return float(np.clip(nu, 2.1, 30.0))

class StudentTNLLLoss(nn.Module):
    def __init__(self, nu):
        """
        nu: fixed degrees of freedom for Student-t distribution
        """
        super().__init__()
        self.nu = nu

    def forward(self, pred_sigma, target_returns):
        """
        pred_sigma: [batch_size, horizon] - predicted scale (volatility)
        target_returns: [batch_size, horizon] - actual returns
        """
        s = torch.clamp(pred_sigma, min=1e-6)
        nu = self.nu
        term1 = torch.log(s)
        term2 = (nu + 1.0) / 2.0 * torch.log(1.0 + (target_returns / s) ** 2 / nu)
        nll = term1 + term2
        return torch.mean(nll)
