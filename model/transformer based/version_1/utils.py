import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis

def calculate_fixed_nu(returns):
    """
    Calculate fixed degrees of freedom (nu) for Student-t distribution based on excess kurtosis of the training data.
    """
    k = kurtosis(returns, fisher=True, nan_policy='omit')
    if k <= 0:
        return 30.0 # If not leptokurtic, assume distribution is near normal
    nu = 4.0 + 6.0 / k
    # Cap nu to a reasonable range [2.1, 30.0] to avoid mathematical instability
    return float(np.clip(nu, 2.1, 30.0))

class StudentTNLLLoss(nn.Module):
    def __init__(self, nu):
        """
        nu: fixed degrees of freedom
        """
        super().__init__()
        self.nu = nu
        
    def forward(self, pred_sigma, target_returns):
        """
        pred_sigma: [batch_size, horizon] - predicted scale parameter (volatility)
        target_returns: [batch_size, horizon] - actual returns to evaluate likelihood
        """
        # Clamp to avoid log(0) or division by zero
        s = torch.clamp(pred_sigma, min=1e-6)
        
        nu = self.nu
        # NLL formula for zero-mean Student-t
        # Constant terms (Gamma functions) are omitted as they don't affect gradients for s
        term1 = torch.log(s)
        term2 = (nu + 1.0) / 2.0 * torch.log(1.0 + (target_returns / s)**2 / nu)
        
        nll = term1 + term2
        return torch.mean(nll)

def calc_mse(pred_vol, target_vol):
    return np.mean((pred_vol - target_vol)**2)

def calc_mae(pred_vol, target_vol):
    return np.mean(np.abs(pred_vol - target_vol))

def calc_qlike(pred_vol, target_vol):
    # QLIKE = y / y_hat - log(y / y_hat) - 1
    # where y is variance (vol^2) and y_hat is predicted variance (pred_vol^2)
    # The paper mentions QLIKE for volatility:
    # y_t / y_hat_t - log(y_t / y_hat_t) - 1. We assume y_t here is realized volatility.
    # Note: Sometimes QLIKE is defined on variance. The paper formula (5) uses y_t, y_hat_t as volatility.
    pred = np.clip(pred_vol, 1e-6, None)
    ratio = target_vol / pred
    return np.mean(ratio - np.log(ratio) - 1.0)
