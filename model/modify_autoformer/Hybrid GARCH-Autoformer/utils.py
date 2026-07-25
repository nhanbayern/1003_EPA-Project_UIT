import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

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
    pred = np.clip(pred_vol, 1e-6, None)
    ratio = target_vol / pred
    return np.mean(ratio - np.log(ratio) - 1.0)

def plot_loss_curve(train_losses, val_losses, save_path, title="Training vs Validation Loss"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Student-t NLL)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_predictions(true_vol, pred_vol, save_path, title="True vs Predicted Volatility"):
    plt.figure(figsize=(10, 5))
    # Taking the first horizon predictions for a clear plot as example
    plt.plot(true_vol, label='True Volatility', color='black', alpha=0.7)
    plt.plot(pred_vol, label='Predicted Volatility', color='orange', alpha=0.8)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
