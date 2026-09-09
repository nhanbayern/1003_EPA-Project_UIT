import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VolatilityDataset(Dataset):
    def __init__(self, df, lookback=60, horizon=21, garch_params=None):
        """
        df: DataFrame containing 'log_return', 'garch_vol', and 'garch_z'
        garch_params: dictionary containing GJR-GARCH parameters: omega, alpha, gamma, beta
        """
        for col in ['log_return', 'garch_vol', 'garch_z']:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
                
        self.returns = df['log_return'].fillna(0).values
        self.garch_z = df['garch_z'].fillna(0).values
        self.garch_vol = df['garch_vol'].fillna(1e-6).values
        self.time = df['time'].values if 'time' in df.columns else np.arange(len(self.returns))
        
        self.lookback = lookback
        self.horizon = horizon
        self.garch_params = garch_params
        
        self.valid_indices = []
        for t in range(lookback, len(self.returns) - horizon + 1):
            self.valid_indices.append(t)
            
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # x is the standardized residuals
        x = self.garch_z[t - self.lookback : t]
        
        y_vol = np.array([
            np.sqrt(np.mean(self.returns[t : t + h] ** 2))
            for h in range(1, self.horizon + 1)
        ], dtype=np.float32)
        y_ret = self.returns[t : t + self.horizon]
        
        # Multi-step GARCH forecast
        garch_vol_forecasts = []
        curr_var = self.garch_vol[t] ** 2
        garch_vol_forecasts.append(np.sqrt(curr_var))
        
        if self.garch_params is not None:
            omega = self.garch_params.get('omega', 0)
            alpha = self.garch_params.get('alpha', 0)
            gamma = self.garch_params.get('gamma', 0)
            beta = self.garch_params.get('beta', 0)
            
            for h in range(1, self.horizon):
                next_var = omega + (alpha + 0.5 * gamma + beta) * curr_var
                garch_vol_forecasts.append(np.sqrt(next_var))
                curr_var = next_var
        else:
            for h in range(1, self.horizon):
                garch_vol_forecasts.append(np.sqrt(curr_var))
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_garch_tensor = torch.tensor(garch_vol_forecasts, dtype=torch.float32)
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32)
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32)
        
        return x_tensor, y_garch_tensor, y_vol_tensor, y_ret_tensor, t
