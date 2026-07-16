import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VolatilityDataset(Dataset):
    def __init__(self, df, lookback=60, horizon=21):
        """
        df: DataFrame containing at least 'close' or 'log_return'
        """
        if 'log_return' not in df.columns:
            if 'close' in df.columns:
                df['log_return'] = np.log(df['close'] / df['close'].shift(1)) * 100.0
            else:
                raise ValueError("DataFrame must contain 'close' or 'log_return' column")
        
        # Drop initial NaNs from log_return calculation
        self.returns = df['log_return'].fillna(0).values
        self.time = df['time'].values if 'time' in df.columns else np.arange(len(self.returns))
        
        self.lookback = lookback
        self.horizon = horizon
        
        self.valid_indices = []
        # t is the end of the lookback window
        for t in range(lookback, len(self.returns) - horizon + 1):
            self.valid_indices.append(t)
            
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # Input features: [t-lookback : t-1] (inclusive of t-1, exclusive of t)
        x = self.returns[t - self.lookback : t]
        
        # Targets for horizon 1 to 21
        y_vol = np.zeros(self.horizon, dtype=np.float32)
        y_ret = np.zeros(self.horizon, dtype=np.float32)
        
        for h in range(1, self.horizon + 1):
            # Target volatility for step h is std over [t+h-60 : t+h-1]
            start_idx = t + h - self.lookback
            end_idx = t + h
            window_returns = self.returns[start_idx : end_idx]
            
            # Using ddof=1 for sample standard deviation as per standard practice
            y_vol[h-1] = np.std(window_returns, ddof=1)
            y_ret[h-1] = self.returns[t + h - 1]
            
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) # [60, 1]
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32) # [21]
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32) # [21]
        
        return x_tensor, y_vol_tensor, y_ret_tensor, t
