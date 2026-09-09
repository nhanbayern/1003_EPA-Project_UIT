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
        
        # Input features: [t-lookback : t] (exclusive of t, so [t-60 : t])
        x = self.returns[t - self.lookback : t]
        
        # At origin t-1, evaluate each h by realized RMS volatility over the
        # next h unobserved returns.  This is the common MoiraiVaR target.
        y_vol = np.array([
            np.sqrt(np.mean(self.returns[t : t + h] ** 2))
            for h in range(1, self.horizon + 1)
        ], dtype=np.float32)
        y_ret = self.returns[t : t + self.horizon]
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) # [60, 1]
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32) # [21]
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32) # [21]
        
        return x_tensor, y_vol_tensor, y_ret_tensor, t
