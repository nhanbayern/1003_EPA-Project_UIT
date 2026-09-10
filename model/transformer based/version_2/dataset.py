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
        # t is the final observed return in the 60-day context.
        for t in range(lookback - 1, len(self.returns) - horizon):
            self.valid_indices.append(t)
            
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # Input features: r[t-59:t+1].
        x = self.returns[t - self.lookback + 1 : t + 1]
        
        # Target is rolling volatility at the future endpoint t+h.
        y_vol = np.array([
            np.std(self.returns[t + h - self.lookback + 1 : t + h + 1], ddof=0)
            for h in range(1, self.horizon + 1)
        ], dtype=np.float32)
        y_ret = self.returns[t + 1 : t + self.horizon + 1]
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) # [60, 1]
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32) # [21]
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32) # [21]
        
        return x_tensor, y_vol_tensor, y_ret_tensor, t
