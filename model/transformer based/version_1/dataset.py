import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VolatilityDataset(Dataset):
    def __init__(self, df, lookback=60, horizon=21):
        """
        df: DataFrame containing at least 'close' or 'log_return'
        """
        df = df.copy()
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
        for t in range(lookback - 1, len(self.returns) - horizon):
            self.valid_indices.append(t)
            
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        # Input features: r[t-lookback+1:t+1], ending at the forecast origin t.
        x = self.returns[t - self.lookback + 1 : t + 1]
        
        # Targets for horizon 1 to 21
        y_vol = np.zeros(self.horizon, dtype=np.float32)
        y_ret = np.zeros(self.horizon, dtype=np.float32)
        
        for h in range(1, self.horizon + 1):
            # Target is rolling volatility at the future endpoint t+h.
            window_returns = self.returns[t + h - self.lookback + 1 : t + h + 1]
            
            y_vol[h-1] = np.std(window_returns, ddof=0)
            y_ret[h-1] = self.returns[t + h]
            
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) # [60, 1]
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32) # [21]
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32) # [21]
        
        return x_tensor, y_vol_tensor, y_ret_tensor, t
