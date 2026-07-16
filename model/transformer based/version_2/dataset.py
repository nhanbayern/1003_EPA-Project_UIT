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
        
        # Precompute rolling standard deviation using pandas to avoid O(N) in __getitem__
        # pandas rolling window is inclusive of the current index, so rolling(60).std() at index t
        # gives the std of [t-59: t+1], which exactly matches the lookback logic.
        returns_series = pd.Series(self.returns)
        self.rolling_std = returns_series.rolling(window=lookback).std().values
        
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
        
        # Target volatility and returns for horizon 1 to 21
        # Precalculated rolling_std[t] corresponds to horizon 1, rolling_std[t+20] to horizon 21.
        y_vol = self.rolling_std[t : t + self.horizon]
        y_ret = self.returns[t : t + self.horizon]
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) # [60, 1]
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32) # [21]
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32) # [21]
        
        return x_tensor, y_vol_tensor, y_ret_tensor, t
