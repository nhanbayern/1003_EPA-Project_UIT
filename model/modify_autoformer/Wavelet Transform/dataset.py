import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class VolatilityDataset(Dataset):
    def __init__(self, df, lookback=60, horizon=21):
        """
        df: DataFrame containing at least 'log_return' and 'time' columns.
        """
        if 'log_return' not in df.columns:
            if 'close' in df.columns:
                df = df.copy()
                df['log_return'] = np.log(df['close'] / df['close'].shift(1)) * 100.0
            else:
                raise ValueError("DataFrame must contain 'close' or 'log_return' column")

        self.returns = df['log_return'].fillna(0).values
        self.time = df['time'].values if 'time' in df.columns else np.arange(len(self.returns))

        # Precompute rolling std as the target volatility proxy
        returns_series = pd.Series(self.returns)
        self.rolling_std = returns_series.rolling(window=lookback).std().values

        self.lookback = lookback
        self.horizon = horizon

        self.valid_indices = []
        for t in range(lookback, len(self.returns) - horizon + 1):
            self.valid_indices.append(t)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        # Input: raw log returns over lookback window
        x = self.returns[t - self.lookback: t]

        # Target: rolling std volatility over horizon
        y_vol = self.rolling_std[t: t + self.horizon]
        y_ret = self.returns[t: t + self.horizon]

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)   # [60, 1]
        y_vol_tensor = torch.tensor(y_vol, dtype=torch.float32)          # [21]
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32)          # [21]

        return x_tensor, y_vol_tensor, y_ret_tensor, t
