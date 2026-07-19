from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import HORIZONS, LOOKBACK, SPLIT_INFO


class VolatilityDataset(Dataset):
    """Return-window samples with volatility targets and realized return for VaR loss."""

    def __init__(self, df: pd.DataFrame, lookback: int = LOOKBACK, horizons: list[int] | None = None) -> None:
        self.lookback = lookback
        self.horizons = horizons or HORIZONS

        df = df.sort_values("time").reset_index(drop=True)
        close = df["close"].values
        returns = np.diff(np.log(close)) * 100
        self.returns = np.concatenate([[0.0], returns])
        df["returns"] = self.returns
        self.times = df["time"].values

        self.valid_indices = df[df["time"] >= "2010-01-01"].index.tolist()
        self.samples = []
        n_returns = len(self.returns)

        for t in self.valid_indices:
            x = self.returns[t - lookback : t]
            y = []
            for horizon in self.horizons:
                start = t + horizon - 1 - lookback
                end = t + horizon - 1
                if end > n_returns:
                    end = n_returns
                    start = max(0, n_returns - lookback)
                y.append(np.std(self.returns[start:end]))
            self.samples.append(
                {
                    "x": torch.tensor(x, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                    "time": self.times[t],
                    "log_return": float(self.returns[t]),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return (
            sample["x"],
            sample["y"],
            torch.tensor(sample["log_return"], dtype=torch.float32),
        )


def load_and_split_dataset(csv_path: str, index_name: str):
    df = pd.read_csv(csv_path)
    full_ds = VolatilityDataset(df)
    split = SPLIT_INFO[index_name]
    train_size = split["train"]
    val_size = split["validation"]
    test_size = split["test"]

    train_ds = torch.utils.data.Subset(full_ds, range(0, train_size))
    val_ds = torch.utils.data.Subset(full_ds, range(train_size, train_size + val_size))
    test_ds = torch.utils.data.Subset(full_ds, range(train_size + val_size, train_size + val_size + test_size))
    return train_ds, val_ds, test_ds

