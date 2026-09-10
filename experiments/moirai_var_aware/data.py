from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import HORIZONS, LOOKBACK, SPLIT_INFO


class VolatilityDataset(Dataset):
    """Origin-inclusive 60-return contexts with strictly future volatility targets."""

    def __init__(self, df: pd.DataFrame, lookback: int = LOOKBACK, horizons: list[int] | None = None) -> None:
        self.lookback = lookback
        self.horizons = horizons or HORIZONS

        df = df.sort_values("time").reset_index(drop=True)
        close = df["close"].values
        returns = np.diff(np.log(close)) * 100
        self.returns = np.concatenate([[0.0], returns])
        df["returns"] = self.returns
        self.times = df["time"].values

        self.valid_indices = [
            i for i in df[df["time"] >= "2010-01-01"].index.tolist()
            if i >= lookback - 1
        ]
        self.samples = []
        n_returns = len(self.returns)

        for origin_position, t in enumerate(self.valid_indices):
            if t + max(self.horizons) >= n_returns:
                continue

            x = self.returns[t - lookback + 1 : t + 1]
            y = []
            for horizon in self.horizons:
                # Origin t may only use observations through r[t].  The
                # target is volatility of the next h returns r[t+1:t+h+1].
                target_window = self.returns[t + 1 : t + horizon + 1]
                y.append(np.std(target_window, ddof=0))
            self.samples.append(
                {
                    "x": torch.tensor(x, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                    "time": self.times[t],
                    "log_return": float(self.returns[t + 1]),
                    "origin_position": origin_position,
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

    max_horizon = max(full_ds.horizons)
    val_start, val_end = train_size, train_size + val_size
    test_start, test_end = val_end, val_end + test_size

    def subset_for(start, end):
        allowed = {
            s["origin_position"]
            for s in full_ds.samples
            if start <= s["origin_position"] < end - max_horizon
        }
        indices = [i for i, s in enumerate(full_ds.samples) if s["origin_position"] in allowed]
        return torch.utils.data.Subset(full_ds, indices)

    train_ds = subset_for(0, train_size)
    val_ds = subset_for(val_start, val_end)
    test_ds = subset_for(test_start, test_end)
    return train_ds, val_ds, test_ds
