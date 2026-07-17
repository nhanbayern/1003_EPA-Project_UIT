import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from config import DEFAULT_SEQ_LEN
from utils import load_close_series, prepare_series, get_split_indices

def create_sliding_windows(returns, volatility, seq_len=DEFAULT_SEQ_LEN):
    """
    Create sliding windows for training (target horizon=1).
    GARCH-LSTM is trained with h=1, and recursively predicts during testing.
    """
    r = pd.Series(returns).astype(float)
    v = pd.Series(volatility).astype(float)
    n = min(len(r), len(v))

    horizon = 1
    max_start = n - seq_len - horizon + 1

    windows = {
        "encoder_returns": [],
        "decoder_volatility": [],
        "target_returns": [],
        "target_variance": [],
    }

    for i in range(max_start):
        windows["encoder_returns"].append(r.iloc[i : i + seq_len].values)
        windows["decoder_volatility"].append(v.iloc[i : i + seq_len].values)
        windows["target_returns"].append(float(r.iloc[i + seq_len + horizon - 1]))
        windows["target_variance"].append(float(v.iloc[i + seq_len + horizon - 1]))

    return {k: np.asarray(v, dtype=np.float32) for k, v in windows.items()}

class GARCHDataset(Dataset):
    def __init__(self, data_dict):
        self.encoder_returns = torch.tensor(data_dict["encoder_returns"], dtype=torch.float32)
        self.decoder_volatility = torch.tensor(data_dict["decoder_volatility"], dtype=torch.float32)
        self.target_returns = torch.tensor(data_dict["target_returns"], dtype=torch.float32)
        self.target_variance = torch.tensor(data_dict["target_variance"], dtype=torch.float32)

    def __len__(self):
        return len(self.target_returns)

    def __getitem__(self, idx):
        return (
            self.encoder_returns[idx],
            self.decoder_volatility[idx],
            self.target_returns[idx],
            self.target_variance[idx]
        )

def get_dataloaders(csv_path, batch_size=32, seq_len=DEFAULT_SEQ_LEN):
    from pathlib import Path
    csv_path = Path(csv_path)
    dataset_name = csv_path.stem
    
    close_series = load_close_series(csv_path)
    returns, volatility = prepare_series(close_series)
    
    # Using fixed splits
    train_end, val_end = get_split_indices(len(returns), dataset_name)
    
    train_r, train_v = returns.iloc[:train_end], volatility.iloc[:train_end]
    val_r, val_v = returns.iloc[train_end:val_end], volatility.iloc[train_end:val_end]
    test_r, test_v = returns.iloc[val_end-seq_len:], volatility.iloc[val_end-seq_len:]
    
    train_data = create_sliding_windows(train_r, train_v, seq_len)
    val_data = create_sliding_windows(val_r, val_v, seq_len)
    
    train_loader = DataLoader(GARCHDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GARCHDataset(val_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_r, test_v, returns.index[val_end:]
