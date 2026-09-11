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
    
    Skips windows where any value is NaN (handles the initial NaN region
    from rolling std + shift).
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
        r_window = r.iloc[i : i + seq_len]
        v_window = v.iloc[i : i + seq_len]
        # Forecast origin is the final observed return in the encoder.
        # The target is rolling volatility at the one-step future endpoint;
        # ``v`` is only the historical causal decoder feature.
        target_r = r.iloc[i + seq_len]
        target_window = r.iloc[i + 1 : i + seq_len + 1]
        target_v = np.std(target_window, ddof=0)
        
        # Skip windows with any NaN values
        if r_window.isna().any() or v_window.isna().any() or pd.isna(target_r) or pd.isna(target_v):
            continue
        
        windows["encoder_returns"].append(r_window.values)
        windows["decoder_volatility"].append(v_window.values)
        windows["target_returns"].append(float(target_r))
        windows["target_variance"].append(float(target_v))

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
    """
    Load data, compute returns/volatility on FULL series, then split using
    FIXED_SPLITS indices on the RAW data positions.
    
    This ensures test period alignment with Moirai and Transformer pipelines.
    """
    from pathlib import Path
    csv_path = Path(csv_path)
    dataset_name = csv_path.stem
    
    close_series = load_close_series(csv_path)
    returns, volatility = prepare_series(close_series)
    
    # Split indices on RAW data (close prices) — matches Moirai/Transformer
    # returns/volatility have the same index as close_series (minus the first row for diff)
    # The first row of returns is NaN (from log diff), and first 61 rows of volatility are NaN
    # (60 for rolling window + 1 for shift). But we keep them to preserve index alignment.
    train_end, val_end = get_split_indices(len(close_series), dataset_name)
    
    # Training data: use returns/volatility from the train region
    # NaN values at the beginning will be skipped by create_sliding_windows
    train_r = returns.iloc[:train_end]
    train_v = volatility.iloc[:train_end]
    
    # Validation data: include lookback window from end of training
    val_r = returns.iloc[max(0, train_end - seq_len):val_end]
    val_v = volatility.iloc[max(0, train_end - seq_len):val_end]
    
    # Test data: include lookback window from end of validation for GARCH-LSTM
    test_r = returns.iloc[max(0, val_end - seq_len):]
    test_v = volatility.iloc[max(0, val_end - seq_len):]
    
    # Test time is only the actual test period (from val_end onwards)
    test_time = returns.index[val_end:]
    
    train_data = create_sliding_windows(train_r, train_v, seq_len)
    val_data = create_sliding_windows(val_r, val_v, seq_len)
    
    train_loader = DataLoader(GARCHDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GARCHDataset(val_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_r, test_v, test_time
