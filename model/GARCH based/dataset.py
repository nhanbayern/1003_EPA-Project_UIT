import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_SEQ_LEN = 60
DEFAULT_VOL_WINDOW = 60

# Fixed split counts per dataset (Train, Val, Test) - data range 2010-2025
FIXED_SPLITS = {
    "VN30_INDEX": (1971, 1096, 925),
    "VN_INDEX": (1964, 1103, 925),
    "DAX_40": (1983, 1104, 972),
    "EURONEXT_100": (2005, 1115, 979),
    "IBEX_35": (1815, 1362, 923),
    "KOSPI_INDEX": (1944, 1109, 881),
    "SMI": (1987, 1135, 901),
    "SNP500": (1988, 1109, 927),
    "NIKKEI_225": (1677, 1174, 1062),
}

def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to FIXED_SPLITS key (e.g., VN30_INDEX.csv -> VN30_INDEX)."""
    return str(name).upper().replace(".CSV", "")

def get_default_dataset_dir():
    kaggle_path = Path("/kaggle/input/datasets/trnhngv/historical-price/dataset")
    if kaggle_path.exists():
        return kaggle_path
    
    # Fallback to local path for testing
    return Path(__file__).resolve().parents[3] / "dataset"

def load_close_series(csv_path):
    """Load close prices from CSV into pandas Series."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"Missing close column in {csv_path}")

    close = pd.to_numeric(df["close"], errors="coerce").dropna()

    if "time" in df.columns:
        idx = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
    else:
        idx = pd.RangeIndex(len(close))

    valid = pd.Series(close.values, index=idx).dropna()
    if valid.empty:
        raise ValueError(f"No valid close prices in {csv_path}")

    return valid.sort_index()

def prepare_series(close_prices, volatility_window=DEFAULT_VOL_WINDOW):
    """
    Compute log-returns and 60-day rolling volatility from close prices.
    Returns: ln(Close_t / Close_{t-1}) * 100
    Vol: sqrt(mean((r_t - mean(r))^2)) over past 60 returns * 100
    """
    close = pd.Series(close_prices).dropna().astype(float)
    if close.empty:
        raise ValueError("close_prices is empty")

    returns = np.log(close / close.shift(1)).dropna()
    volatility = returns.rolling(window=int(volatility_window)).std(ddof=0).dropna()
    returns = returns.loc[volatility.index]

    returns = returns * 100.0
    volatility = volatility * 100.0

    return returns, volatility

def get_split_indices(n_samples, dataset_name):
    norm_name = _normalize_dataset_name(dataset_name)
    if norm_name not in FIXED_SPLITS:
        raise ValueError(f"Dataset '{dataset_name}' not in FIXED_SPLITS")

    train_cnt, val_cnt, test_cnt = FIXED_SPLITS[norm_name]
    return train_cnt, train_cnt + val_cnt

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
    csv_path = Path(csv_path)
    dataset_name = csv_path.stem
    
    close_series = load_close_series(csv_path)
    returns, volatility = prepare_series(close_series)
    
    # Using fixed splits
    train_end, val_end = get_split_indices(len(returns), dataset_name)
    
    train_r, train_v = returns.iloc[:train_end], volatility.iloc[:train_end]
    val_r, val_v = returns.iloc[train_end:val_end], volatility.iloc[train_end:val_end]
    # For testing, we need sequential evaluation, so we might just use the raw series in testing logic,
    # but we can provide a test_loader for consistent batching if needed.
    test_r, test_v = returns.iloc[val_end-seq_len:], volatility.iloc[val_end-seq_len:]
    
    train_data = create_sliding_windows(train_r, train_v, seq_len)
    val_data = create_sliding_windows(val_r, val_v, seq_len)
    
    train_loader = DataLoader(GARCHDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GARCHDataset(val_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_r, test_v, returns.index[val_end:]
