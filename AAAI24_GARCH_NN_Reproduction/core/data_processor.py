from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEQ_LEN = 60
DEFAULT_VOL_WINDOW = 60
EVAL_HORIZONS = (1, 3, 5, 10, 21)

# Fixed split counts per dataset (Train, Val, Test) - data range 2010-2025
# Keys must match SPLIT_COUNTS from notebook specification
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

# For reference: Original SPLIT_COUNTS specification (with original case)
SPLIT_COUNTS_SPEC = {
    "VN30_INDEX": (1971, 1096, 925),
    "VN_INDEX": (1964, 1103, 925),
    "DAX_40": (1983, 1104, 972),
    "EuroNext_100": (2005, 1115, 979),
    "IBEX_35": (1815, 1362, 923),
    "KOSPI_index": (1944, 1109, 881),
    "SMI": (1987, 1135, 901),
    "snp500": (1988, 1109, 927),
    "Nikkei_225": (1677, 1174, 1062),
}


def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to FIXED_SPLITS key (e.g., VN30_INDEX.csv -> VN30_INDEX)."""
    return str(name).upper().replace(".CSV", "")


def get_default_dataset_dir():
    # Keep data external to this reproduction folder as requested.
    return Path(__file__).resolve().parents[2] / "dataset"


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

    # Try to parse date/time column
    if "time" in df.columns:
        idx = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
    else:
        idx = pd.RangeIndex(len(close))

    # Align close and index, drop NaNs
    valid = pd.Series(close.values, index=idx).dropna()
    if valid.empty:
        raise ValueError(f"No valid close prices in {csv_path}")

    return valid.sort_index()


def filter_close_by_date(close_prices, date_start=None, date_end=None):
    """Filter close prices by date range. None means no filter."""
    if date_start is None and date_end is None:
        return pd.Series(close_prices).dropna().astype(float)

    close = pd.Series(close_prices).dropna().astype(float)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close_prices index must be DatetimeIndex for date filtering")

    if date_start:
        close = close[close.index >= pd.Timestamp(date_start)]
    if date_end:
        close = close[close.index <= pd.Timestamp(date_end)]

    return close


def prepare_aaai24_series(close_prices, volatility_window=DEFAULT_VOL_WINDOW):
    """
    Compute log-returns and a historical 60-day volatility feature.
    
    Returns: ln(Close_t / Close_{t-1}) × 100  (log-return in percent-points)
    Vol: population std over the observed rolling window × 100 (feature only)
    
    Both are scaled to 100× for percentage representation.
    """
    close = pd.Series(close_prices).dropna().astype(float)
    if close.empty:
        raise ValueError("close_prices is empty")

    # Log-return: ln(Close_t / Close_{t-1})
    returns = np.log(close / close.shift(1)).dropna()

    # Historical rolling volatility is an input feature.  It is evaluated at
    # the current observed return and is never used as the future target.
    volatility = returns.rolling(window=int(volatility_window)).std(ddof=0).dropna()

    # Align returns to volatility index (remove leading NaNs from rolling)
    returns = returns.loc[volatility.index]

    # Convert to percent-points
    returns = returns * 100.0
    volatility = volatility * 100.0

    if returns.empty or volatility.empty:
        raise ValueError("Not enough samples after preprocessing")

    return returns, volatility


def _get_split_indices(n_samples, split_mode, dataset_name=None):
    """Compute train/val split indices based on split_mode."""
    if split_mode == "ratio":
        train_end = int(n_samples * 0.8)
        val_end = int(n_samples * 0.9)
        return train_end, val_end

    if split_mode == "fixed_counts":
        if not dataset_name:
            raise ValueError("dataset_name required for fixed_counts mode")

        norm_name = _normalize_dataset_name(dataset_name)
        if norm_name not in FIXED_SPLITS:
            raise ValueError(f"Dataset '{dataset_name}' not in FIXED_SPLITS")

        train_cnt, val_cnt, test_cnt = FIXED_SPLITS[norm_name]
        required = train_cnt + val_cnt + test_cnt
        if n_samples < required:
            raise ValueError(
                f"Not enough data for declared fixed split {norm_name}: "
                f"need {required}, got {n_samples}. Refusing to shrink the test set."
            )

        return train_cnt, train_cnt + val_cnt

    raise ValueError(f"Unknown split_mode: {split_mode}")


def prepare_aaai24_data(close_prices, split_mode="ratio", dataset_name=None, 
                        date_start=None, date_end=None):
    """
    Prepare data: filter by date, compute returns/volatility, split train/val/test.
    
    Returns: ((train_r, train_v), (val_r, val_v), (test_r, test_v))
    """
    # Filter by date
    close = filter_close_by_date(close_prices, date_start, date_end)

    # Compute returns and volatility
    returns, volatility = prepare_aaai24_series(close)

    # Split data
    train_end, val_end = _get_split_indices(len(returns), split_mode, dataset_name)

    train_r, train_v = returns.iloc[:train_end], volatility.iloc[:train_end]
    val_r, val_v = returns.iloc[train_end:val_end], volatility.iloc[train_end:val_end]
    test_r, test_v = returns.iloc[val_end:], volatility.iloc[val_end:]

    return (train_r, train_v), (val_r, val_v), (test_r, test_v)


def create_sliding_windows(returns, volatility, seq_len=DEFAULT_SEQ_LEN, horizon=1, multi_horizon=False):
    """
    Create sliding windows for seq2seq model training.
    
    Args:
        multi_horizon: if True, create targets for all horizons [1..max_horizon]
                      else create targets for single horizon only
    """
    r = pd.Series(returns).reset_index(drop=True).astype(float)
    v = pd.Series(volatility).reset_index(drop=True).astype(float)
    n = min(len(r), len(v))

    if seq_len < 1 or horizon < 1:
        raise ValueError("seq_len and horizon must be >= 1")

    if multi_horizon:
        # One-shot multi-horizon: create targets for [1, 3, 5, 10, 21] in one sample
        horizons = list(EVAL_HORIZONS)
        max_horizon = max(horizons)
        max_start = n - seq_len - max_horizon + 1
        
        if max_start <= 0:
            raise ValueError(f"Not enough samples for multi-horizon: n={n}, seq_len={seq_len}, max_horizon={max_horizon}")

        windows = {
            "encoder_returns": [],
            "decoder_volatility": [],
            "target_returns": [],
            "target_variance": [],
        }

        for i in range(max_start):
            windows["encoder_returns"].append(r.iloc[i : i + seq_len].values)
            windows["decoder_volatility"].append(v.iloc[i : i + seq_len].values)
            # The return target is the return at the forecast endpoint.  The
            # volatility target is computed strictly from future returns; the
            # historical rolling series is a causal input feature only.
            target_ret = [float(r.iloc[i + seq_len + h - 1]) for h in horizons]
            target_var = [
                float(np.std(r.iloc[i + seq_len : i + seq_len + h], ddof=0))
                for h in horizons
            ]
            windows["target_returns"].append(target_ret)
            windows["target_variance"].append(target_var)

        return {
            "encoder_returns": np.asarray(windows["encoder_returns"], dtype=np.float32),
            "decoder_volatility": np.asarray(windows["decoder_volatility"], dtype=np.float32),
            "target_returns": np.asarray(windows["target_returns"], dtype=np.float32),  # (N, 5)
            "target_variance": np.asarray(windows["target_variance"], dtype=np.float32),  # (N, 5)
        }
    else:
        # Rolling forecast: single horizon target
        max_start = n - seq_len - horizon + 1
        if max_start <= 0:
            raise ValueError(f"Not enough samples: n={n}, seq_len={seq_len}, horizon={horizon}")

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
            future_returns = r.iloc[i + seq_len : i + seq_len + horizon]
            windows["target_variance"].append(float(np.std(future_returns, ddof=0)))

        return {k: np.asarray(v, dtype=np.float32) for k, v in windows.items()}

