from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
    """Normalize dataset name for lookup (e.g., VN30_INDEX.csv -> VN30_INDEX)."""
    name = str(name).upper().replace(".CSV", "").replace("DAX", "DAX").replace("EURONEXT", "EURONEXT").replace("KOSPI", "KOSPI").replace("SNP", "SNP").replace("NIKKEI", "NIKKEI")
    name = name.replace("_INDEX", "_INDEX").replace("_40", "_40").replace("_100", "_100").replace("_35", "_35")
    return name


def get_default_dataset_dir():
    # Keep data external to this reproduction folder as requested.
    return Path(__file__).resolve().parents[2] / "dataset"


def load_close_series(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"Missing close column in file: {csv_path}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        index = df["time"]
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        index = df["date"]
    else:
        index = pd.RangeIndex(start=0, stop=len(df), step=1)

    close = pd.to_numeric(df["close"], errors="coerce")
    out = pd.Series(close.values, index=index, name="close").dropna()
    if out.empty:
        raise ValueError(f"No valid close price values in file: {csv_path}")
    return out


def filter_close_by_date(close_prices, date_start=None, date_end=None, verbose=False):
    """Filter close prices by date range. Dates as None means no filter."""
    close = pd.Series(close_prices).dropna().astype(float)
    
    if date_start is None and date_end is None:
        if verbose:
            print(f"No date filter applied. Data points: {len(close)}")
        return close
    
    # Ensure index is datetime
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close_prices index must be a DatetimeIndex for date filtering")
    
    if date_start:
        date_start = pd.Timestamp(date_start)
    if date_end:
        date_end = pd.Timestamp(date_end)
    
    close_filtered = close.copy()
    if date_start:
        close_filtered = close_filtered[close_filtered.index >= date_start]
    if date_end:
        close_filtered = close_filtered[close_filtered.index <= date_end]
    
    if verbose:
        orig_min = close.index.min().date() if len(close) > 0 else None
        orig_max = close.index.max().date() if len(close) > 0 else None
        filt_min = close_filtered.index.min().date() if len(close_filtered) > 0 else None
        filt_max = close_filtered.index.max().date() if len(close_filtered) > 0 else None
        print(f"Original data: {len(close)} points, range {orig_min} to {orig_max}")
        print(f"After date filter [{date_start}, {date_end}]: {len(close_filtered)} points, range {filt_min} to {filt_max}")
    
    return close_filtered


def prepare_aaai24_series(close_prices, volatility_window=DEFAULT_VOL_WINDOW, verbose=False):
    close = pd.Series(close_prices).dropna().astype(float)
    if close.empty:
        raise ValueError("close_prices is empty after dropping NaN")

    returns = np.log(close / close.shift(1)).dropna()
    volatility = returns.rolling(window=int(volatility_window)).std(ddof=0).dropna()
    returns = returns.loc[volatility.index]

    returns = returns * 100.0
    volatility = volatility * 100.0

    if returns.empty or volatility.empty:
        raise ValueError("Not enough samples after preprocessing")

    if verbose:
        print(f"After rolling volatility (window={volatility_window}): {len(returns)} return points")

    return returns, volatility


def prepare_aaai24_data(close_prices, split_mode="ratio", dataset_name=None, date_start=None, date_end=None, verbose=False):
    """
    Prepare data with optional date filtering and split mode selection.
    
    Args:
        close_prices: pandas Series of close prices
        split_mode: "ratio" (8:1:1) or "fixed_counts" (from FIXED_SPLITS table)
        dataset_name: required if split_mode="fixed_counts" (e.g., "VN30_INDEX", "DAX_40")
        date_start: start date for filtering (e.g., "2010-01-01")
        date_end: end date for filtering (e.g., "2025-12-31")
        verbose: print detailed split report
    
    Returns:
        ((train_r, train_v), (val_r, val_v), (test_r, test_v)): train/val/test splits
    """
    
    # Step 1: Filter by date if specified
    close_filtered = filter_close_by_date(close_prices, date_start, date_end, verbose=verbose)
    
    # Step 2: Preprocess (returns + rolling vol)
    returns, volatility = prepare_aaai24_series(close_filtered, volatility_window=DEFAULT_VOL_WINDOW, verbose=verbose)
    
    n = len(returns)
    if n < 20:
        raise ValueError(f"Not enough samples after preprocessing: {n}")
    
    # Step 3: Split data
    if split_mode == "ratio":
        # Default 8:1:1 ratio
        if verbose:
            print(f"Using split mode: ratio (8:1:1)")
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
    elif split_mode == "fixed_counts":
        # Use FIXED_SPLITS table
        if not dataset_name:
            raise ValueError("dataset_name is required when split_mode='fixed_counts'")
        
        norm_name = _normalize_dataset_name(dataset_name)
        if norm_name not in FIXED_SPLITS:
            valid_names = ", ".join(sorted(FIXED_SPLITS.keys()))
            raise ValueError(f"Dataset '{dataset_name}' not in FIXED_SPLITS. Valid: {valid_names}")
        
        train_cnt, val_cnt, test_cnt = FIXED_SPLITS[norm_name]
        total_expected = train_cnt + val_cnt + test_cnt
        
        if verbose:
            print(f"Using split mode: fixed_counts")
            print(f"Dataset: {norm_name}")
            print(f"Expected split counts: Train={train_cnt}, Val={val_cnt}, Test={test_cnt} (total={total_expected})")
            print(f"Actual returns points: {n}")
        
        # Use exact counts, discard surplus
        if n < total_expected:
            if verbose:
                print(f"WARNING: Got {n} points, need {total_expected}. Test set will be truncated.")
            test_cnt = n - train_cnt - val_cnt
            if test_cnt < 0:
                raise ValueError(f"Not enough data: need {total_expected}, got {n}")
        
        train_end = train_cnt
        val_end = train_cnt + val_cnt
        
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}. Use 'ratio' or 'fixed_counts'")
    
    # Step 4: Extract splits
    train_r = returns.iloc[:train_end]
    train_v = volatility.iloc[:train_end]

    val_r = returns.iloc[train_end:val_end]
    val_v = volatility.iloc[train_end:val_end]

    test_r = returns.iloc[val_end:]
    test_v = volatility.iloc[val_end:]

    if verbose:
        print(f"\nFinal split (returns/volatility space):")
        print(f"  Train: {len(train_r)} points")
        print(f"  Val:   {len(val_r)} points")
        print(f"  Test:  {len(test_r)} points")
        print(f"  Total: {len(train_r) + len(val_r) + len(test_r)} points")

    return (train_r, train_v), (val_r, val_v), (test_r, test_v)


def create_sliding_windows(returns, volatility, seq_len=DEFAULT_SEQ_LEN, horizon=1):
    r = pd.Series(returns).astype(float)
    v = pd.Series(volatility).astype(float)

    n = min(len(r), len(v))
    r = r.iloc[:n].reset_index(drop=True)
    v = v.iloc[:n].reset_index(drop=True)

    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    max_start = n - seq_len - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough samples for seq_len={seq_len}, horizon={horizon}, n={n}"
        )

    encoder_returns = []
    decoder_volatility = []
    target_returns = []
    target_variance = []

    for start in range(max_start):
        end = start + seq_len
        target_idx = end + horizon - 1

        encoder_returns.append(r.iloc[start:end].values)
        decoder_volatility.append(v.iloc[start:end].values)
        target_returns.append(float(r.iloc[target_idx]))
        target_variance.append(float(v.iloc[target_idx]))

    return {
        "encoder_returns": np.asarray(encoder_returns, dtype=np.float32),
        "decoder_volatility": np.asarray(decoder_volatility, dtype=np.float32),
        "target_returns": np.asarray(target_returns, dtype=np.float32),
        "target_variance": np.asarray(target_variance, dtype=np.float32),
    }


def describe_split(train_split, val_split, test_split):
    train_n = len(train_split[0])
    val_n = len(val_split[0])
    test_n = len(test_split[0])
    total = train_n + val_n + test_n

    if total == 0:
        raise ValueError("Empty split")

    return {
        "train": train_n,
        "val": val_n,
        "test": test_n,
        "total": total,
        "train_ratio": train_n / total,
        "val_ratio": val_n / total,
        "test_ratio": test_n / total,
    }


def print_split_report(train_split, val_split, test_split, seq_len=DEFAULT_SEQ_LEN, split_mode="ratio", date_range=None):
    info = describe_split(train_split, val_split, test_split)
    print("\n" + "="*70)
    print("SPLIT REPORT")
    print("="*70)
    if date_range:
        print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Split mode: {split_mode}")
    print("-"*70)
    print(f"Train: {info['train']:6d} samples ({info['train_ratio']:.1%})")
    print(f"Val:   {info['val']:6d} samples ({info['val_ratio']:.1%})")
    print(f"Test:  {info['test']:6d} samples ({info['test_ratio']:.1%})")
    print("-"*70)
    print(f"Total: {info['total']:6d} samples")
    print(f"Seq len: {seq_len}")
    print("="*70 + "\n")
