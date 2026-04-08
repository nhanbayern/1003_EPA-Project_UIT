from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEQ_LEN = 126


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


def prepare_aaai24_data(close_prices):
    close = pd.Series(close_prices).dropna().astype(float)
    if close.empty:
        raise ValueError("close_prices is empty after dropping NaN")

    returns = np.log(close / close.shift(1)).dropna()

    volatility = (returns ** 2).rolling(window=5).mean().dropna()
    returns = returns.loc[volatility.index]

    returns = returns * 100.0
    volatility = volatility * 100.0

    n = len(returns)
    if n < 20:
        raise ValueError("Not enough samples after preprocessing")

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_r = returns.iloc[:train_end]
    train_v = volatility.iloc[:train_end]

    val_r = returns.iloc[train_end:val_end]
    val_v = volatility.iloc[train_end:val_end]

    test_r = returns.iloc[val_end:]
    test_v = volatility.iloc[val_end:]

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


def print_split_report(train_split, val_split, test_split, seq_len=DEFAULT_SEQ_LEN):
    info = describe_split(train_split, val_split, test_split)
    print("Split report")
    print(f"Train: {info['train']} ({info['train_ratio']:.4f})")
    print(f"Val:   {info['val']} ({info['val_ratio']:.4f})")
    print(f"Test:  {info['test']} ({info['test_ratio']:.4f})")
    print(f"seq_len: {seq_len}")
