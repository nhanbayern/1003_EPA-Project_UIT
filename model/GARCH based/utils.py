import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import FIXED_SPLITS

def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to FIXED_SPLITS key (e.g., VN30_INDEX.csv -> VN30_INDEX)."""
    return str(name).upper().replace(".CSV", "")

def get_default_dataset_dir():
    # Check possible Kaggle paths
    possible_kaggle_paths = [
        Path("/kaggle/input/datasets/trnhngv/historical-price"),
        Path("/kaggle/input/historical-price"),
        Path("/kaggle/input/historical-price/dataset")
    ]
    
    for p in possible_kaggle_paths:
        if p.exists() and any(p.glob("*.csv")):
            return p
    
    # Fallback to local path for testing (safe for Jupyter)
    try:
        return Path(__file__).resolve().parents[3] / "dataset"
    except NameError:
        return Path.cwd() / "dataset"

def load_close_series(csv_path):
    """Load close prices from CSV into pandas Series."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"Missing close column in {csv_path}")

    if "time" in df.columns:
        idx = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
    else:
        idx = pd.RangeIndex(len(close))

    close = pd.to_numeric(df["close"], errors="coerce")
    valid = pd.Series(close.values, index=idx).dropna().sort_index()
    valid = valid[valid.index >= pd.Timestamp("2010-01-01")]
    if valid.empty:
        raise ValueError(f"No valid close prices in {csv_path}")

    return valid.sort_index()

def prepare_series(close_prices, volatility_window=60):
    """
    Compute log-returns and 60-day rolling volatility from close prices.
    
    Returns are in percentage scale: ln(Close_t / Close_{t-1}) * 100
    Volatility uses shift(1) to ensure causality: at time t, vol is computed
    from returns ending at t-1, matching the definition in problem.md.
    
    Returns and volatility are aligned but may contain NaN at the beginning.
    NaN rows are NOT dropped here to preserve raw data indexing for splits.
    """
    close = pd.Series(close_prices).dropna().astype(float)
    if close.empty:
        raise ValueError("close_prices is empty")

    returns = np.log(close / close.shift(1)) * 100.0
    # shift(1): vol at time t = std(returns[t-60 : t-1]), ensuring causality
    volatility = returns.rolling(window=int(volatility_window)).std(ddof=0).shift(1)

    return returns, volatility

def get_split_indices(n_raw_samples, dataset_name):
    """
    Get split indices based on FIXED_SPLITS.
    These indices are applied to the RAW data (close prices after loading),
    NOT to the processed returns/volatility series.
    This ensures alignment with Moirai and Transformer pipelines.
    """
    norm_name = _normalize_dataset_name(dataset_name)
    if norm_name not in FIXED_SPLITS:
        raise ValueError(f"Dataset '{dataset_name}' not in FIXED_SPLITS")

    train_cnt, val_cnt, test_cnt = FIXED_SPLITS[norm_name]
    required = train_cnt + val_cnt + test_cnt
    if int(n_raw_samples) < required:
        raise ValueError(
            f"Not enough rows for declared fixed split {norm_name}: "
            f"need {required}, got {n_raw_samples}"
        )
    return train_cnt, train_cnt + val_cnt

def save_predictions_csv(index_name, model_name, predictions_dict, origin_times,
                         future_returns, out_dir, history_returns=None, split="test"):
    """
    Save forecasts using the shared causal evaluation schema.

    Each row is indexed by forecast origin ``t``. For horizon ``h`` the
    realized target is rolling ``std(60)`` at endpoint ``t+h`` and
    ``log_return`` is the first unseen return ``r[t+1]``.
    """
    records = []
    horizons = sorted(list(predictions_dict.keys()))

    future_arr = np.asarray(future_returns, dtype=float)
    history_arr = np.asarray(history_returns if history_returns is not None else [], dtype=float)
    combined_returns = np.concatenate([history_arr, future_arr])
    origin_arr = np.asarray(origin_times)
    if len(origin_arr) != len(future_arr):
        raise ValueError(
            "origin_times and future_returns must be aligned one-for-one "
            f"(got {len(origin_arr)} and {len(future_arr)})"
        )
    if not horizons or any(int(h) < 1 for h in horizons):
        raise ValueError("predictions_dict must contain positive horizons")

    for h in horizons:
        h = int(h)
        preds = np.asarray(predictions_dict[h], dtype=float)
        valid_len = min(len(preds), len(origin_arr) - h + 1)
        if valid_len <= 0:
            continue
        
        for i in range(valid_len):
            realized = future_arr[i:i + h]
            end_position = len(history_arr) + i + h - 1
            target_window = combined_returns[end_position - 59:end_position + 1]
            true_vol = np.std(target_window, ddof=0) if len(target_window) == 60 and np.isfinite(target_window).all() else np.nan
            time_val = origin_arr[i]
            log_ret = realized[0] if len(realized) else np.nan
            
            records.append({
                "dataset": index_name,
                "model": model_name,
                "horizon": h,
                "time": time_val,
                "log_return": log_ret,
                "actual_vol": true_vol,
                "pred_vol": preds[i]
            })
            
    df = pd.DataFrame(records)
    suffix = "" if split == "test" else f"_{split}"
    out_path = Path(out_dir) / f"{index_name}_{model_name}{suffix}_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

def plot_predictions(index_name, model_name, predictions_dict, test_time, test_v, out_dir, horizon=21):
    """Plot true vs predicted volatility for a specific horizon."""
    if horizon not in predictions_dict:
        return
    
    preds = predictions_dict[horizon]
    valid_len = len(preds)
    align_time = test_time[-valid_len:]
    align_v = test_v[-valid_len:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(align_time, align_v, label='True Volatility', alpha=0.7)
    plt.plot(align_time, preds, label=f'Predicted Volatility (h={horizon})', alpha=0.7)
    plt.title(f'{index_name} - {model_name} Volatility Forecast (Horizon={horizon})')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(True)
    
    out_path = Path(out_dir) / f"{index_name}_{model_name}_h{horizon}_plot.png"
    plt.savefig(out_path)
    plt.close()
