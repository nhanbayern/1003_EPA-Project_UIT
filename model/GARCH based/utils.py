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

def prepare_series(close_prices, volatility_window=60):
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

def save_predictions_csv(index_name, model_name, predictions_dict, test_time, test_r, test_v, out_dir):
    """
    Saves predictions in the format: time, log_return, horizon, true_volatility, predict_volatility
    predictions_dict: {horizon: array_of_predicted_volatilities}
    """
    records = []
    horizons = sorted(list(predictions_dict.keys()))
    
    for h in horizons:
        preds = predictions_dict[h]
        valid_len = len(preds)
        
        align_time = test_time[-valid_len:]
        align_r = test_r[-valid_len:]
        align_v = test_v[-valid_len:]
        
        for i in range(valid_len):
            records.append({
                "time": align_time[i],
                "log_return": align_r.iloc[i] if isinstance(align_r, pd.Series) else align_r[i],
                "horizon": h,
                "true_volatility": align_v.iloc[i] if isinstance(align_v, pd.Series) else align_v[i],
                "predict_volatility": preds[i]
            })
            
    df = pd.DataFrame(records)
    out_path = Path(out_dir) / f"{index_name}_{model_name}_predictions.csv"
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
