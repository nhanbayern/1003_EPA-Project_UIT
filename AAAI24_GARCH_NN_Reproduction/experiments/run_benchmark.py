from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from AAAI24_GARCH_NN_Reproduction.core.custom_metrics import compute_metrics
from AAAI24_GARCH_NN_Reproduction.core.data_processor import (
    DEFAULT_VOL_WINDOW,
    DEFAULT_SEQ_LEN,
    create_sliding_windows,
    get_default_dataset_dir,
    load_close_series,
    prepare_aaai24_data,
)
from AAAI24_GARCH_NN_Reproduction.models.dl_baselines import (
    SUPPORTED_DL_MODELS,
    build_dataloaders,
    build_dl_model,
    rolling_forecast_variance as rolling_dl_forecast_variance,
    train_dl_model,
)
from AAAI24_GARCH_NN_Reproduction.models.garch_lstm_hybrid import (
    GARCHLSTMHybrid,
    rolling_forecast_variance as rolling_hybrid_forecast_variance,
    train_garch_lstm_hybrid,
)
from AAAI24_GARCH_NN_Reproduction.models.stat_baselines import (
    rolling_forecast_variance as rolling_stat_forecast_variance,
)


SEEDS = [42, 123, 202, 303, 404]
HORIZONS = [1, 3, 5, 10, 21]
STAT_MODELS = ["GARCH", "GJR-GARCH", "FI-GARCH"]
HYBRID_MODEL_NAME = "GARCH-LSTM-Hybrid"
ROLLING_VOL_WINDOW = DEFAULT_VOL_WINDOW

# Multi-horizon forecasting mode: "rolling" or "one_shot"
# Will be set from notebook or CLI
MULTI_HORIZON_MODE = "one_shot"

# ==================== HYPERPARAMETER TUNING CONFIGURATION ====================
# Tier 1 (High Priority - tune first):
#   - LEARNING_RATE: primary optimization rate
#   - EARLY_STOPPING_PATIENCE: prevent overfitting
#   - LR_PATIENCE: learning rate scheduler frequency
# Tier 2 (Medium Priority - tune after Tier 1 is stable):
#   - LR_FACTOR: learning rate decay multiplier
#   - MIN_LR: learning rate floor
# Strategy: Start with these values, then grid search around them
HYPERPARAMETER_CONFIG = {
    "learning_rate": 1e-2,              # [Tier 1] Optimizer step size. Try: 1e-3, 5e-3, 1e-2, 5e-2
    "lr_factor": 0.5,                  # [Tier 2] LR decay multiplier when val_loss plateaus. Try: 0.3, 0.5, 0.7
    "lr_patience": 8,                  # [Tier 1] Epochs before LR decay. Try: 3, 5, 8, 10
    "early_stopping_patience": 15,     # [Tier 1] Epochs before early stopping. Try: 10, 15, 20, 30
    "min_lr": 1e-5,                    # [Tier 2] Learning rate floor. Try: 1e-7, 1e-6, 1e-5
}
# ==================================================================================

NU_TEST_BY_DATASET = {
    "VN30INDEX": 2.385,
    "VNINDEX": 2.510,
    "DAX40": 4.298,
    "EURONEXT100": 4.072,
    "IBEX35": 5.963,
    "KOSPIINDEX": 5.169,
    "SMI": 4.725,
    "SNP500": 3.423,
    "NIKKEI225": 4.596,
}


def _log(message, enabled=True):
    if not enabled:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _normalize_dataset_key(name):
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def _get_dataset_nu_test(dataset_name):
    key = _normalize_dataset_key(dataset_name)
    if key not in NU_TEST_BY_DATASET:
        valid = ", ".join(sorted(NU_TEST_BY_DATASET.keys()))
        raise KeyError(f"No nu_test mapping for dataset={dataset_name}. Valid keys: {valid}")
    return float(NU_TEST_BY_DATASET[key])


def resolve_device(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return dev


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _discover_dataset_files(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if dataset_dir.is_file():
        if dataset_dir.suffix.lower() != ".csv":
            return []

        # For explicit file input, validate once and surface a precise error if invalid.
        try:
            _ = load_close_series(dataset_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid dataset file: {dataset_dir} ({type(exc).__name__}: {exc})"
            ) from exc

        return [dataset_dir]

    files = sorted(dataset_dir.glob("*.csv"))

    valid_files = []
    for csv_path in files:
        try:
            _ = load_close_series(csv_path)
            valid_files.append(csv_path)
        except Exception:
            continue
    return valid_files


def _build_val_windows(train_r, train_v, val_r, val_v, seq_len, multi_horizon=False):
    try:
        return create_sliding_windows(val_r, val_v, seq_len=seq_len, horizon=1, multi_horizon=multi_horizon)
    except ValueError:
        context_r = pd.concat([train_r.iloc[-seq_len:], val_r], axis=0)
        context_v = pd.concat([train_v.iloc[-seq_len:], val_v], axis=0)
        return create_sliding_windows(context_r, context_v, seq_len=seq_len, horizon=1, multi_horizon=multi_horizon)


def _build_target_matrix_point_in_time(
    test_returns,
    max_horizon,
    rolling_window=ROLLING_VOL_WINDOW,
    history_returns=None,
):
    """
    Build point-in-time volatility targets for horizons 1..max_horizon.

    Target definition for anchor i and horizon h:
        target(i, h) = rolling_std_rolling_window(test_returns.shift(1))[i + h]

    This matches the realized-vol formula over past returns only:
        sigma_t = sqrt((1 / w) * sum_{k=1..w} (r_{t-k} - r_bar)^2)
    implemented with population std (ddof=0).
    """
    test_returns = np.asarray(test_returns, dtype=float)
    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1")

    n_test = test_returns.size
    n_anchors = n_test - max_horizon
    if n_anchors <= 0:
        return np.empty((0, max_horizon), dtype=float), 0

    if history_returns is not None and len(history_returns) > 0:
        history_returns = np.asarray(history_returns, dtype=float)
        full_returns = np.concatenate([history_returns, test_returns])
        rolling_full = (
            pd.Series(full_returns)
            .shift(1)
            .rolling(window=int(rolling_window))
            .std(ddof=0)
            .to_numpy(dtype=float)
        )
        rolling_vol = rolling_full[-n_test:]
    else:
        rolling_vol = (
            pd.Series(test_returns)
            .shift(1)
            .rolling(window=int(rolling_window))
            .std(ddof=0)
            .to_numpy(dtype=float)
        )

    target_matrix = np.asarray(
        [rolling_vol[i + 1 : i + max_horizon + 1] for i in range(n_anchors)],
        dtype=float,
    )

    if np.isnan(target_matrix).any():
        nan_count = int(np.isnan(target_matrix).sum())
        raise ValueError(
            "Target matrix contains NaN values "
            f"(count={nan_count}). Ensure rolling context is available."
        )

    return target_matrix, n_anchors


def _build_pred_matrix_point_in_time(pred_var_1d, max_horizon, n_anchors, horizons_idx=None):
    """
    Build point-in-time volatility forecasts for horizons 1..max_horizon.

    Args:
        pred_var_1d: 
            - rolling mode: 1D array of single-step predictions
            - one-shot mode: 1D array of tuples, each tuple has 5 values for 5 horizons
        max_horizon: max horizon (21)
        n_anchors: number of anchors
        horizons_idx: indices of selected horizons [1, 3, 5, 10, 21] (for extracting from full 21)
    
    Forecast definition for anchor i and horizon h:
        rolling: pred(i, h) = sqrt(pred_var[i + h])
        one-shot: pred(i, h_idx) = pred_var[i][h_idx]  (already has all horizons)
    """
    # Check if one-shot mode (predictions are tuples)
    is_one_shot = (
        len(pred_var_1d) > 0 
        and isinstance(pred_var_1d[0], tuple) 
        and horizons_idx is not None
    )
    
    if is_one_shot:
        # One-shot multi-horizon: each prediction already has all 5 horizons
        # pred_var_1d[i] is a tuple of (v1, v3, v5, v10, v21)
        # horizons_idx are the indices to extract from tuple (typically [0,1,2,3,4] for all 5)
        if n_anchors <= 0:
            return np.empty((0, len(horizons_idx)), dtype=float)
        
        # Build matrix from tuple predictions
        matrix = []
        for i in range(min(n_anchors, len(pred_var_1d))):
            pred_tuple = pred_var_1d[i]
            row = [np.sqrt(max(float(pred_tuple[h_idx]), 1e-8)) for h_idx in horizons_idx]
            matrix.append(row)
        
        return np.asarray(matrix, dtype=float) if matrix else np.empty((0, len(horizons_idx)), dtype=float)
    else:
        # Rolling mode: build matrix from 1D predictions using indexing
        pred_var_1d = np.maximum(np.asarray(pred_var_1d, dtype=float), 1e-8)
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        if n_anchors <= 0:
            return np.empty((0, max_horizon), dtype=float)

        required_len = n_anchors + max_horizon
        if pred_var_1d.size < required_len:
            return np.empty((0, max_horizon), dtype=float)

        anchor_idx = np.arange(n_anchors, dtype=int)[:, None]
        horizon_offsets = np.arange(1, max_horizon + 1, dtype=int)[None, :]
        return np.sqrt(pred_var_1d[anchor_idx + horizon_offsets])


def _train_non_stat_models(
    train_r,
    train_v,
    val_r,
    val_v,
    seq_len,
    epochs,
    batch_size,
    device,
    num_workers,
    log_progress=False,
    dataset_name="",
    seed=None,
    multi_horizon_mode="rolling",
):
    # Determine if we're using one-shot multi-horizon
    is_one_shot = (multi_horizon_mode == "one_shot")
    num_horizons_out = 5 if is_one_shot else 1
    
    train_windows = create_sliding_windows(
        train_r, train_v, seq_len=seq_len, horizon=1, multi_horizon=is_one_shot
    )
    val_windows = _build_val_windows(
        train_r, train_v, val_r, val_v, seq_len=seq_len, 
        multi_horizon=is_one_shot
    )
    train_loader, val_loader = build_dataloaders(
        train_windows,
        val_windows,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    trained_dl_models = {}
    train_time_by_model = {}
    for model_name in SUPPORTED_DL_MODELS:
        _log(
            f"[{dataset_name}][seed={seed}] Training {model_name}...",
            enabled=log_progress,
        )
        train_start = time.perf_counter()
        model = build_dl_model(
            model_name=model_name, seq_len=seq_len, num_horizons_out=num_horizons_out
        )
        model, history = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=HYPERPARAMETER_CONFIG["learning_rate"],
            lr_factor=HYPERPARAMETER_CONFIG["lr_factor"],
            lr_patience=HYPERPARAMETER_CONFIG["lr_patience"],
            early_stopping_patience=HYPERPARAMETER_CONFIG["early_stopping_patience"],
            min_lr=HYPERPARAMETER_CONFIG["min_lr"],
        )
        duration = time.perf_counter() - train_start
        best_val_loss = float(np.min(history["val_loss"])) if history["val_loss"] else np.nan
        _log(
            (
                f"[{dataset_name}][seed={seed}] Done {model_name} "
                f"| epochs={len(history['train_loss'])} "
                f"| best_val_loss={best_val_loss:.6f} "
                f"| time={duration:.1f}s"
            ),
            enabled=log_progress,
        )
        trained_dl_models[model_name] = model
        train_time_by_model[model_name] = duration

    # For one-shot multi-horizon, skip GARCH-LSTM-Hybrid (needs architectural changes)
    if is_one_shot:
        _log(
            f"[{dataset_name}][seed={seed}] Skipping {HYBRID_MODEL_NAME} (one-shot mode)",
            enabled=log_progress,
        )
        hybrid = None
    else:
        _log(
            f"[{dataset_name}][seed={seed}] Training {HYBRID_MODEL_NAME}...",
            enabled=log_progress,
        )
        hybrid_start = time.perf_counter()
        hybrid = GARCHLSTMHybrid(hidden_size=16)
        hybrid, hybrid_history = train_garch_lstm_hybrid(
            model=hybrid,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=HYPERPARAMETER_CONFIG["learning_rate"],
            lr_factor=HYPERPARAMETER_CONFIG["lr_factor"],
            lr_patience=HYPERPARAMETER_CONFIG["lr_patience"],
            early_stopping_patience=HYPERPARAMETER_CONFIG["early_stopping_patience"],
            min_lr=HYPERPARAMETER_CONFIG["min_lr"],
        )
        hybrid_duration = time.perf_counter() - hybrid_start
        hybrid_best_val = (
            float(np.min(hybrid_history["val_loss"])) if hybrid_history["val_loss"] else np.nan
        )
        _log(
            (
                f"[{dataset_name}][seed={seed}] Done {HYBRID_MODEL_NAME} "
                f"| epochs={len(hybrid_history['train_loss'])} "
                f"| best_val_loss={hybrid_best_val:.6f} "
                f"| time={hybrid_duration:.1f}s"
            ),
            enabled=log_progress,
        )

        train_time_by_model[HYBRID_MODEL_NAME] = hybrid_duration
    
    return trained_dl_models, hybrid, train_time_by_model


def _collect_one_step_predictions(
    train_r,
    train_v,
    test_r,
    test_v,
    dl_models,
    hybrid,
    seq_len,
    device,
    log_progress=False,
    dataset_name="",
    seed=None,
):
    preds_var = {}
    stat_fit_time_by_model = {}

    for model_name in STAT_MODELS:
        _log(
            f"[{dataset_name}][seed={seed}] Forecasting with {model_name}...",
            enabled=log_progress,
        )
        start_t = time.perf_counter()
        preds_var[model_name] = rolling_stat_forecast_variance(
            train_returns=train_r,
            test_returns=test_r,
            model_name=model_name,
            dist="t",
        )
        duration = time.perf_counter() - start_t
        stat_fit_time_by_model[model_name] = duration
        _log(
            (
                f"[{dataset_name}][seed={seed}] Done {model_name} forecast "
                f"| n_pred={len(preds_var[model_name])} | time={duration:.1f}s"
            ),
            enabled=log_progress,
        )

    for model_name, model in dl_models.items():
        _log(
            f"[{dataset_name}][seed={seed}] Forecasting with {model_name}...",
            enabled=log_progress,
        )
        start_t = time.perf_counter()
        preds_var[model_name] = rolling_dl_forecast_variance(
            model=model,
            train_returns=train_r,
            train_variance=train_v,
            test_returns=test_r,
            test_variance=test_v,
            seq_len=seq_len,
            device=device,
        )
        duration = time.perf_counter() - start_t
        _log(
            (
                f"[{dataset_name}][seed={seed}] Done {model_name} forecast "
                f"| n_pred={len(preds_var[model_name])} | time={duration:.1f}s"
            ),
            enabled=log_progress,
        )

    # For one-shot mode, hybrid will be None
    if hybrid is not None:
        _log(
            f"[{dataset_name}][seed={seed}] Forecasting with {HYBRID_MODEL_NAME}...",
            enabled=log_progress,
        )
        hybrid_start_t = time.perf_counter()
        preds_var[HYBRID_MODEL_NAME] = rolling_hybrid_forecast_variance(
            model=hybrid,
            train_returns=train_r,
            train_variance=train_v,
            test_returns=test_r,
            test_variance=test_v,
            seq_len=seq_len,
            device=device,
        )
        hybrid_duration = time.perf_counter() - hybrid_start_t
        _log(
            (
                f"[{dataset_name}][seed={seed}] Done {HYBRID_MODEL_NAME} forecast "
                f"| n_pred={len(preds_var[HYBRID_MODEL_NAME])} | time={hybrid_duration:.1f}s"
            ),
            enabled=log_progress,
        )

    return preds_var, stat_fit_time_by_model


def run_benchmark(
    dataset_dir=None,
    seq_len=DEFAULT_SEQ_LEN,
    epochs=60,
    batch_size=64,
    output_csv=None,
    device=None,
    num_workers=None,
    log_progress=True,
    split_mode="ratio",
    date_start=None,
    date_end=None,
    num_seeds=5,  # Number of seeds to use (default 5, set to 1 for smoke tests)
):
    if dataset_dir is None:
        dataset_dir = get_default_dataset_dir()
    dataset_dir = Path(dataset_dir)

    runtime_device = resolve_device(device)
    if num_workers is None:
        num_workers = 2 if runtime_device.type == "cuda" else 0

    dataset_files = _discover_dataset_files(dataset_dir)
    if not dataset_files:
        raise RuntimeError(f"No usable CSV files found in {dataset_dir}")

    _log(
        (
            f"Benchmark started | device={runtime_device} | datasets={len(dataset_files)} "
            f"| seeds={num_seeds} | horizons={HORIZONS}"
        ),
        enabled=log_progress,
    )

    rows = []
    detailed_rows = []

    for dataset_csv in dataset_files:
        dataset_start = time.perf_counter()
        _log(
            f"Dataset start: {dataset_csv.name}",
            enabled=log_progress,
        )
        close = load_close_series(dataset_csv)
        train_split, val_split, test_split = prepare_aaai24_data(
            close,
            split_mode=split_mode,
            dataset_name=dataset_csv.stem,
            date_start=date_start,
            date_end=date_end,
        )

        train_r, train_v = train_split
        val_r, val_v = val_split
        test_r, test_v = test_split

        train_r_np = train_r.to_numpy(dtype=float)
        train_v_np = train_v.to_numpy(dtype=float)
        test_r_np = test_r.to_numpy(dtype=float)
        test_v_np = test_v.to_numpy(dtype=float)
        test_time_index = test_r.index
        nu_test = _get_dataset_nu_test(dataset_csv.stem)

        max_horizon = max(HORIZONS)
        history_returns_np = np.concatenate([train_r_np, val_r.to_numpy(dtype=float)])
        target_matrix, n_anchors = _build_target_matrix_point_in_time(
            test_returns=test_r_np,
            max_horizon=max_horizon,
            rolling_window=ROLLING_VOL_WINDOW,
            history_returns=history_returns_np,
        )
        if n_anchors <= 0:
            _log(
                f"[{dataset_csv.stem}] Skipped: not enough test samples for max horizon={max_horizon}",
                enabled=log_progress,
            )
            continue

        for seed in SEEDS[:num_seeds]:
            seed_start = time.perf_counter()
            set_global_seed(seed)
            _log(
                f"[{dataset_csv.stem}] Seed {seed} started",
                enabled=log_progress,
            )

            dl_models, hybrid, train_time_by_model = _train_non_stat_models(
                train_r=train_r,
                train_v=train_v,
                val_r=val_r,
                val_v=val_v,
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                device=str(runtime_device),
                num_workers=num_workers,
                log_progress=log_progress,
                dataset_name=dataset_csv.stem,
                seed=seed,
                multi_horizon_mode=MULTI_HORIZON_MODE,
            )

            preds_var, stat_fit_time_by_model = _collect_one_step_predictions(
                train_r=train_r_np,
                train_v=train_v_np,
                test_r=test_r_np,
                test_v=test_v_np,
                dl_models=dl_models,
                hybrid=hybrid,
                seq_len=seq_len,
                device=str(runtime_device),
                log_progress=log_progress,
                dataset_name=dataset_csv.stem,
                seed=seed,
            )

            # For stat models, fitting happens inside rolling forecast.
            time_train_by_model = dict(train_time_by_model)
            for stat_model, stat_fit_time in stat_fit_time_by_model.items():
                time_train_by_model[stat_model] = stat_fit_time

            rows_before = len(rows)
            for model_name, pred_var_1d in preds_var.items():
                # For one-shot mode, pass horizons_idx to extract from tuple predictions
                horizons_idx = [0, 1, 2, 3, 4] if MULTI_HORIZON_MODE == "one_shot" else None
                pred_matrix = _build_pred_matrix_point_in_time(
                    pred_var_1d,
                    max_horizon=max_horizon,
                    n_anchors=n_anchors,
                    horizons_idx=horizons_idx,
                )
                if pred_matrix.size == 0:
                    continue

                for horizon in HORIZONS:
                    h_idx = horizon - 1
                    pred_vol_h = pred_matrix[:, h_idx]
                    true_vol_h = target_matrix[:, h_idx]

                    n_eval = n_anchors
                    target_times = test_time_index[horizon : horizon + n_eval]
                    returns_eval_h = test_r_np[horizon : horizon + n_eval]

                    metric_values = compute_metrics(
                        true_vol_h,
                        pred_vol_h,
                        returns_eval=returns_eval_h,
                        nu=nu_test,
                    )
                    rows.append(
                        {
                            "Dataset": dataset_csv.stem,
                            "Seed": seed,
                            "Horizon": horizon,
                            "Model": model_name,
                            "MAE": metric_values["MAE"],
                            "MSE": metric_values["MSE"],
                            "QLIKE": metric_values["QLIKE"],
                            "Violation_Rate": metric_values["Violation_Rate"],
                            "Kupiec_LR": metric_values["Kupiec_LR"],
                            "Kupiec_p": metric_values["Kupiec_p"],
                            "LR_Ind": metric_values["LR_Ind"],
                            "N_eval": n_eval,
                            "Seq_len": seq_len,
                        }
                    )

                    for i in range(n_eval):
                        detailed_rows.append(
                            {
                                "time": str(target_times[i]),
                                "dataset": dataset_csv.stem,
                                "model": model_name,
                                "horizon": horizon,
                                "True_Volatility": float(true_vol_h[i]),
                                "Pred_Volatility": float(pred_vol_h[i]),
                                "return_1_day": float(returns_eval_h[i]),
                            }
                        )

            produced = len(rows) - rows_before
            seed_duration = time.perf_counter() - seed_start
            _log(
                (
                    f"[{dataset_csv.stem}] Seed {seed} finished "
                    f"| rows_added={produced} | time={seed_duration:.1f}s"
                ),
                enabled=log_progress,
            )

            del dl_models, hybrid, preds_var
            if runtime_device.type == "cuda":
                torch.cuda.empty_cache()

        dataset_duration = time.perf_counter() - dataset_start
        _log(
            f"Dataset done: {dataset_csv.name} | time={dataset_duration:.1f}s",
            enabled=log_progress,
        )

    results_df = pd.DataFrame(rows)
    detailed_df = pd.DataFrame(detailed_rows)

    if output_csv is None:
        output_csv = Path(__file__).resolve().parent / "results" / "model_results.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mean_seed_predictions_csv = output_csv.with_name("predictions.csv")

    if detailed_df.empty:
        mean_seed_predictions_df = pd.DataFrame(
            columns=[
                "time",
                "dataset",
                "model",
                "horizon",
                "True_Volatility",
                "Pred_Volatility",
                "return_1_day",
            ]
        )
    else:
        mean_seed_predictions_df = (
            detailed_df.groupby(["time", "dataset", "model", "horizon"], as_index=False, sort=False)
            .agg({
                "True_Volatility": "mean",
                "Pred_Volatility": "mean",
                "return_1_day": "first"
            })
            .sort_values(["dataset", "model", "horizon", "time"])
        )

    mean_seed_predictions_df.to_csv(mean_seed_predictions_csv, index=False)
    _log(
        f"Benchmark finished | total_rows={len(results_df)}",
        enabled=log_progress,
    )
    _log(
        (
            "Mean-seed predictions saved "
            f"| rows={len(mean_seed_predictions_df)} | saved={mean_seed_predictions_csv}"
        ),
        enabled=log_progress,
    )
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run AAAI24 benchmark reproduction")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging")
    parser.add_argument("--split-mode", type=str, default="ratio", choices=["ratio", "fixed_counts"],
                        help="Split mode: 'ratio' (8:1:1) or 'fixed_counts' (predefined table)")
    parser.add_argument("--date-start", type=str, default=None, help="Start date for filtering (e.g., 2010-01-01)")
    parser.add_argument("--date-end", type=str, default=None, help="End date for filtering (e.g., 2025-12-31)")
    args = parser.parse_args()

    results_df = run_benchmark(
        dataset_dir=args.dataset_dir,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        device=args.device,
        num_workers=args.num_workers,
        log_progress=not args.quiet,
        split_mode=args.split_mode,
        date_start=args.date_start,
        date_end=args.date_end,
    )

    if results_df.empty:
        print("No benchmark rows were produced.")
    else:
        summary = (
            results_df.groupby(["Model", "Horizon"], as_index=False)[
                ["MAE", "MSE", "QLIKE", "Violation_Rate", "Kupiec_LR", "Kupiec_p", "LR_Ind"]
            ]
            .mean()
            .sort_values(["Model", "Horizon"])
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
