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


def _log(message, enabled=True):
    if not enabled:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


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


def _build_val_windows(train_r, train_v, val_r, val_v, seq_len):
    try:
        return create_sliding_windows(val_r, val_v, seq_len=seq_len, horizon=1)
    except ValueError:
        context_r = pd.concat([train_r.iloc[-seq_len:], val_r], axis=0)
        context_v = pd.concat([train_v.iloc[-seq_len:], val_v], axis=0)
        return create_sliding_windows(context_r, context_v, seq_len=seq_len, horizon=1)


def _compute_horizon_targets(test_returns, horizon):
    test_returns = np.asarray(test_returns, dtype=float)
    n_eval = len(test_returns) - horizon
    if n_eval <= 0:
        return np.asarray([]), np.asarray([])

    future_returns = np.asarray(
        [np.nansum(test_returns[i + 1 : i + 1 + horizon]) for i in range(n_eval)],
        dtype=float,
    )
    realized_vol_h = np.asarray(
        [
            np.sqrt(np.nansum(np.square(test_returns[i + 1 : i + 1 + horizon])))
            for i in range(n_eval)
        ],
        dtype=float,
    )
    return future_returns, realized_vol_h


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
):
    train_windows = create_sliding_windows(train_r, train_v, seq_len=seq_len, horizon=1)
    val_windows = _build_val_windows(train_r, train_v, val_r, val_v, seq_len=seq_len)
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
        model = build_dl_model(model_name=model_name, seq_len=seq_len)
        model, history = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=1e-2,
            lr_factor=0.5,
            lr_patience=5,
            early_stopping_patience=20,
            min_lr=1e-6,
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
        learning_rate=1e-2,
        lr_factor=0.5,
        lr_patience=5,
        early_stopping_patience=20,
        min_lr=1e-6,
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
            f"| seeds={len(SEEDS)} | horizons={HORIZONS}"
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
        train_split, val_split, test_split = prepare_aaai24_data(close)

        train_r, train_v = train_split
        val_r, val_v = val_split
        test_r, test_v = test_split

        train_r_np = train_r.to_numpy(dtype=float)
        train_v_np = train_v.to_numpy(dtype=float)
        test_r_np = test_r.to_numpy(dtype=float)
        test_v_np = test_v.to_numpy(dtype=float)
        test_time_index = test_r.index

        for seed in SEEDS:
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
            for horizon in HORIZONS:
                _, realized_vol_h = _compute_horizon_targets(test_r_np, horizon)
                if realized_vol_h.size == 0:
                    continue

                for model_name, pred_var_1d in preds_var.items():
                    pred_std_1d = np.sqrt(np.maximum(pred_var_1d, 1e-8))

                    n_eval = min(pred_std_1d.size, realized_vol_h.size)
                    if n_eval <= 0:
                        continue

                    pred_vol_h = pred_std_1d[:n_eval] * np.sqrt(horizon)
                    true_vol_h = realized_vol_h[:n_eval]
                    target_times = test_time_index[horizon : horizon + n_eval]

                    metric_values = compute_metrics(true_vol_h, pred_vol_h)
                    rows.append(
                        {
                            "Dataset": dataset_csv.stem,
                            "Seed": seed,
                            "Horizon": horizon,
                            "Model": model_name,
                            "MAE": metric_values["MAE"],
                            "MSE": metric_values["MSE"],
                            "N_eval": n_eval,
                            "Seq_len": seq_len,
                        }
                    )

                    train_time = float(time_train_by_model.get(model_name, np.nan))
                    for i in range(n_eval):
                        detailed_rows.append(
                            {
                                "time": str(target_times[i]),
                                "dataset": dataset_csv.stem,
                                "model": model_name,
                                "horizon": horizon,
                                "seed": seed,
                                "True_Volatility": float(true_vol_h[i]),
                                "Pred_Volatility": float(pred_vol_h[i]),
                                "time_train": train_time,
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
        output_csv = Path(__file__).resolve().parent / "results" / "benchmark_results.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_csv, index=False)
    mean_seed_predictions_csv = output_csv.with_name(
        output_csv.stem + "_predictions_mean_seed.csv"
    )

    if detailed_df.empty:
        mean_seed_predictions_df = pd.DataFrame(
            columns=[
                "time",
                "dataset",
                "model",
                "horizon",
                "True_Volatility",
                "Pred_Volatility",
                "time_train",
            ]
        )
    else:
        mean_seed_predictions_df = (
            detailed_df.groupby(["time", "dataset", "model", "horizon"], as_index=False)[
                ["True_Volatility", "Pred_Volatility", "time_train"]
            ]
            .mean()
            .sort_values(["dataset", "model", "horizon", "time"])
        )

    mean_seed_predictions_df.to_csv(mean_seed_predictions_csv, index=False)
    _log(
        f"Benchmark finished | total_rows={len(results_df)} | saved={output_csv}",
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
    )

    if results_df.empty:
        print("No benchmark rows were produced.")
    else:
        summary = (
            results_df.groupby(["Model", "Horizon"], as_index=False)[["MAE", "MSE"]]
            .mean()
            .sort_values(["Model", "Horizon"])
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
