import copy
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

import ultility.data_loader as data_loader
import ultility.metrics as metrics
import ultility.models_garch as models_garch
import ultility.models_lstm_baseline as models_lstm_baseline
import ultility.models_transformer as models_transformer
import ultility.var_calculator as var_calculator
from ultility.lstmgarch import LSTMGARCH
from ultility.transformer_garch import TransformerGARCH


def _require_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pipeline. No GPU detected.")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def create_sequences(data, seq_len):
    sequences = []
    for index in range(len(data) - seq_len):
        sequences.append(data[index : index + seq_len])
    return np.array(sequences, dtype=np.float32)


def lstm_baseline_forecast(train_data, val_data, test_data, seq_len=60, epochs=80):
    device = str(_require_cuda_device())
    model, scaler, _, _ = models_lstm_baseline.train_lstm_baseline(
        train_data,
        val_data,
        seq_len=seq_len,
        epochs=epochs,
        batch_size=64,
        learning_rate=1e-3,
        device=device,
    )
    return models_lstm_baseline.predict_lstm_baseline_rolling(
        model,
        train_data,
        test_data,
        seq_len=seq_len,
        scaler=scaler,
        device=device,
    )


def transformer_forecast(train_data, test_data, seq_len=60, epochs=100):
    return np.abs(models_transformer.train_transformer(np.abs(train_data), np.abs(test_data), seq_len=seq_len, epochs=epochs))


def transformer_garch_forecast(train_data, test_data, seq_len=60, epochs=50):
    device = _require_cuda_device()
    train_seq = create_sequences(train_data, seq_len)
    if len(train_seq) == 0:
        raise ValueError("Not enough data for Transformer-GARCH sequences")

    train_tensor = torch.tensor(train_seq, dtype=torch.float32)
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True, pin_memory=True)
    model = TransformerGARCH().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()

    model.eval()
    history = list(train_data)
    predictions = []
    with torch.no_grad():
        for value in test_data:
            if len(history) < seq_len:
                history.append(value)
                continue
            sequence = torch.tensor(history[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            _, sigma2_sequence = model(sequence)
            sigma_prediction = torch.sqrt(sigma2_sequence[:, -1]).cpu().numpy()[0]
            predictions.append(sigma_prediction)
            history.append(value)

    return np.array(predictions)


def _build_lstm_garch_model(hidden_dim):
    return LSTMGARCH(hidden_dim=hidden_dim)


def _val_loss_for_model(model, train_data, val_data, seq_len, device, batch_size=64):
    if val_data is None or len(val_data) == 0:
        return None

    context = np.concatenate([train_data[-seq_len:], val_data]) if len(train_data) >= seq_len else np.concatenate([train_data, val_data])
    val_sequences = create_sequences(context, seq_len)
    if len(val_sequences) == 0:
        return None

    loader = DataLoader(torch.tensor(val_sequences, dtype=torch.float32), batch_size=batch_size, shuffle=False, pin_memory=True)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            loss, _ = model(batch)
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else None


def train_lstm_garch_with_validation(
    train_data,
    val_data,
    seq_len=60,
    hidden_dim=16,
    learning_rate=1e-3,
    weight_decay=0.0,
    epochs=200,
    batch_size=64,
    lr_factor=0.5,
    lr_patience=5,
    early_stopping_patience=20,
    min_lr=1e-6,
):
    device = _require_cuda_device()
    train_sequences = create_sequences(train_data, seq_len)
    if len(train_sequences) == 0:
        raise ValueError("Not enough data for LSTM-GARCH sequences")

    train_loader = DataLoader(
        torch.tensor(train_sequences, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    model = _build_lstm_garch_model(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
        val_loss = _val_loss_for_model(model, train_data, val_data, seq_len, device, batch_size=batch_size)
        if val_loss is None:
            val_loss = train_loss

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "history": history,
    }


def tune_lstm_garch_hyperparams(
    train_data,
    val_data,
    seq_len=60,
    search_space=None,
    epochs=200,
    batch_size=64,
    lr_factor=0.5,
    lr_patience=5,
    early_stopping_patience=20,
    min_lr=1e-6,
):
    if search_space is None:
        search_space = {
            "hidden_dim": [8, 16],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [0.0],
        }

    best = None
    keys = ["hidden_dim", "learning_rate", "weight_decay"]
    grid = [search_space.get(key, [None]) for key in keys]

    for hidden_dim, learning_rate, weight_decay in product(*grid):
        model, info = train_lstm_garch_with_validation(
            train_data=train_data,
            val_data=val_data,
            seq_len=seq_len,
            hidden_dim=int(hidden_dim),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            epochs=epochs,
            batch_size=batch_size,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            early_stopping_patience=early_stopping_patience,
            min_lr=min_lr,
        )

        candidate = {
            "model": model,
            "info": info,
            "val_loss": info["best_val_loss"],
        }
        if best is None or candidate["val_loss"] < best["val_loss"]:
            best = candidate

    if best is None:
        raise RuntimeError("LSTM-GARCH tuning failed")

    return best


def lstm_garch_forecast(
    train_data,
    val_data,
    test_data,
    seq_len=60,
    search_space=None,
    epochs=200,
    batch_size=64,
    lr_factor=0.5,
    lr_patience=5,
    early_stopping_patience=20,
    min_lr=1e-6,
    return_best_params=False,
):
    device = _require_cuda_device()
    tuned = tune_lstm_garch_hyperparams(
        train_data=train_data,
        val_data=val_data,
        seq_len=seq_len,
        search_space=search_space,
        epochs=epochs,
        batch_size=batch_size,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        early_stopping_patience=early_stopping_patience,
        min_lr=min_lr,
    )
    model = tuned["model"]
    model.eval()

    history = list(train_data)
    predictions = []
    with torch.no_grad():
        for value in test_data:
            if len(history) < seq_len:
                history.append(value)
                continue
            sequence = torch.tensor(history[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            _, sigma2_sequence = model(sequence)
            sigma_prediction = torch.sqrt(sigma2_sequence[:, -1]).cpu().numpy()[0]
            predictions.append(sigma_prediction)
            history.append(value)

    predictions = np.array(predictions)
    if return_best_params:
        return predictions, tuned["info"]
    return predictions


def run_benchmark(datasets, split_df, window_size=60, seq_len=60, confidence_level=0.95):
    _require_cuda_device()
    results = []

    models_to_run = {
        "GARCH": lambda train, val, test: models_garch.rolling_garch_forecast(train, test, model_name="GARCH"),
        "GJR-GARCH": lambda train, val, test: models_garch.rolling_garch_forecast(train, test, model_name="GJR-GARCH"),
        "Transformer": lambda train, val, test: transformer_forecast(train, test, seq_len=seq_len, epochs=100),
        "LSTM-Baseline": lambda train, val, test: lstm_baseline_forecast(train, val, test, seq_len=seq_len, epochs=80),
        "LSTM-GARCH": lambda train, val, test: lstm_garch_forecast(
            train,
            val,
            test,
            seq_len=seq_len,
            search_space={
                "hidden_dim": [8, 16],
                "learning_rate": [1e-3, 5e-4],
                "weight_decay": [0.0],
            },
            epochs=120,
            batch_size=64,
            lr_factor=0.5,
            lr_patience=5,
            early_stopping_patience=20,
            min_lr=1e-6,
        ),
        "Transformer-GARCH": lambda train, val, test: transformer_garch_forecast(train, test, seq_len=seq_len, epochs=50),
    }

    var_calc = var_calculator.RollingVaRCalculator(
        confidence_level=confidence_level,
        min_history=30,
        refit_frequency=5,
    )

    for dataset_name, series in tqdm.tqdm(datasets.items(), desc="Processing Datasets"):
        try:
            train_data, val_data, test_data = data_loader.get_splits_for_dataset(series, split_df, dataset_name, seq_len)

            for model_name, model_func in models_to_run.items():
                started_at = None
                if model_name in ("Transformer", "Transformer-GARCH"):
                    tqdm.tqdm.write(f"[{dataset_name}] Start {model_name}")
                    started_at = time.perf_counter()

                predictions = model_func(train_data, val_data, test_data)

                if started_at is not None:
                    tqdm.tqdm.write(f"[{dataset_name}] Done {model_name} ({time.perf_counter() - started_at:.2f}s)")

                returns_eval = test_data[-len(predictions):]
                var_estimates = var_calc.compute_var_rolling_window(returns_eval, predictions, confidence_level=confidence_level)

                valid_mask = (~np.isnan(var_estimates)) & (~np.isnan(returns_eval)) & (~np.isnan(predictions))
                if not valid_mask.any():
                    continue

                returns_valid = returns_eval[valid_mask]
                predictions_valid = predictions[valid_mask]
                var_valid = var_estimates[valid_mask]

                violation_rate, kupiec_lr, kupiec_p = var_calculator.compute_kupiec_test(
                    returns_valid,
                    var_valid,
                    confidence_level=confidence_level,
                )
                traffic_light, cum_violations = var_calculator.compute_traffic_light_test(
                    returns_valid,
                    var_valid,
                    confidence_level=confidence_level,
                )
                lr_ind, p_ind = var_calculator.compute_christoffersen_independence(
                    returns_valid,
                    var_valid,
                    confidence_level=confidence_level,
                )

                realized_vol = np.abs(returns_valid)
                mse, qlike = metrics.compute_mse_qlike(realized_vol, predictions_valid)

                results.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "MSE": mse,
                        "QLIKE": qlike,
                        "Violation_Rate": violation_rate,
                        "Kupiec_LR": kupiec_lr,
                        "Kupiec_p": kupiec_p,
                        "LR_Ind": lr_ind,
                        "p_Ind": p_ind,
                        "Traffic_Light": traffic_light,
                        "Cum_Violations": cum_violations,
                        "Mean_Nu": "N/A",
                    }
                )

        except (ValueError, IndexError) as error:
            print(f"Skipping dataset {dataset_name} due to error: {error}")
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["Dataset", "Model"]).reset_index(drop=True)
        for column in ["MSE", "QLIKE", "Violation_Rate", "Kupiec_LR", "Kupiec_p", "LR_Ind", "p_Ind", "Mean_Nu"]:
            if column in results_df.columns:
                results_df[column] = results_df[column].apply(
                    lambda value: f"{value:.4f}" if isinstance(value, (int, float)) and not np.isnan(value) else (
                        "N/A" if isinstance(value, float) and np.isnan(value) else value
                    )
                )

    return results_df
