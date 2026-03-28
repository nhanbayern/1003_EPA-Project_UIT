import pandas as pd
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader

import ultility.data_loader as data_loader
import ultility.models_garch as models_garch
import ultility.models_transformer as models_transformer
import ultility.models_lstm_baseline as models_lstm_baseline
import ultility.var_calculator as var_calculator
from ultility.lstmgarch import LSTMGARCH


def lstm_baseline_forecast(train_data, val_data, test_data, seq_len=60, epochs=80):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler, _, _ = models_lstm_baseline.train_lstm_baseline(
        train_data,
        val_data,
        seq_len=seq_len,
        epochs=epochs,
        batch_size=64,
        learning_rate=1e-3,
        device=device,
    )
    preds = models_lstm_baseline.predict_lstm_baseline_rolling(
        model,
        train_data,
        test_data,
        seq_len=seq_len,
        scaler=scaler,
        device=device,
    )
    return preds


def create_sequences(data, seq_len):
    xs = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
    return np.array(xs, dtype=np.float32)


def lstm_garch_forecast(train_data, test_data, seq_len=60, epochs=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    train_seq = create_sequences(train_data, seq_len)
    if len(train_seq) == 0:
        raise ValueError("Not enough data for LSTM-GARCH sequences")
    train_tensor = torch.tensor(train_seq, dtype=torch.float32)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True, pin_memory=pin)
    model = LSTMGARCH().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            opt.step()
    model.eval()
    history = list(train_data)
    preds = []
    with torch.no_grad():
        for x in test_data:
            if len(history) < seq_len:
                history.append(x)
                continue
            seq = torch.tensor(history[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            _, sigma2_seq = model(seq)
            sigma_pred = torch.sqrt(sigma2_seq[:, -1]).cpu().numpy()[0]
            preds.append(sigma_pred)
            history.append(x)
    return np.array(preds)


def run_benchmark(datasets, split_df, window_size=60, seq_len=60, confidence_level=0.95):
    all_results = []

    models_to_run = {
        "GARCH": lambda tr, val, te: models_garch.rolling_garch_forecast(tr, te, model_name="GARCH"),
        "GJR-GARCH": lambda tr, val, te: models_garch.rolling_garch_forecast(tr, te, model_name="GJR-GARCH"),
        "Transformer": lambda tr, val, te: models_transformer.train_transformer(np.abs(tr), np.abs(te), seq_len=seq_len, epochs=100),
        "LSTM-Baseline": lambda tr, val, te: lstm_baseline_forecast(tr, val, te, seq_len=seq_len, epochs=80),
        "LSTM-GARCH": lambda tr, val, te: lstm_garch_forecast(tr, te, seq_len=seq_len, epochs=50),
    }

    var_calc = var_calculator.RollingVaRCalculator(
        confidence_level=confidence_level,
        min_history=30,
        refit_frequency=5,
    )

    for name, series in tqdm.tqdm(datasets.items(), desc="Processing Datasets"):
        try:
            train_data, val_data, test_data = data_loader.get_splits_for_dataset(series, split_df, name, seq_len)

            for model_name, model_func in models_to_run.items():
                vol_forecast = model_func(train_data, val_data, test_data)

                returns_eval = test_data[-len(vol_forecast):]

                var_estimates = var_calc.compute_var_rolling_window(
                    returns_eval, vol_forecast, confidence_level=confidence_level
                )

                valid_mask = (~np.isnan(var_estimates)) & (~np.isnan(returns_eval))
                if not valid_mask.any():
                    continue

                returns_valid = returns_eval[valid_mask]
                vol_valid = vol_forecast[valid_mask]
                var_valid = var_estimates[valid_mask]
                violation_rate, kupiec_lr, kupiec_p = var_calculator.compute_kupiec_test(
                    returns_valid, var_valid, confidence_level=confidence_level
                )

                traffic_light, cum_violations = var_calculator.compute_traffic_light_test(
                    returns_valid, var_valid, confidence_level=confidence_level
                )

                lr_ind, p_ind = var_calculator.compute_christoffersen_independence(
                    returns_valid, var_valid, confidence_level=confidence_level
                )

                realized_vol = np.abs(returns_valid)
                realized_var = returns_valid ** 2
                forecast_var = vol_valid ** 2

                mse = np.nanmean((realized_vol - vol_valid) ** 2)
                qlike = np.nanmean(np.log(forecast_var) + realized_var / np.maximum(forecast_var, 1e-8))

                result_row = {
                    "Dataset": name,
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
                all_results.append(result_row)

        except (ValueError, IndexError) as e:
            print(f"Skipping dataset {name} due to error: {e}")
            continue

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["Dataset", "Model"]).reset_index(drop=True)
        for col in ["MSE", "QLIKE", "Violation_Rate", "Kupiec_LR", "Kupiec_p", "LR_Ind", "p_Ind", "Mean_Nu"]:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else (
                        "N/A" if isinstance(x, float) and np.isnan(x) else x
                    )
                )

    return results_df