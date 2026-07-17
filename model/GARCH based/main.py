import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Local imports
from dataset import get_dataloaders, get_default_dataset_dir, prepare_series, load_close_series, get_split_indices
from losses import TLoss
from models.stat_models import evaluate_stat_model, MODEL_SPECS
from models.garch_lstm import GARCHLSTMHybrid, evaluate_garch_lstm
from train_evaluate import train_garch_lstm_hybrid

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_predictions_csv(index_name, model_name, predictions_dict, test_time, test_r, test_v, out_dir):
    """
    Saves predictions in the format: time, log_return, horizon, true_volatility, predict_volatility
    predictions_dict: {horizon: array_of_predicted_volatilities}
    """
    records = []
    
    # We only have predictions for timestamps starting from seq_len (or depending on the model offset)
    # For simplicity, ensure predictions align with the end of the test series.
    # predictions_dict[h] should have length equal to the number of test steps evaluated.
    
    horizons = sorted(list(predictions_dict.keys()))
    
    for h in horizons:
        preds = predictions_dict[h]
        # Align with the end of the test data
        valid_len = len(preds)
        
        # In case of seq_len offset for LSTM, we align with the last valid_len elements
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

def run_pipeline():
    dataset_dir = get_default_dataset_dir()
    csv_files = glob.glob(str(dataset_dir / "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {dataset_dir}")
        return
        
    out_base = Path("results")
    pred_dir = out_base / "predictions"
    viz_dir = out_base / "visualizations"
    model_dir = out_base / "model_params"
    
    ensure_dir(pred_dir)
    ensure_dir(viz_dir)
    ensure_dir(model_dir)
    
    horizons = [1, 3, 5, 10, 21]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for csv_file in csv_files:
        index_name = Path(csv_file).stem.upper()
        print(f"\n{'='*50}\nProcessing {index_name}\n{'='*50}")
        
        # 1. Load data
        train_loader, val_loader, test_r, test_v, test_time = get_dataloaders(csv_file, batch_size=32)
        
        # Re-fetch the raw train data for statistical models which need raw returns
        close_series = load_close_series(csv_file)
        returns, volatility = prepare_series(close_series)
        train_end, val_end = get_split_indices(len(returns), index_name)
        raw_train_r = returns.iloc[:train_end]
        raw_test_r = returns.iloc[train_end:] # Use val+test for rolling forecast history, but evaluate on test
        
        # We need the full history up to the test set for rolling forecast
        stat_train_r = returns.iloc[:val_end]
        stat_test_r = returns.iloc[val_end:]
        stat_test_time = returns.index[val_end:]
        stat_test_v = volatility.iloc[val_end:]
        
        # 2. Train and Evaluate Statistical Models
        for stat_model_name in MODEL_SPECS.keys():
            print(f"--- Running {stat_model_name} ---")
            predictions, params = evaluate_stat_model(
                stat_train_r, stat_test_r, 
                model_name=stat_model_name, 
                dist="t", 
                horizons=horizons
            )
            
            # Save predictions
            save_predictions_csv(index_name, stat_model_name, predictions, stat_test_time, stat_test_r, stat_test_v, pred_dir)
            plot_predictions(index_name, stat_model_name, predictions, stat_test_time, stat_test_v, viz_dir, horizon=21)
            
            # Save params
            param_df = pd.DataFrame(params)
            param_df.to_csv(model_dir / f"{index_name}_{stat_model_name}_params.csv", index=False)
            
        # 3. Train and Evaluate GARCH-LSTM Hybrid
        print(f"--- Running GARCH-LSTM Hybrid ---")
        model = GARCHLSTMHybrid(hidden_size=16)
        criterion = TLoss(v=5.0)
        
        model, history = train_garch_lstm_hybrid(
            model, criterion, train_loader, val_loader, 
            device=device, epochs=50 # Using 50 epochs for quicker Kaggle runs, can be adjusted
        )
        
        # Save model weights
        torch.save(model.state_dict(), model_dir / f"{index_name}_GARCH_LSTM_weights.pth")
        
        # Evaluate multi-horizon
        # test_r and test_v from get_dataloaders already include seq_len history before the test set
        lstm_predictions = evaluate_garch_lstm(
            model, test_r, test_v, 
            seq_len=60, horizons=horizons, device=device
        )
        
        save_predictions_csv(index_name, "GARCH-LSTM-Hybrid", lstm_predictions, test_time, test_r.iloc[60:], test_v.iloc[60:], pred_dir)
        plot_predictions(index_name, "GARCH-LSTM-Hybrid", lstm_predictions, test_time, test_v.iloc[60:], viz_dir, horizon=21)

if __name__ == "__main__":
    run_pipeline()
