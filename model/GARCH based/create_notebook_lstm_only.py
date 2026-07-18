import nbformat
from nbformat.v4 import new_notebook, new_code_cell

nb = new_notebook()

cells = []

# Cell 1: Setup
cells.append(new_code_cell("""\
!git clone -b kaggle-implementation --single-branch https://github.com/nhanbayern/1003_EPA-Project_UIT.git
!pip install arch -q
import sys
import os
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/kaggle/working/1003_EPA-Project_UIT/model/GARCH based')

# Base directories for results
os.makedirs('/kaggle/working/results/predictions', exist_ok=True)
os.makedirs('/kaggle/working/results/visualizations', exist_ok=True)
os.makedirs('/kaggle/working/results/model_params', exist_ok=True)
"""))

# Cell 2: Imports
cells.append(new_code_cell("""\
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import copy

from config import DEFAULT_SEQ_LEN, DEFAULT_VOL_WINDOW, HORIZONS, BATCH_SIZE, EPOCHS, LR, DEVICE, FIXED_SPLITS
from dataset import get_dataloaders
from losses import TLoss
from utils import get_default_dataset_dir, load_close_series, prepare_series, get_split_indices, save_predictions_csv, plot_predictions
from models.garch_lstm import GARCHLSTMHybrid, evaluate_garch_lstm
"""))

# Cell 3: Configs / Setup Device
cells.append(new_code_cell("""\
print('Using device:', DEVICE)
print('Horizons:', HORIZONS)
print('Epochs:', EPOCHS)
print('Batch Size:', BATCH_SIZE)
"""))

# Cell 4: Train Function
cells.append(new_code_cell("""\
def train_garch_lstm_hybrid(model, train_loader, val_loader):
    criterion = TLoss(v=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_state = None
    best_val_loss = float("inf")
    wait = 0
    early_stopping_patience = 20
    
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for enc_r, dec_v, target_r, _ in train_loader:
            enc_r, dec_v, target_r = enc_r.to(DEVICE), dec_v.to(DEVICE), target_r.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred_var = model(enc_r, dec_v)  # (batch, max_horizon)
            
            loss = criterion(pred_var, target_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_r, dec_v, target_r, _ in val_loader:
                enc_r, dec_v, target_r = enc_r.to(DEVICE), dec_v.to(DEVICE), target_r.to(DEVICE)
                pred_var = model(enc_r, dec_v)
                val_losses.append(float(criterion(pred_var, target_r).item()))

        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
"""))

# Cell 5: Main Pipeline Loop
cells.append(new_code_cell("""\
dataset_dir = get_default_dataset_dir()
csv_files = glob.glob(str(dataset_dir / "*.csv"))

if not csv_files:
    print(f"No CSV files found in {dataset_dir}")
else:
    pred_dir = '/kaggle/working/results/predictions'
    viz_dir = '/kaggle/working/results/visualizations'
    model_dir = '/kaggle/working/results/model_params'
    
    for csv_file in csv_files:
        index_name = Path(csv_file).stem.upper()
        print(f"\\n{'='*50}\\nProcessing {index_name}\\n{'='*50}")
        
        # 1. Data Prep — compute returns/vol on full series, split on raw indices
        close_series = load_close_series(csv_file)
        returns, volatility = prepare_series(close_series)
        train_end, val_end = get_split_indices(len(close_series), index_name)
        
        # Get dataloaders for GARCH-LSTM training
        train_loader, val_loader, test_r, test_v, test_time = get_dataloaders(csv_file, batch_size=BATCH_SIZE)
        
        # Only evaluate GARCH-LSTM Hybrid
        print(f"--- Running GARCH-LSTM Hybrid ---")
        model = GARCHLSTMHybrid(hidden_size=16).to(DEVICE)
        model, history = train_garch_lstm_hybrid(model, train_loader, val_loader)
        
        torch.save(model.state_dict(), f"{model_dir}/{index_name}_GARCH_LSTM_weights.pth")
        
        # test_r and test_v from get_dataloaders include seq_len lookback window
        lstm_predictions = evaluate_garch_lstm(
            model, test_r, test_v, 
            seq_len=DEFAULT_SEQ_LEN, horizons=HORIZONS, device=DEVICE
        )
        
        save_predictions_csv(
            index_name, "GARCH-LSTM-Hybrid", lstm_predictions,
            test_time, test_r.iloc[DEFAULT_SEQ_LEN:], volatility, val_end, pred_dir
        )
        plot_predictions(index_name, "GARCH-LSTM-Hybrid", lstm_predictions, test_time, test_v.iloc[DEFAULT_SEQ_LEN:], viz_dir, horizon=21)

print('\\nALL TASKS DONE! Check /kaggle/working/results/')
"""))

nb.cells = cells

# Use absolute path assuming it will be run in Cwd
with open('D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/GARCH based/GARCH_LSTM_Only_Kaggle_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook generated successfully.')
