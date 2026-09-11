import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from pathlib import Path

nb = new_notebook()
cells = []

# Cell 1: Setup
cells.append(new_code_cell("""\
!pip install PyWavelets ptwt
!git clone -b modify_autoformer --single-branch https://github.com/nhanbayern/1003_EPA-Project_UIT.git
import sys, os
sys.path.append('/kaggle/working/1003_EPA-Project_UIT/model/modify_autoformer/Wavelet Transform')
"""))

# Cell 2: Imports
cells.append(new_code_cell("""\
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import VolatilityDataset
from models import WaveletAutoformer
from config import TIERS_CONFIG
import glob
from pathlib import Path
import os
"""))

# Cell 3: Global Config
cells.append(new_code_cell("""\
DATA_DIR = '/kaggle/input/datasets/trnhngv/historical-price'
HORIZONS = [1, 3, 5, 10, 21]
EVAL_INDICES = [0, 2, 4, 9, 20]
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

SPLITS = {
    'DAX_40': (1983, 1104, 972),
    'Euronext_100': (2005, 1115, 979),
    'IBEX_35': (1815, 1362, 923),
    'KOSPI_Index': (1944, 1109, 881),
    'Nikkei_225': (1677, 1174, 1062),
    'SMI': (1987, 1135, 901),
    'S&P_500': (1988, 1109, 927),
    'VN30': (1971, 1096, 925),
    'VN-Index': (1964, 1103, 925)
}
"""))

# Cell 4: Train Function
cells.append(new_code_cell("""\
def train_model(model, train_loader, val_loader):
# The model target is the shared rolling-60 volatility at endpoint t+h.
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for x, y_vol, _, _ in train_loader:
            x, y_vol = x.to(DEVICE), y_vol.to(DEVICE)
            optimizer.zero_grad()
            pred_vol = model(x)
            loss = criterion(pred_vol, y_vol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_vol, _, _ in val_loader:
                x, y_vol = x.to(DEVICE), y_vol.to(DEVICE)
                val_loss += criterion(model(x), y_vol).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
"""))

# Cell 5: Evaluate & Save Predictions
cells.append(new_code_cell("""\
def evaluate_and_save(model, loader, df, index_name, model_name, tier_name, split='test'):
    model.eval()
    model.to(DEVICE)
    results = []

    with torch.no_grad():
        for x, y_vol, y_ret, ts in loader:
            x = x.to(DEVICE)
            pred_vol = model(x).cpu().numpy()
            y_vol = y_vol.numpy()
            ts = ts.numpy()

            for i in range(len(ts)):
                t = ts[i]
                for j, h in enumerate(HORIZONS):
                    idx = EVAL_INDICES[j]
                    origin_time = df['time'].iloc[t] if t < len(df) else None
                    next_return = df['log_return'].iloc[t + 1] if t + 1 < len(df) else None
                    if origin_time is not None:
                        results.append({
                            'time': origin_time,
                            'log_return': next_return,
                            'horizon': h,
                            'true_volatility': y_vol[i, idx],
                            'predict_volatility': pred_vol[i, idx]
                        })

    res_df = pd.DataFrame(results)
    pred_dir = f'/kaggle/working/results_wavelet/all_predictions/{tier_name}'
    os.makedirs(pred_dir, exist_ok=True)
    suffix = '' if split == 'test' else f'_{split}'
    res_df.to_csv(f'{pred_dir}/{index_name}_{model_name}{suffix}_predictions.csv', index=False)
"""))

# Cell 6: Main Loop
cells.append(new_code_cell("""\
csv_files = glob.glob(f'{DATA_DIR}/*.csv')
if len(csv_files) == 0:
    print("WARNING: No CSV files found. Running mock logic for syntax check.")
    csv_files = ['mock_DAX_40.csv']

for tier_name, config in TIERS_CONFIG.items():
    print(f'\\n========================================')
    print(f'   RUNNING TIER: {tier_name}')
    print(f'   Config: {config}')
    print(f'========================================')

    for fpath in csv_files:
        fname = Path(fpath).stem
        index_name = fname.replace(' ', '_')
        print(f'\\n--- Processing {index_name} for {tier_name} ---')

        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
        else:
            dates = pd.date_range('2010-01-01', periods=4059)
            df = pd.DataFrame({'time': dates, 'close': np.random.randn(4059).cumsum() + 1000})
            index_name = 'DAX_40'

        df['time'] = pd.to_datetime(df['time'])
        df = df[df['time'] >= '2010-01-01'].copy()
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1)) * 100.0

        df = df.dropna(subset=['log_return']).reset_index(drop=True)

        if index_name in SPLITS:
            n_train, n_val, n_test = SPLITS[index_name]
        else:
            N = len(df)
            n_train = int(N * 0.6)
            n_val   = int(N * 0.2)
            n_test  = N - n_train - n_val

        df_train = df.iloc[:n_train].copy()
        df_val   = df.iloc[max(0, n_train - 60): n_train + n_val].copy()
        df_test  = df.iloc[max(0, n_train + n_val - 60):].copy()

        ds_train = VolatilityDataset(df_train, lookback=60, horizon=21)
        ds_val   = VolatilityDataset(df_val,   lookback=60, horizon=21)
        ds_test  = VolatilityDataset(df_test,  lookback=60, horizon=21)

        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
        val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        m_name = 'WaveletAutoformer'
        model = WaveletAutoformer(**config)
        print(f'Training {m_name}...')
        model.to(DEVICE)
        model = train_model(model, train_loader, val_loader)

        weight_dir = f'/kaggle/working/results_wavelet/models_weights/{tier_name}'
        os.makedirs(weight_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{weight_dir}/{index_name}_{m_name}.pt')

        evaluate_and_save(model, val_loader, df_val, index_name, m_name, tier_name, split='validation')
        evaluate_and_save(model, test_loader, df_test, index_name, m_name, tier_name, split='test')

print('\\nALL TIERS DONE! Check /kaggle/working/results_wavelet/')
"""))

nb.cells = cells

out_path = Path(__file__).with_name('kaggle_notebook_wavelet.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Notebook written to: {out_path}')
