import nbformat
from nbformat.v4 import new_notebook, new_code_cell

nb = new_notebook()

cells = []

# Cell 1: Setup
cells.append(new_code_cell("""\
!git clone -b kaggle-implementation --single-branch https://github.com/nhanbayern/1003_EPA-Project_UIT.git
import sys
import os
sys.path.append('/kaggle/working/1003_EPA-Project_UIT/model/transformer based/version_2')

# Base directories
os.makedirs('/kaggle/working/results_v2/metrics', exist_ok=True)
"""))

# Cell 2: Imports
cells.append(new_code_cell("""\
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import VolatilityDataset
from models import VanillaTransformer, Autoformer, Informer, Reformer
from utils import calculate_fixed_nu, StudentTNLLLoss, calc_mse, calc_mae, calc_qlike, plot_loss_curve, plot_predictions
from config import TIERS_CONFIG
import glob
from pathlib import Path
import os
"""))

# Cell 3: Configs
cells.append(new_code_cell("""\
DATA_DIR = '/kaggle/input/datasets/trnhngv/historical-price'
HORIZONS = [1, 3, 5, 10, 21]
EVAL_INDICES = [0, 2, 4, 9, 20] # Indices corresponding to the horizons in the 21-length output
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

# Table I Splits: (Train, Val, Test)
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
def train_model(model, train_loader, val_loader, nu):
    criterion = StudentTNLLLoss(nu=nu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, _, y_ret, _ in train_loader:
            x, y_ret = x.to(DEVICE), y_ret.to(DEVICE)
            optimizer.zero_grad()
            pred_vol = model(x)
            loss = criterion(pred_vol, y_ret)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _, y_ret, _ in val_loader:
                x, y_ret = x.to(DEVICE), y_ret.to(DEVICE)
                pred_vol = model(x)
                val_loss += criterion(pred_vol, y_ret).item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
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
    return model, train_losses, val_losses
"""))

# Cell 5: Eval Function
cells.append(new_code_cell("""\
def evaluate_and_save(model, test_loader, df, index_name, model_name, tier_name):
    model.eval()
    model.to(DEVICE)
    
    results = []
    
    # Tracking for plotting
    first_horizon_true = []
    first_horizon_pred = []
    
    with torch.no_grad():
        for x, y_vol, y_ret, ts in test_loader:
            x = x.to(DEVICE)
            pred_vol = model(x)
            
            pred_vol = pred_vol.cpu().numpy()
            y_vol = y_vol.numpy()
            ts = ts.numpy()
            
            for i in range(len(ts)):
                t = ts[i]
                
                # Plotting data (just take horizon 1)
                first_horizon_true.append(y_vol[i, EVAL_INDICES[0]])
                first_horizon_pred.append(pred_vol[i, EVAL_INDICES[0]])
                
                for j, h in enumerate(HORIZONS):
                    idx = EVAL_INDICES[j]
                    # Dataset index t starts the future-return sequence; the
                    # reported timestamp is the forecast origin t-1.
                    origin_t = t - 1
                    origin_time = df['time'].iloc[origin_t] if origin_t >= 0 else None
                    next_return = df['log_return'].iloc[t] if t < len(df) else None
                    
                    if origin_time is not None:
                        results.append({
                            'time': origin_time,
                            'log_return': next_return,
                            'horizon': h,
                            'true_volatility': y_vol[i, idx],
                            'predict_volatility': pred_vol[i, idx]
                        })
    
    res_df = pd.DataFrame(results)
    
    pred_dir = f'/kaggle/working/results_v2/all_predictions/{tier_name}'
    os.makedirs(pred_dir, exist_ok=True)
    save_path = f'{pred_dir}/{index_name}_{model_name}_predictions.csv'
    res_df.to_csv(save_path, index=False)
    
    # Save Plot
    plot_dir = f'/kaggle/working/results_v2/visualizations/Predictions/{tier_name}'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = f'{plot_dir}/{index_name}_{model_name}_horizon_1.png'
    plot_predictions(first_horizon_true, first_horizon_pred, plot_path, title=f'{index_name} - {model_name} ({tier_name}) Horizon 1')
    
    return res_df
"""))

# Cell 6: Metrics Function
cells.append(new_code_cell("""\
def calculate_metrics_for_df(res_df, model_name, index_name):
    metrics_list = []
    for h in HORIZONS:
        h_df = res_df[res_df['horizon'] == h]
        if len(h_df) == 0: continue
        
        pred = h_df['predict_volatility'].values
        target = h_df['true_volatility'].values
        
        mse = calc_mse(pred, target)
        mae = calc_mae(pred, target)
        qlike = calc_qlike(pred, target)
        
        metrics_list.append({
            'index': index_name,
            'metric': 'MSE',
            'horizon': h,
            model_name: mse
        })
        metrics_list.append({
            'index': index_name,
            'metric': 'MAE',
            'horizon': h,
            model_name: mae
        })
        metrics_list.append({
            'index': index_name,
            'metric': 'QLIKE',
            'horizon': h,
            model_name: qlike
        })
    return pd.DataFrame(metrics_list)
"""))

# Cell 7: Main Loop
cells.append(new_code_cell("""\
# MAIN LOOP
csv_files = glob.glob(f'{DATA_DIR}/*.csv')
if len(csv_files) == 0:
    print("WARNING: No CSV files found. Running mock logic for syntax check.")
    csv_files = ['mock_DAX_40.csv']

for tier_name, config in TIERS_CONFIG.items():
    print(f'\\n========================================')
    print(f'   RUNNING TIER: {tier_name}')
    print(f'   Config: {config}')
    print(f'========================================')
    
    all_metrics_dfs = []
    
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

        # All benchmark families apply FIXED_SPLITS after the common 2010
        # evaluation start, rather than from market-specific 2008/2009 rows.
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
            n_val = int(N * 0.2)
            n_test = N - n_train - n_val
            
        train_returns = df['log_return'].iloc[:n_train].values
        nu = calculate_fixed_nu(train_returns)
        
        df_train = df.iloc[:n_train].copy()
        df_val = df.iloc[max(0, n_train - 60) : n_train + n_val].copy()
        df_test = df.iloc[max(0, n_train + n_val - 60) : ].copy()
        
        ds_train = VolatilityDataset(df_train, lookback=60, horizon=21)
        ds_val = VolatilityDataset(df_val, lookback=60, horizon=21)
        ds_test = VolatilityDataset(df_test, lookback=60, horizon=21)
        
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        
        # Initialize models with the current Tier's config
        models_dict = {
            'Vanilla': VanillaTransformer(**config),
            'Autoformer': Autoformer(**config),
            'Informer': Informer(**config),
            'Reformer': Reformer(**config)
        }
        
        index_metrics_dfs = []
        for m_name, model in models_dict.items():
            print(f'Training {m_name}...')
            model.to(DEVICE)
            model, t_losses, v_losses = train_model(model, train_loader, val_loader, nu)
            
            # Save Model Weight
            weight_dir = f'/kaggle/working/results_v2/models_weights/{tier_name}'
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{weight_dir}/{index_name}_{m_name}.pt')
            
            # Save Loss Curve
            loss_dir = f'/kaggle/working/results_v2/visualizations/Loss_Curves/{tier_name}'
            os.makedirs(loss_dir, exist_ok=True)
            plot_loss_curve(t_losses, v_losses, f'{loss_dir}/{index_name}_{m_name}_loss.png', title=f'{m_name} Loss ({tier_name})')
            
            # Evaluate & Predict
            res_df = evaluate_and_save(model, test_loader, df_test, index_name, m_name, tier_name)
            
            m_metrics = calculate_metrics_for_df(res_df, m_name, index_name)
            index_metrics_dfs.append(m_metrics)
            
        merged_index_metrics = index_metrics_dfs[0]
        for i in range(1, len(index_metrics_dfs)):
            merged_index_metrics = pd.merge(merged_index_metrics, index_metrics_dfs[i], 
                                            on=['index', 'metric', 'horizon'], how='outer')
        all_metrics_dfs.append(merged_index_metrics)

    if len(all_metrics_dfs) > 0:
        final_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)
        final_metrics_df.to_csv(f'/kaggle/working/results_v2/metrics/comparison_metrics_{tier_name}.csv', index=False)
        print(f'\\nMetrics for {tier_name} saved.')

print('\\nALL TIERS DONE! Check /kaggle/working/results_v2/')
"""))

nb.cells = cells

with open('D:/UIT/1003_EPA_PROJECT/1.0.0/1003_EPA-Project_UIT/model/transformer based/version_2/kaggle_notebook_v2.ipynb', 'w') as f:
    nbformat.write(nb, f)
