import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from .metrics import compute_mse_qlike, kupiec_test

def realized_vol(returns, window):
    return np.sqrt(pd.Series(returns).rolling(window+1).apply(lambda x: np.mean(x**2), raw=True).values)

def map_predictions(datasets, split_df, predictions_dict, seq_len=60):
    preds_list = []
    for ds, series in datasets.items():
        tr, va, te = int(split_df.loc[ds, 'Train']), int(split_df.loc[ds, 'Val']), int(split_df.loc[ds, 'Test'])
        test_start = tr + va
        test_data = series.iloc[test_start:test_start+te].values if hasattr(series, 'iloc') else series[test_start:test_start+te]
        test_dates = series.index[test_start+seq_len:test_start+te] if hasattr(series, 'index') else np.arange(test_start+seq_len, test_start+te)
        
        for model, preds in predictions_dict.items():
            if ds not in preds:
                continue
            y_pred = np.abs(preds[ds][:len(test_data)-seq_len])
            y_real = realized_vol(test_data, seq_len)[seq_len:]
            
            for i, d in enumerate(test_dates):
                if i < len(y_pred):
                    preds_list.append({
                        'Dataset': ds,
                        'Model': model,
                        'Date': d,
                        'return': test_data[seq_len+i],
                        'predicted_vol': y_pred[i],
                        'vol_realized': y_real[i]
                    })
    
    return pd.DataFrame(preds_list)

def compute_results(datasets, split_df, predictions_dict, nu_dict, seq_len=60):
    results = []
    preds_df = map_predictions(datasets, split_df, predictions_dict, seq_len)
    
    for ds, series in datasets.items():
        tr, va, te = int(split_df.loc[ds, 'Train']), int(split_df.loc[ds, 'Val']), int(split_df.loc[ds, 'Test'])
        test_start = tr + va
        test_data = series.iloc[test_start:test_start+te].values if hasattr(series, 'iloc') else series[test_start:test_start+te]
        
        for model in predictions_dict.keys():
            df_model = preds_df[(preds_df['Dataset']==ds) & (preds_df['Model']==model)]
            if df_model.empty:
                continue
            
            y_pred = df_model['predicted_vol'].values
            y_real = df_model['vol_realized'].values
            returns = df_model['return'].values
            
            mse, qlike = compute_mse_qlike(y_real, y_pred)
            
            var_rolling = -1.645 * y_pred
            vio_rolling = np.mean(returns < var_rolling)
            
            nu = nu_dict.get(ds, 8.0)
            var_dist = -np.sqrt((nu-2)/nu) * y_pred * stats.t.ppf(0.05, nu)
            violations_dist = returns < var_dist
            vio_dist = np.mean(violations_dist)
            
            kupiec_lr, kupiec_p = kupiec_test(violations_dist, 0.95)[:2]
            lr_ind, p_ind = np.nan, np.nan
            
            results.append({
                'Dataset': ds,
                'Model': model,
                'MSE': mse,
                'QLIKE': qlike,
                'Violation_Rate_rolling': vio_rolling,
                'Violation_Rate_Distribution': vio_dist,
                'Kupiec_LR': kupiec_lr,
                'Kupiec_p': kupiec_p,
                'LR_Ind': lr_ind,
                'p_Ind': p_ind
            })
    
    return pd.DataFrame(results)

def save_and_plot(datasets, split_df, predictions_dict, nu_dict, seq_len=60, output_dir='./'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preds_df = map_predictions(datasets, split_df, predictions_dict, seq_len)
    results_df = compute_results(datasets, split_df, predictions_dict, nu_dict, seq_len)
    
    preds_df.to_csv(output_path / 'predicts.csv', index=False)
    results_df.to_csv(output_path / 'models_results.csv', index=False)
    
    datasets_list = sorted(preds_df['Dataset'].unique())
    models_list = sorted(preds_df['Model'].unique())
    n_datasets = len(datasets_list)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, ds in enumerate(datasets_list):
        ax = axes[idx]
        df_ds = preds_df[preds_df['Dataset'] == ds]
        
        for model in models_list:
            df_model = df_ds[df_ds['Model'] == model]
            if not df_model.empty:
                ax.plot(df_model.index, df_model['predicted_vol'], marker='o', label=f'{model} (pred)', alpha=0.7)
        
        ax.plot(df_ds.index, df_ds['vol_realized'], marker='s', label='Realized', linewidth=2, alpha=0.8)
        ax.set_title(f'{ds}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(datasets_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'vol_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return preds_df, results_df
