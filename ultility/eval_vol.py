import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from .metrics import compute_mse_qlike, kupiec_test, rolling_std_vol

def realized_vol(returns, window):
    return np.sqrt(pd.Series(returns).rolling(window + 1).apply(lambda x: np.mean(x**2), raw=True).values)

def _parse_prediction_dates(preds_df):
    pred = preds_df.copy()
    pred['Date'] = pd.to_datetime(pred['Date'], errors='coerce', format='mixed', dayfirst=True)
    pred['Date'] = pred['Date'].fillna(pd.to_datetime(preds_df['Date'], errors='coerce', format='mixed', dayfirst=False))
    return pred[pred['Model'] != 'TransformerGARCH'].dropna(subset=['Date']).sort_values(['Dataset', 'Model', 'Date']).reset_index(drop=True)

def _iter_aligned_frames(preds_df, datasets, split_df, seq_len=60):
    pred = _parse_prediction_dates(preds_df)
    for ds in split_df.index:
        if ds not in datasets:
            continue

        tr, va, te = map(int, split_df.loc[ds, ['Train', 'Val', 'Test']])
        series = datasets[ds]
        test_start = tr + va
        test_data = series.iloc[test_start:test_start + te].values
        dates = pd.to_datetime(series.index[test_start + seq_len:test_start + te])

        realized = realized_vol(test_data, seq_len)[seq_len:]
        har_vol = rolling_std_vol(test_data[:-1], seq_len)[seq_len - 1:] if len(test_data) > seq_len else np.array([])
        realized_df = pd.DataFrame({'Date': dates, 'realized_vol': realized[:len(dates)], 'har_vol': har_vol[:len(dates)]})
        return_df = pd.DataFrame({'Date': pd.to_datetime(series.index[test_start:test_start + te]), 'return': test_data})

        for model in pred['Model'].dropna().unique():
            pred_df = pred[(pred['Dataset'] == ds) & (pred['Model'] == model)][['Date', 'predicted_vol']].dropna().sort_values('Date')
            if pred_df.empty:
                continue

            frame = realized_df.merge(pred_df, on='Date', how='inner')
            if frame.empty:
                continue

            frame = frame.merge(return_df, on='Date', how='inner')
            if not frame.empty:
                yield ds, model, frame

def build_predictions_df(datasets, split_df, predictions_dict, seq_len=60):
    rows = []
    for ds, series in datasets.items():
        tr, va, te = map(int, split_df.loc[ds, ['Train', 'Val', 'Test']])
        test_start = tr + va
        test_data = series.iloc[test_start:test_start + te].values if hasattr(series, 'iloc') else series[test_start:test_start + te]
        test_dates = series.index[test_start + seq_len:test_start + te] if hasattr(series, 'index') else np.arange(test_start + seq_len, test_start + te)
        y_real = realized_vol(test_data, seq_len)[seq_len:]

        for model, preds in predictions_dict.items():
            if ds not in preds:
                continue
            y_pred = np.abs(preds[ds][:len(test_dates)])
            for i, d in enumerate(test_dates[:len(y_pred)]):
                rows.append({
                    'Dataset': ds,
                    'Model': model,
                    'Date': d,
                    'return': test_data[seq_len + i],
                    'predicted_vol': y_pred[i],
                    'vol_realized': y_real[i],
                })
    return pd.DataFrame(rows)

def compute_metrics_from_preds(preds_df, datasets, split_df, nu_dict, seq_len=60, confidence_level=0.95):
    rows = []
    for ds, model, eval_df in _iter_aligned_frames(preds_df, datasets, split_df, seq_len):
        ret_eval = eval_df['return'].values
        y_pred = eval_df['predicted_vol'].values
        y_real = eval_df['realized_vol'].values
        y_har = eval_df['har_vol'].values

        mse, qlike = compute_mse_qlike(y_real, y_pred)

        nu = nu_dict.get(ds, 8.0)
        t_alpha = -stats.t.ppf(1 - confidence_level, df=nu)
        scale = np.sqrt((nu - 2) / nu)

        var_har = scale * t_alpha * y_har
        var_dist = scale * t_alpha * y_pred
        vio_har_mask = ret_eval < -var_har[:len(ret_eval)]
        vio_dist_mask = ret_eval < -var_dist[:len(ret_eval)]
        vio_har = np.mean(vio_har_mask)
        vio_dist = np.mean(vio_dist_mask)

        kupiec_har = kupiec_test(vio_har_mask, confidence_level)
        kupiec_dist = kupiec_test(vio_dist_mask, confidence_level)

        rows.append({
            'Dataset': ds,
            'Model': model,
            'MSE': mse,
            'QLIKE': qlike,
            'vio_HAR': vio_har,
            'vio_distribution': vio_dist,
            'Kupiec_LR_HAR': kupiec_har[1],
            'Kupiec_p_HAR': kupiec_har[2],
            'Kupiec_LR_distribution': kupiec_dist[1],
            'Kupiec_p_distribution': kupiec_dist[2],
        })

    return pd.DataFrame(rows)

def save_and_plot(datasets, split_df, predictions_dict, nu_dict, seq_len=60, output_dir='./'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preds_df = build_predictions_df(datasets, split_df, predictions_dict, seq_len)
    results_df = compute_metrics_from_preds(preds_df, datasets, split_df, nu_dict, seq_len)

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
                df_model = df_model.sort_values('Date')
                ax.plot(df_model['Date'], df_model['predicted_vol'], marker='o', label=f'{model} (pred)', alpha=0.7)
        
        df_ds = df_ds.sort_values('Date')
        ax.plot(df_ds['Date'], df_ds['vol_realized'], marker='s', label='Realized', linewidth=2, alpha=0.8)
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
