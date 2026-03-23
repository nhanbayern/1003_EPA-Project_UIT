
import pandas as pd
import numpy as np
from scipy import stats
import tqdm
import importlib

import ultility.data_loader as data_loader
import ultility.metrics as metrics
import ultility.models_garch as models_garch
import ultility.models_transformer as models_transformer

def run_benchmark(datasets, split_df, window_size=60, seq_len=60):

    all_results = []

    models_to_run = {
        'GARCH': models_garch.rolling_garch_forecast,
        'GJR-GARCH': models_garch.rolling_garch_forecast,
        'Transformer': models_transformer.train_transformer
    }

    for name, series in tqdm.tqdm(datasets.items(), desc="Processing Datasets"):
        try:
            train_data, _, test_data = data_loader.get_splits_for_dataset(series, split_df, name, seq_len)
            nu, _, _ = stats.t.fit(series.dropna().values)

            for model_name, model_func in models_to_run.items():

                if model_name in ['GARCH', 'GJR-GARCH']:
                    vol_forecast = model_func(train_data, test_data, model_name=model_name)
                elif model_name == 'Transformer':
                    vol_forecast = model_func(np.abs(train_data), np.abs(test_data), seq_len=seq_len, epochs=100)
                else:
                    continue

                returns_eval = test_data[-len(vol_forecast):]
                eval_metrics = metrics.evaluate_all(returns_eval, vol_forecast, nu)

                result_row = {"Dataset": name, "Model": model_name, **eval_metrics}
                all_results.append(result_row)

        except (ValueError, IndexError) as e:
            print(f"Skipping dataset {name} due to error: {e}")
            continue

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["Dataset", "Model"]).reset_index(drop=True)
        for col in ['MSE', 'QLIKE', 'Violation_Rate', 'Kupiec_LR', 'Kupiec_p']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else ('N/A' if np.isnan(x) else x))

    return results_df
