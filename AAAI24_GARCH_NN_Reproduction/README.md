# AAAI24_GARCH_NN_Reproduction

This folder is isolated from the legacy codebase.
No existing source files were modified.

## Data source

This reproduction package reads directly from the existing dataset folder:

- ../dataset

No local data folder is created in this package.

## Implemented structure

- core/data_processor.py
- core/custom_losses.py
- core/custom_metrics.py
- models/stat_baselines.py
- models/dl_baselines.py
- models/garch_lstm_hybrid.py
- experiments/train_dl_models.py
- experiments/run_benchmark.py

## Notes

- Data preprocessing includes:
  - log return
  - MA(5) on squared returns
  - scale by 100
  - split 8:1:1 in chronological order
  - sequence length default seq_len=126
- Statistical baselines include rolling refit forecast:
  - GARCH(1,1)
  - GJR-GARCH
  - FI-GARCH
- DL training scripts use:
  - Adam with initial learning rate 1e-2
  - ReduceLROnPlateau factor=0.5
  - early stopping patience=20
- Benchmark script loops over:
  - 5 seeds: [42, 123, 202, 303, 404]
  - horizons: [1, 3, 5, 10, 21]
- Benchmark outputs:
  - Main metrics CSV (MAE, MSE by dataset, seed, horizon, model)
  - Mean-over-seeds prediction CSV: <output_stem>_predictions_mean_seed.csv
    with columns: time, dataset, model, horizon, True_Volatility,
    Pred_Volatility, time_train

## Quick start

Run one DL training job:

python -m AAAI24_GARCH_NN_Reproduction.experiments.train_dl_models --dataset VN30_INDEX.csv --seq-len 126 --epochs 120

Run benchmark:

python -m AAAI24_GARCH_NN_Reproduction.experiments.run_benchmark --seq-len 126 --epochs 60
