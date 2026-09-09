# Unified benchmark on Modal

From the repository root, authenticate once with `py -3.11 -m modal setup`, then run one or more model families:

```powershell
py -3.11 -m modal run experiments/all_models_modal/modal_app.py --families garch
py -3.11 -m modal run experiments/all_models_modal/modal_app.py --families transformer
py -3.11 -m modal run experiments/all_models_modal/modal_app.py --families moirai
py -3.11 -m modal run experiments/all_models_modal/modal_app.py --families moiraivar
py -3.11 -m modal run experiments/all_models_modal/modal_app.py --families garch,transformer,moiraivar
```

Artifacts are extracted locally to `output/<YYYYMMDD_HHMMSS>/`. The run directory contains predictions, figures/metrics where produced, and model artifacts:

- `garch/model_params/`: fitted GARCH/GJR-GARCH/FI-GARCH parameters and GARCH-LSTM weights.
- `transformer/models_weights/`: Transformer, Autoformer, Informer, and Reformer weights.
- `moiraivar/weights/`: MoiraiVaR model state dictionaries.

The command packages the local working tree sent to Modal, rather than the old `kaggle-implementation` clone hard-coded in the notebooks. Commit or stash intended changes before a production run for reproducibility.
