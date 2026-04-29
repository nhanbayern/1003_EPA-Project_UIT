"""Model modules for AAAI24 GARCH-NN reproduction."""

from .stat_baselines import rolling_forecast_variance as rolling_stat_forecast_variance
from .dl_baselines import (
    SUPPORTED_DL_MODELS,
    build_dl_model,
    build_dataloaders,
    rolling_forecast_variance as rolling_dl_forecast_variance,
    train_dl_model,
)
from .garch_lstm_hybrid import (
    GARCH_LSTM_Cell,
    GARCHLSTMHybrid,
    rolling_forecast_variance as rolling_hybrid_forecast_variance,
    train_garch_lstm_hybrid,
)

__all__ = [
    "SUPPORTED_DL_MODELS",
    "GARCH_LSTM_Cell",
    "GARCHLSTMHybrid",
    "build_dataloaders",
    "build_dl_model",
    "rolling_dl_forecast_variance",
    "rolling_hybrid_forecast_variance",
    "rolling_stat_forecast_variance",
    "train_dl_model",
    "train_garch_lstm_hybrid",
]
