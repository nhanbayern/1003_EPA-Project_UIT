from __future__ import annotations

import warnings

import numpy as np

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover
    arch_model = None
    ARCH_IMPORT_ERROR = exc
else:
    ARCH_IMPORT_ERROR = None


MODEL_SPECS = {
    "GARCH": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "power": 2.0},
    "GJR-GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "power": 2.0},
    "FI-GARCH": {"vol": "FIGARCH", "p": 1, "o": 0, "q": 1, "power": 2.0},
}


def _check_arch_ready():
    if arch_model is None:
        raise ImportError(
            "arch package is required for statistical baselines. "
            "Install it with: pip install arch"
        ) from ARCH_IMPORT_ERROR


def _fit_arch_model(history, model_name="GARCH", dist="t"):
    _check_arch_ready()

    if model_name not in MODEL_SPECS:
        valid = ", ".join(MODEL_SPECS.keys())
        raise ValueError(f"Unsupported model_name={model_name}. Valid: {valid}")

    spec = MODEL_SPECS[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            np.asarray(history, dtype=float),
            mean="Zero",
            vol=spec["vol"],
            p=spec["p"],
            o=spec["o"],
            q=spec["q"],
            power=spec["power"],
            dist=dist,
            rescale=False,
        )
        fit_result = model.fit(disp="off", show_warning=False)

    return fit_result


def rolling_forecast_variance(train_returns, test_returns, model_name="GARCH", dist="t"):
    history = list(np.asarray(train_returns, dtype=float))
    test_arr = np.asarray(test_returns, dtype=float)

    if len(history) < 30:
        raise ValueError("Need at least 30 training points for stable ARCH fitting")

    predictions = []
    fallback_var = float(np.var(history)) if len(history) > 1 else 1.0
    fallback_var = max(fallback_var, 1e-6)

    for obs in test_arr:
        try:
            fit_result = _fit_arch_model(history, model_name=model_name, dist=dist)
            forecast = fit_result.forecast(horizon=1, reindex=False)
            next_var = float(forecast.variance.values[-1, 0])
        except Exception:
            next_var = fallback_var

        next_var = max(next_var, 1e-6)
        predictions.append(next_var)

        fallback_var = next_var
        history.append(float(obs))

    return np.asarray(predictions, dtype=float)


def rolling_forecast_volatility(train_returns, test_returns, model_name="GARCH", dist="t"):
    pred_var = rolling_forecast_variance(
        train_returns=train_returns,
        test_returns=test_returns,
        model_name=model_name,
        dist=dist,
    )
    return np.sqrt(np.maximum(pred_var, 1e-8))
