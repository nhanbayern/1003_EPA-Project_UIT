import warnings
import numpy as np

try:
    from arch import arch_model
except ImportError as exc:
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
    return fit_result, model

def evaluate_stat_model(train_returns, test_returns, model_name="GARCH", dist="t", horizons=[1, 3, 5, 10, 21]):
    """
    Rolling forecast for multiple horizons.
    Returns:
    - predictions: dict of {h: array_of_volatilities}
    - all_params: list of dicts containing the fitted parameters
    """
    history = list(np.asarray(train_returns, dtype=float))
    test_arr = np.asarray(test_returns, dtype=float)
    max_h = max(horizons)

    if len(history) < 30:
        raise ValueError("Need at least 30 training points for stable ARCH fitting")

    predictions = {h: [] for h in horizons}
    all_params = []
    
    # Fallback to empirical variance if fit fails
    fallback_var = float(np.var(history)) if len(history) > 1 else 1.0
    fallback_var = max(fallback_var, 1e-6)

    # To avoid data leakage, at time t, we predict t+1..t+h
    for obs in test_arr:
        try:
            fit_result, model = _fit_arch_model(history, model_name=model_name, dist=dist)
            forecast = fit_result.forecast(horizon=max_h, reindex=False)
            pred_vars = forecast.variance.values[-1, :] # shape (max_h,)
            all_params.append(dict(fit_result.params))
        except Exception as e:
            pred_vars = np.full(max_h, fallback_var)
            all_params.append({})
            
        pred_vars = np.maximum(pred_vars, 1e-6)
        
        for h in horizons:
            # Aggregate variance: sum of variances from step 1 to h, divided by h, then sqrt
            agg_var = np.mean(pred_vars[:h])
            vol = np.sqrt(agg_var)
            predictions[h].append(vol)
            
        # Update history with the TRUE observation
        history.append(float(obs))
        
    for h in horizons:
        predictions[h] = np.asarray(predictions[h][:len(test_arr)], dtype=float)

    return predictions, all_params
