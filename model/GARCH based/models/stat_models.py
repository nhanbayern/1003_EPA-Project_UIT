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

# Maximum history length for arch fitting (longer series slow down and may cause convergence issues)
MAX_HISTORY = 1000

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
            rescale=True,
        )
        fit_result = model.fit(disp="off", show_warning=False)
    return fit_result, model

def evaluate_stat_model(train_returns, test_returns, model_name="GARCH", dist="t", horizons=[1, 3, 5, 10, 21]):
    """
    Rolling forecast for multiple horizons.
    
    For each test observation, fit GARCH on recent history, forecast conditional
    variance for h steps ahead, and convert to conditional std: sqrt(sigma2_h).
    
    Returns:
    - predictions: dict of {h: array_of_volatilities}
    - all_params: list of dicts containing the fitted parameters
    """
    train_arr = np.asarray(train_returns, dtype=float)
    test_arr = np.asarray(test_returns, dtype=float)
    max_h = max(horizons)

    # Use only the most recent MAX_HISTORY observations for initial history
    history = list(train_arr[-MAX_HISTORY:])

    if len(history) < 30:
        raise ValueError("Need at least 30 training points for stable ARCH fitting")

    predictions = {h: [] for h in horizons}
    all_params = []
    
    # Fallback to empirical variance if fit fails
    fallback_var = float(np.var(history)) if len(history) > 1 else 1.0
    fallback_var = max(fallback_var, 1e-6)

    n_fallback = 0

    for i, obs in enumerate(test_arr):
        try:
            # Limit history to MAX_HISTORY most recent observations
            recent_history = history[-MAX_HISTORY:]
            fit_result, model = _fit_arch_model(recent_history, model_name=model_name, dist=dist)
            forecast = fit_result.forecast(horizon=max_h, reindex=False)
            # Use .iloc[-1] for robust indexing regardless of reindex setting
            pred_vars = forecast.variance.iloc[-1, :].values  # shape (max_h,)
            all_params.append(dict(fit_result.params))
        except Exception as e:
            if i < 3 or n_fallback == 0:
                print(f"[WARN] {model_name} fit failed at step {i}: {type(e).__name__}: {e}")
            pred_vars = np.full(max_h, fallback_var)
            all_params.append({})
            n_fallback += 1
            
        pred_vars = np.maximum(pred_vars, 1e-6)
        
        for h in horizons:
            # Take conditional std at horizon h: sqrt(sigma2_h)
            # pred_vars[h-1] is the h-step ahead conditional variance
            vol = np.sqrt(pred_vars[h - 1])
            predictions[h].append(vol)
            
        # Update history with the TRUE observation
        history.append(float(obs))
        # Update fallback with expanding variance
        fallback_var = max(float(np.var(history[-MAX_HISTORY:])), 1e-6)
        
    if n_fallback > 0:
        print(f"[INFO] {model_name}: {n_fallback}/{len(test_arr)} steps used fallback ({100*n_fallback/len(test_arr):.1f}%)")

    for h in horizons:
        predictions[h] = np.asarray(predictions[h][:len(test_arr)], dtype=float)

    return predictions, all_params
