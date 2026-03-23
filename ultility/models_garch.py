
import numpy as np
from arch import arch_model
import tqdm

def rolling_garch_forecast(train_data, test_data, model_name='GARCH'):
    history = list(train_data)
    predictions = []

    p, o, q = (1, 1, 1) if model_name == 'GJR-GARCH' else (1, 0, 1)

    for t in range(len(test_data)):
        model = arch_model(history, p=p, o=o, q=q, vol='Garch', dist='t')
        model_fit = model.fit(disp='off', show_warning=False)

        forecast = model_fit.forecast(horizon=1, reindex=False)
        pred_vol = np.sqrt(forecast.variance.values[-1, 0])
        predictions.append(pred_vol)

        history.append(test_data[t])

    return np.array(predictions)
