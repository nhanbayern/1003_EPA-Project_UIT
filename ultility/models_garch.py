import numpy as np
from arch import arch_model

def garch_forecast_fixed_params(train_data, test_data, model_name='GARCH'):
    history = list(train_data)
    predictions = []

    p, o, q = (1, 1, 1) if model_name == 'GJR-GARCH' else (1, 0, 1)

    # 🔹 Fit ONLY on train
    model = arch_model(train_data, p=p, o=o, q=q, vol='Garch', dist='t')
    model_fit = model.fit(disp='off', show_warning=False)

    # 🔹 Extract parameters
    params = model_fit.params

    # 🔹 Get last conditional variance from train
    sigma2 = model_fit.conditional_volatility[-1] ** 2

    for t in range(len(test_data)):
        eps_prev = history[-1]

        # GARCH recursion
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']

        # GJR term nếu có
        if model_name == 'GJR-GARCH':
            gamma = params.get('gamma[1]', 0.0)
            indicator = 1.0 if eps_prev < 0 else 0.0
            sigma2 = omega + alpha * eps_prev**2 + beta * sigma2 + gamma * eps_prev**2 * indicator
        else:
            sigma2 = omega + alpha * eps_prev**2 + beta * sigma2

        pred_vol = np.sqrt(sigma2)
        predictions.append(pred_vol)

        # update history với test data
        history.append(test_data[t])

    return np.array(predictions)


def rolling_garch_forecast(train_data, test_data, model_name='GARCH'):
    return garch_forecast_fixed_params(train_data, test_data, model_name=model_name)