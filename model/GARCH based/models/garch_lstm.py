import torch
import torch.nn as nn
import numpy as np

class GARCH_LSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GARCH_LSTM_Cell, self).__init__()
        self.hidden_size = hidden_size

        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)

        self.omega = nn.Parameter(torch.rand(1))
        self.alpha = nn.Parameter(torch.rand(1))
        self.beta = nn.Parameter(torch.rand(1))
        self.gamma = nn.Parameter(torch.rand(1))

        self.w = nn.Parameter(torch.tensor(0.1))

    def forward(self, eps_t_minus_1, sigma2_t_minus_1, c_t_minus_1):
        inputs = torch.cat([eps_t_minus_1, sigma2_t_minus_1], dim=-1)

        f_t = torch.sigmoid(self.W_f(inputs))
        i_t = torch.sigmoid(self.W_i(inputs))
        c_tilde = torch.tanh(self.W_c(inputs))

        c_t = f_t * c_t_minus_1 + i_t * c_tilde

        i_t_minus_1 = (eps_t_minus_1 < 0).float()

        o_t = (
            self.omega
            + self.alpha * (eps_t_minus_1 ** 2)
            + self.gamma * (eps_t_minus_1 ** 2) * i_t_minus_1
            + self.beta * sigma2_t_minus_1
        )

        sigma2_t = o_t * (1 + self.w * torch.tanh(c_t))
        sigma2_t = torch.clamp(sigma2_t, min=1e-6)

        return sigma2_t, c_t

class GARCHLSTMHybrid(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GARCH_LSTM_Cell(input_size=2, hidden_size=hidden_size)

    def forward(self, encoder_returns, decoder_variance):
        if encoder_returns.dim() == 1:
            encoder_returns = encoder_returns.unsqueeze(0)
        if decoder_variance.dim() == 1:
            decoder_variance = decoder_variance.unsqueeze(0)

        batch_size, seq_len = encoder_returns.shape
        c_t = torch.zeros(batch_size, self.hidden_size, device=encoder_returns.device)

        sigma2_prev = decoder_variance[:, 0:1]
        sigma2_path = []

        for t in range(1, seq_len):
            eps_prev = encoder_returns[:, t - 1 : t]
            sigma2_t, c_t = self.cell(eps_prev, sigma2_prev, c_t)
            sigma2_scalar = sigma2_t.mean(dim=-1, keepdim=True)
            sigma2_path.append(sigma2_scalar)
            sigma2_prev = decoder_variance[:, t : t + 1]

        if sigma2_path:
            return torch.cat(sigma2_path, dim=1)
        return torch.zeros(batch_size, 0, device=encoder_returns.device)

    def forecast_multi_variance(self, encoder_returns, decoder_variance, max_horizon=21):
        if encoder_returns.dim() == 1:
            encoder_returns = encoder_returns.unsqueeze(0)
        if decoder_variance.dim() == 1:
            decoder_variance = decoder_variance.unsqueeze(0)

        batch_size, seq_len = encoder_returns.shape
        c_t = torch.zeros(batch_size, self.hidden_size, device=encoder_returns.device)

        sigma2_prev = decoder_variance[:, 0:1]

        # Process historical window
        for t in range(1, seq_len):
            eps_prev = encoder_returns[:, t - 1 : t]
            _, c_t = self.cell(eps_prev, sigma2_prev, c_t)
            sigma2_prev = decoder_variance[:, t : t + 1]

        # Forecast h steps
        eps_prev = encoder_returns[:, -1:]
        forecasts = []
        for h in range(max_horizon):
            sigma2_t, c_t = self.cell(eps_prev, sigma2_prev, c_t)
            sigma2_scalar = sigma2_t.mean(dim=-1, keepdim=True)
            forecasts.append(sigma2_scalar)
            
            # For next step: we don't have true return, expected return is 0
            eps_prev = torch.zeros_like(eps_prev) 
            sigma2_prev = sigma2_scalar

        return torch.cat(forecasts, dim=1) # shape: (batch_size, max_horizon)

def evaluate_garch_lstm(model, test_returns, test_variance, seq_len=60, horizons=[1, 3, 5, 10, 21], device="cpu"):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    history_r = list(np.asarray(test_returns[:seq_len], dtype=float))
    history_v = list(np.asarray(test_variance[:seq_len], dtype=float))
    
    test_r = np.asarray(test_returns[seq_len:], dtype=float)
    test_v = np.asarray(test_variance[seq_len:], dtype=float)
    
    max_h = max(horizons)
    predictions = {h: [] for h in horizons}
    
    with torch.no_grad():
        for i, (r_next, v_next) in enumerate(zip(test_r, test_v)):
            enc_r = torch.tensor(history_r[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            dec_v = torch.tensor(history_v[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            
            pred_var_t = model.forecast_multi_variance(enc_r, dec_v, max_horizon=max_h) # (1, max_h)
            pred_vars = pred_var_t[0].cpu().numpy()
            pred_vars = np.maximum(pred_vars, 1e-6)
            
            for h in horizons:
                # Take conditional std at horizon h: sqrt(sigma2_h)
                vol = np.sqrt(pred_vars[h - 1])
                predictions[h].append(vol)
                
            history_r.append(float(r_next))
            history_v.append(float(v_next))
            
    for h in horizons:
        predictions[h] = np.asarray(predictions[h], dtype=float)
        
    return predictions
