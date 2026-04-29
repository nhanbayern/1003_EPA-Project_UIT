from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from AAAI24_GARCH_NN_Reproduction.core.custom_losses import TLoss


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

    def forecast_next_variance(self, encoder_returns, decoder_variance):
        if encoder_returns.dim() == 1:
            encoder_returns = encoder_returns.unsqueeze(0)
        if decoder_variance.dim() == 1:
            decoder_variance = decoder_variance.unsqueeze(0)

        batch_size, seq_len = encoder_returns.shape
        c_t = torch.zeros(batch_size, self.hidden_size, device=encoder_returns.device)

        sigma2_prev = decoder_variance[:, 0:1]

        for t in range(1, seq_len):
            eps_prev = encoder_returns[:, t - 1 : t]
            _, c_t = self.cell(eps_prev, sigma2_prev, c_t)
            sigma2_prev = decoder_variance[:, t : t + 1]

        eps_prev = encoder_returns[:, -1:]
        sigma2_t, _ = self.cell(eps_prev, sigma2_prev, c_t)
        return sigma2_t.mean(dim=-1)


def train_garch_lstm_hybrid(
    model,
    train_loader,
    val_loader,
    device="cpu",
    epochs=100,
    learning_rate=1e-2,
    lr_factor=0.5,
    lr_patience=5,
    early_stopping_patience=20,
    min_lr=1e-6,
):
    device = torch.device(device)
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    criterion = TLoss(v=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for _ in range(epochs):
        model.train()
        train_losses = []

        for enc_r, dec_v, target_r, _ in train_loader:
            enc_r = enc_r.to(device, non_blocking=use_amp)
            dec_v = dec_v.to(device, non_blocking=use_amp)
            target_r = target_r.to(device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_var = model.forecast_next_variance(enc_r, dec_v)
                    loss = criterion(pred_var, target_r)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_var = model.forecast_next_variance(enc_r, dec_v)
                loss = criterion(pred_var, target_r)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_r, dec_v, target_r, _ in val_loader:
                enc_r = enc_r.to(device, non_blocking=use_amp)
                dec_v = dec_v.to(device, non_blocking=use_amp)
                target_r = target_r.to(device, non_blocking=use_amp)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_var = model.forecast_next_variance(enc_r, dec_v)
                        val_loss = criterion(pred_var, target_r)
                else:
                    pred_var = model.forecast_next_variance(enc_r, dec_v)
                    val_loss = criterion(pred_var, target_r)
                val_losses.append(float(val_loss.item()))

        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def rolling_forecast_variance(
    model,
    train_returns,
    train_variance,
    test_returns,
    test_variance,
    seq_len=126,
    device="cpu",
):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    use_amp = device.type == "cuda"

    history_r = list(np.asarray(train_returns, dtype=float))
    history_v = list(np.asarray(train_variance, dtype=float))

    test_r = np.asarray(test_returns, dtype=float)
    test_v = np.asarray(test_variance, dtype=float)

    preds = []
    with torch.no_grad():
        for r_next, v_next in zip(test_r, test_v):
            if len(history_r) < seq_len:
                history_r.append(float(r_next))
                history_v.append(float(v_next))
                continue

            enc_r = torch.tensor(
                history_r[-seq_len:], dtype=torch.float32, device=device
            ).unsqueeze(0)
            dec_v = torch.tensor(
                history_v[-seq_len:], dtype=torch.float32, device=device
            ).unsqueeze(0)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_var_t = model.forecast_next_variance(enc_r, dec_v)
            else:
                pred_var_t = model.forecast_next_variance(enc_r, dec_v)

            pred_var = float(pred_var_t.cpu().item())
            preds.append(max(pred_var, 1e-6))

            history_r.append(float(r_next))
            history_v.append(float(v_next))

    return np.asarray(preds, dtype=float)
