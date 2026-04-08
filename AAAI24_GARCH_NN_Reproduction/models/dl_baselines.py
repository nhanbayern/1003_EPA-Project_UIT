from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from AAAI24_GARCH_NN_Reproduction.core.custom_losses import TLoss


SUPPORTED_DL_MODELS = ["Autoformer", "Informer", "Reformer", "Transformer"]


class SequenceDataset(Dataset):
    def __init__(self, windows):
        self.encoder_returns = torch.from_numpy(windows["encoder_returns"]).float()
        self.decoder_volatility = torch.from_numpy(windows["decoder_volatility"]).float()
        self.target_returns = torch.from_numpy(windows["target_returns"]).float()
        self.target_variance = torch.from_numpy(windows["target_variance"]).float()

    def __len__(self):
        return self.encoder_returns.size(0)

    def __getitem__(self, idx):
        return (
            self.encoder_returns[idx],
            self.decoder_volatility[idx],
            self.target_returns[idx],
            self.target_variance[idx],
        )


def build_dataloaders(train_windows, val_windows, batch_size=64, device="cpu", num_workers=0):
    train_ds = SequenceDataset(train_windows)
    val_ds = SequenceDataset(val_windows)

    device = torch.device(device)
    use_cuda = device.type == "cuda"

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(max(0, num_workers)),
    }
    if use_cuda:
        loader_kwargs["pin_memory"] = True
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _build_causal_mask(length, device):
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )


class _DualStreamBase(nn.Module):
    def __init__(self, seq_len=126):
        super().__init__()
        self.seq_len = seq_len

    def _shape_sequence(self, x):
        current_len = x.size(1)
        if current_len > self.seq_len:
            return x[:, -self.seq_len :, :]
        if current_len < self.seq_len:
            pad_len = self.seq_len - current_len
            pad = torch.zeros(x.size(0), pad_len, x.size(2), device=x.device, dtype=x.dtype)
            return torch.cat([pad, x], dim=1)
        return x

    def _stack_inputs(self, encoder_returns, decoder_volatility):
        if encoder_returns.dim() == 1:
            encoder_returns = encoder_returns.unsqueeze(0)
        if decoder_volatility.dim() == 1:
            decoder_volatility = decoder_volatility.unsqueeze(0)

        x = torch.stack([encoder_returns, decoder_volatility], dim=-1)
        return self._shape_sequence(x)


class DualStreamTransformer(_DualStreamBase):
    def __init__(self, seq_len=126, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__(seq_len=seq_len)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, encoder_returns, decoder_volatility):
        x = self._stack_inputs(encoder_returns, decoder_volatility)

        seq_len = x.size(1)
        z = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]

        causal_mask = _build_causal_mask(seq_len, z.device)
        z = self.encoder(z, mask=causal_mask)

        pred_var = F.softplus(self.head(z[:, -1, :])).squeeze(-1) + 1e-6
        return pred_var


class DualStreamAutoformer(_DualStreamBase):
    def __init__(self, seq_len=126, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__(seq_len=seq_len)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.ma_window = 5

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def _decompose(self, z):
        trend = F.avg_pool1d(
            z.transpose(1, 2),
            kernel_size=self.ma_window,
            stride=1,
            padding=self.ma_window // 2,
        ).transpose(1, 2)
        season = z - trend
        return season, trend

    def forward(self, encoder_returns, decoder_volatility):
        x = self._stack_inputs(encoder_returns, decoder_volatility)

        seq_len = x.size(1)
        z = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        season, trend = self._decompose(z)

        causal_mask = _build_causal_mask(seq_len, z.device)
        season = self.encoder(season, mask=causal_mask)
        fused = season[:, -1, :] + trend[:, -1, :]

        pred_var = F.softplus(self.head(fused)).squeeze(-1) + 1e-6
        return pred_var


class DualStreamInformer(_DualStreamBase):
    def __init__(self, seq_len=126, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__(seq_len=seq_len)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.distill = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, encoder_returns, decoder_volatility):
        x = self._stack_inputs(encoder_returns, decoder_volatility)

        seq_len = x.size(1)
        z = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        z = F.gelu(z)

        z = self.distill(z.transpose(1, 2)).transpose(1, 2)
        z = F.gelu(z)

        reduced_len = z.size(1)
        causal_mask = _build_causal_mask(reduced_len, z.device)
        z = self.encoder(z, mask=causal_mask)

        pred_var = F.softplus(self.head(z[:, -1, :])).squeeze(-1) + 1e-6
        return pred_var


class DualStreamReformer(_DualStreamBase):
    def __init__(self, seq_len=126, d_model=64, num_layers=2, dropout=0.1, chunk_size=8):
        super().__init__(seq_len=seq_len)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.chunk_size = max(1, int(chunk_size))

        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.score_proj = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, 1)

    def _chunk_pool(self, z):
        if self.chunk_size == 1:
            return z

        batch_size, seq_len, d_model = z.shape
        remainder = seq_len % self.chunk_size
        if remainder:
            pad_len = self.chunk_size - remainder
            pad = torch.zeros(batch_size, pad_len, d_model, device=z.device, dtype=z.dtype)
            z = torch.cat([z, pad], dim=1)

        z = z.view(batch_size, -1, self.chunk_size, d_model).mean(dim=2)
        return z

    def forward(self, encoder_returns, decoder_volatility):
        x = self._stack_inputs(encoder_returns, decoder_volatility)

        seq_len = x.size(1)
        z = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        z, _ = self.encoder(z)
        z = self._chunk_pool(z)

        scores = self.score_proj(z).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.sum(z * attn_weights.unsqueeze(-1), dim=1)

        pred_var = F.softplus(self.head(context)).squeeze(-1) + 1e-6
        return pred_var


def build_dl_model(model_name, seq_len=126):
    if model_name not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unsupported model_name={model_name}")

    if model_name == "Autoformer":
        return DualStreamAutoformer(seq_len=seq_len)
    if model_name == "Informer":
        return DualStreamInformer(seq_len=seq_len)
    if model_name == "Reformer":
        return DualStreamReformer(seq_len=seq_len)
    return DualStreamTransformer(seq_len=seq_len)


def train_dl_model(
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
                    pred_var = model(enc_r, dec_v)
                    loss = criterion(pred_var, target_r)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_var = model(enc_r, dec_v)
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
                        pred_var = model(enc_r, dec_v)
                        val_loss = criterion(pred_var, target_r)
                else:
                    pred_var = model(enc_r, dec_v)
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
                    pred_var_t = model(enc_r, dec_v)
            else:
                pred_var_t = model(enc_r, dec_v)

            pred_var = float(pred_var_t.cpu().item())
            preds.append(max(pred_var, 1e-6))

            history_r.append(float(r_next))
            history_v.append(float(v_next))

    return np.asarray(preds, dtype=float)
