import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class VolTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, num_layers=2, seq_len=60):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = self.output_layer(x[:, -1, :])
        return x.squeeze(-1)


def make_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_transformer(
    train_data,
    test_data,
    seq_len=60,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

    X_train, y_train = make_sequences(train_scaled, seq_len)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = VolTransformer(seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_cuda)

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            y_batch = y_batch.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

    model.eval()
    predictions = []
    history = list(train_scaled)

    with torch.no_grad():
        for i in range(len(test_data)):
            seq_tensor = torch.tensor(history[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                pred_scaled = float(model(seq_tensor).detach().cpu().reshape(-1)[0])

            pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred_unscaled)
            history.append(float(scaler.transform([[test_data[i]]])[0, 0]))

    return np.abs(np.array(predictions, dtype=np.float32))
