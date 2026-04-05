import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class LSTMBaseline(nn.Module):
    """Simple LSTM baseline model for volatility forecasting."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, seq_len, input_size)
        Returns:
            output: shape (batch_size,)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.squeeze(-1)


def make_sequences(data, seq_len):
    """Create sequences for time series prediction."""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm_baseline(
    train_data,
    val_data=None,
    seq_len=60,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device=None,
):
    """Train LSTM baseline model."""

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

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).unsqueeze(-1),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = LSTMBaseline(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_cuda)

    train_loss_history = []
    val_loss_history = [] if val_data is not None else None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=use_cuda)
            y_batch = y_batch.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss += loss.item() * X_batch.size(0)

        train_loss = train_loss / len(train_dataset)
        train_loss_history.append(train_loss)

        if val_data is not None:
            model.eval()
            val_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
            X_val, y_val = make_sequences(val_scaled, seq_len)

            X_val_tensor = torch.from_numpy(X_val).unsqueeze(-1).to(device, non_blocking=use_cuda)
            y_val_tensor = torch.from_numpy(y_val).to(device, non_blocking=use_cuda)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                val_loss_history.append(val_loss)

        if (epoch + 1) % 10 == 0:
            if val_data is not None:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}")

    return model, scaler, train_loss_history, val_loss_history


def predict_lstm_baseline(model, data, seq_len, scaler, device="cpu"):
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    model.eval()

    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    X, _ = make_sequences(data_scaled, seq_len)
    X_tensor = torch.from_numpy(X).unsqueeze(-1).to(device, non_blocking=use_cuda)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            predictions_scaled = model(X_tensor).detach().cpu().numpy()

    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    return predictions


def predict_lstm_baseline_rolling(model, train_data, test_data, seq_len, scaler, device="cpu"):
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    model.eval()

    history = scaler.transform(train_data.reshape(-1, 1)).flatten().tolist()
    preds = []

    with torch.no_grad():
        for x in test_data:
            if len(history) < seq_len:
                break

            seq = torch.tensor(history[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                pred_scaled = float(model(seq).detach().cpu().reshape(-1)[0])

            pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
            preds.append(abs(pred_unscaled))
            history.append(float(scaler.transform([[x]])[0, 0]))

    return np.array(preds, dtype=np.float32)
