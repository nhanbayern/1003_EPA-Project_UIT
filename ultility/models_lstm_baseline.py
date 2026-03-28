import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMBaseline(nn.Module):
    """
    Simple LSTM baseline model for volatility forecasting
    """
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
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: shape (batch_size, seq_len, input_size)
        Returns:
            output: shape (batch_size, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        output = self.fc(last_hidden)
        
        return output.squeeze(-1)


def make_sequences(data, seq_len):
    """Create sequences for time series prediction"""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm_baseline(train_data, val_data=None, seq_len=60, epochs=100, batch_size=32, 
                        learning_rate=0.001, device='cpu'):
    """
    Train LSTM baseline model
    
    Args:
        train_data: numpy array of training data
        val_data: numpy array of validation data (optional)
        seq_len: sequence length for creating sequences
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        device: 'cpu' or 'cuda'
    
    Returns:
        model: trained model
        train_loss_history: list of training losses
        val_loss_history: list of validation losses if val_data provided
    """
    
    device = torch.device(device)
    
    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train, y_train = make_sequences(train_scaled, seq_len)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).unsqueeze(-1),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model setup
    model = LSTMBaseline(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loss_history = []
    val_loss_history = [] if val_data is not None else None
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss = train_loss / len(train_dataset)
        train_loss_history.append(train_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            val_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
            X_val, y_val = make_sequences(val_scaled, seq_len)
            
            X_val_tensor = torch.from_numpy(X_val).unsqueeze(-1).to(device)
            y_val_tensor = torch.from_numpy(y_val).to(device)
            
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_loss_history.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            if val_data is not None:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}")
    
    return model, scaler, train_loss_history, val_loss_history


def predict_lstm_baseline(model, data, seq_len, scaler, device='cpu'):
    device = torch.device(device)
    model.eval()
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    X, _ = make_sequences(data_scaled, seq_len)
    X_tensor = torch.from_numpy(X).unsqueeze(-1).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_tensor).cpu().numpy()
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    return predictions


def predict_lstm_baseline_rolling(model, train_data, test_data, seq_len, scaler, device='cpu'):
    device = torch.device(device)
    model.eval()
    history = scaler.transform(train_data.reshape(-1, 1)).flatten().tolist()
    preds = []
    with torch.no_grad():
        for x in test_data:
            if len(history) < seq_len:
                break
            seq = torch.tensor(history[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            pred_scaled = model(seq).cpu().numpy()[0]
            pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
            preds.append(abs(pred_unscaled))
            history.append(scaler.transform([[x]])[0, 0])
    return np.array(preds)
