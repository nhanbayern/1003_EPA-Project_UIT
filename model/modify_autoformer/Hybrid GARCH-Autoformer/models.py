import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PackedStdScaler(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True, unbiased=False) + self.eps
        x_scaled = (x - mean) / std
        return x_scaled

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class ResidualHead(nn.Module):
    def __init__(self, in_features, hidden_dim=64, out_features=21):
        super().__init__()
        # For predicting residuals, we do not use Softplus at the end.
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features)
        )
    def forward(self, x):
        return self.net(x)

# ----------------- AUTOFORMER COMPONENTS -----------------
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads=1, c=1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c = c
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, E = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        res = q_fft * torch.conj(k_fft)
        # One delay score per attention head; aggregate its feature channels.
        corr = torch.fft.irfft(res, dim=1).mean(dim=-1)
        
        weights, delays = torch.topk(corr, self.c, dim=1)
        weights = torch.softmax(weights, dim=1)
        
        out = torch.zeros_like(v)
        positions = torch.arange(L, device=x.device).view(1, L, 1)
        for i in range(self.c):
            delay = delays[:, i, :]
            gather_index = (positions - delay.unsqueeze(1)) % L
            gather_index = gather_index.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            rolled_v = torch.gather(v, 1, gather_index)
            out += rolled_v * weights[:, i, :].unsqueeze(1).unsqueeze(-1)
            
        return self.out(out.reshape(B, L, E))

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads=1, dropout=0.3):
        super().__init__()
        self.decomp1 = series_decomp(kernel_size=5)
        self.decomp2 = series_decomp(kernel_size=5)
        self.autocorr = AutoCorrelation(d_model, n_heads=n_heads, c=1)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x_res, x_trend1 = self.decomp1(x)
        attn_out = self.autocorr(x_res)
        x = x + self.dropout(attn_out)
        
        x_res, x_trend2 = self.decomp2(x)
        ff_out = self.ff(x_res)
        x = x + self.dropout(ff_out)
        return x

class AutoformerResidual(nn.Module):
    def __init__(self, seq_len=60, pred_len=21, d_model=32, e_layers=2, n_heads=4, d_ff=128, dropout=0.3):
        super().__init__()
        self.scaler = PackedStdScaler()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([AutoformerEncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(e_layers)])
        self.head = ResidualHead(seq_len * d_model, out_features=pred_len)

    def forward(self, x):
        x = self.scaler(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.flatten(1)
        return self.head(x)

# ----------------- HYBRID GARCH-AUTOFORMER -----------------
class HybridGARCHAutoformer(nn.Module):
    def __init__(self, seq_len=60, pred_len=21, d_model=32, e_layers=2, n_heads=4, d_ff=128, dropout=0.3):
        super().__init__()
        self.autoformer = AutoformerResidual(seq_len, pred_len, d_model, e_layers, n_heads, d_ff, dropout)
        
    def forward(self, x_z, y_garch_vol):
        """
        x_z: Standardized residuals [batch, seq_len, 1]
        y_garch_vol: Base GARCH forecast [batch, pred_len]
        """
        # Predict the variance correction
        e_pred = self.autoformer(x_z)
        
        # Ensembling GARCH variance and predicted residual variance
        garch_var = y_garch_vol ** 2
        final_var = F.softplus(garch_var + e_pred)
        
        return torch.sqrt(final_var)
