import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PackedStdScaler(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
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
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class VolatilityHead(nn.Module):
    def __init__(self, in_features, hidden_dim=64, out_features=21):
        super().__init__()
        # Matching Moirai MoE GARCHs: Linear -> GELU -> Linear -> Softplus
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
            nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

# ----------------- 1. VANILLA TRANSFORMER -----------------
class VanillaTransformer(nn.Module):
    def __init__(self, seq_len=60, pred_len=21, d_model=32, n_heads=4, e_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.scaler = PackedStdScaler()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        self.head = VolatilityHead(seq_len * d_model, out_features=pred_len)

    def forward(self, x):
        # x: [batch, 60, 1]
        x = self.scaler(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x) # [batch, 60, d_model]
        x = x.flatten(1) # [batch, 60 * d_model]
        out = self.head(x) # [batch, 21]
        return out

# ----------------- 2. AUTOFORMER (Miniaturized) -----------------
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
    def __init__(self, d_model, c=1): # c is top_k delay
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, E = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, E).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # FFT based auto-correlation
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=1) # [B, L, E]
        
        # simplified top-k delays (we just take top c delays)
        weights, delays = torch.topk(corr, self.c, dim=1)
        weights = torch.softmax(weights, dim=1)
        
        # Roll v by delays and aggregate
        out = torch.zeros_like(v)
        for i in range(self.c):
            delay = delays[:, i, 0] # simplified assumption across batches
            # a mock roll for simplicity in miniaturized version
            rolled_v = torch.roll(v, shifts=-1, dims=1) 
            out += rolled_v * weights[:, i:i+1, :]
            
        return self.out(out)

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()
        self.decomp1 = series_decomp(kernel_size=5)
        self.decomp2 = series_decomp(kernel_size=5)
        self.autocorr = AutoCorrelation(d_model, c=1)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Auto-correlation
        x_res, x_trend1 = self.decomp1(x)
        attn_out = self.autocorr(x_res)
        x = x + self.dropout(attn_out)
        
        # FF
        x_res, x_trend2 = self.decomp2(x)
        ff_out = self.ff(x_res)
        x = x + self.dropout(ff_out)
        return x

class Autoformer(nn.Module):
    def __init__(self, seq_len=60, pred_len=21, d_model=32, e_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.scaler = PackedStdScaler()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([AutoformerEncoderLayer(d_model, d_ff, dropout) for _ in range(e_layers)])
        self.head = VolatilityHead(seq_len * d_model, out_features=pred_len)

    def forward(self, x):
        x = self.scaler(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.flatten(1)
        return self.head(x)

# ----------------- 3. INFORMER (Miniaturized) -----------------
class ProbSparseAttentionMock(nn.Module):
    def __init__(self, d_model, n_heads, factor=1):
        super().__init__()
        self.n_heads = n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, L, E = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, E // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Full attention is calculated, but this represents the ProbSparse concept for small L
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E // self.n_heads)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, E)
        return self.out(out)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super().__init__()
        self.attn = ProbSparseAttentionMock(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x

class Distilling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.elu(self.conv(x))
        x = self.pool(x)
        return x.permute(0, 2, 1)

class Informer(nn.Module):
    def __init__(self, seq_len=60, pred_len=21, d_model=32, n_heads=4, e_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.scaler = PackedStdScaler()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        # Distilling layers reduce seq_len by half each time
        self.distils = nn.ModuleList([Distilling(d_model) for _ in range(e_layers - 1)])
        
        # Calculate final sequence length after distilling
        final_seq_len = seq_len
        for _ in range(e_layers - 1):
            final_seq_len = (final_seq_len + 1) // 2
            
        self.head = VolatilityHead(final_seq_len * d_model, out_features=pred_len)

    def forward(self, x):
        x = self.scaler(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.distils):
                x = self.distils[i](x)
        x = x.flatten(1)
        return self.head(x)

# ----------------- 4. REFORMER (Miniaturized) -----------------
class LSHAttentionMock(nn.Module):
    def __init__(self, d_model, n_heads, bucket_size=4):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # A mock LSH for demonstration on L=60 without complex CUDA dependencies
        B, L, E = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, E // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Shared query-key space approximation
        qk = (q + k) / 2
        scores = torch.matmul(qk, qk.transpose(-2, -1)) / math.sqrt(E // self.n_heads)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, E)
        return self.out(out)

class ReformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super().__init__()
        self.attn = LSHAttentionMock(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x

class Reformer(nn.Module):
    def __init__(self, seq_len=60, pred_len=21, d_model=32, n_heads=4, e_layers=2, d_ff=128, dropout=0.3):
        super().__init__()
        self.scaler = PackedStdScaler()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([ReformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.head = VolatilityHead(seq_len * d_model, out_features=pred_len)

    def forward(self, x):
        x = self.scaler(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.flatten(1)
        return self.head(x)
