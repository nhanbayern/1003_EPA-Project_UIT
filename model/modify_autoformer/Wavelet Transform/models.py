import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ptwt
import pywt


# ----------------- UTILITY MODULES -----------------
class PackedStdScaler(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: [batch, seq_len, features]
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True, unbiased=False) + self.eps
        return (x - mean) / std


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VolatilityHead(nn.Module):
    def __init__(self, in_features, hidden_dim=64, out_features=21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)


# ----------------- WAVELET DECOMPOSITION MODULE -----------------
class WaveletDecomp(nn.Module):
    """
    Replaces the Moving Average series_decomp in Autoformer.

    Decomposes x into (residual, trend) using a single-level
    Discrete Wavelet Transform (DWT).
    - trend    : reconstructed from Approximation coefficients only
    - residual : reconstructed from Detail coefficients only
    Guarantees lossless split: x == residual + trend
    """
    def __init__(self, wavelet='haar', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        # x: [B, L, C]  (batch, seq_len, channels)
        B, L, C = x.shape

        # ptwt operates on real-valued signals
        # Process each channel independently
        x_perm = x.permute(0, 2, 1)          # [B, C, L]
        x_flat = x_perm.reshape(B * C, L)    # [B*C, L]

        # Single-level DWT; returns [cA, cD]
        coeffs = ptwt.wavedec(x_flat, self.wavelet, level=self.level)
        cA, cD = coeffs[0], coeffs[1]

        # Reconstruct trend from approximation only (zero out detail)
        cD_zeros = torch.zeros_like(cD)
        trend_flat = ptwt.waverec([cA, cD_zeros], self.wavelet)

        # Reconstruct residual from detail only (zero out approximation)
        cA_zeros = torch.zeros_like(cA)
        residual_flat = ptwt.waverec([cA_zeros, cD], self.wavelet)

        # Trim or pad to original length L (waverec may slightly differ)
        trend_flat = trend_flat[:, :L]
        residual_flat = residual_flat[:, :L]

        # Reshape back: [B, C, L] -> [B, L, C]
        trend = trend_flat.reshape(B, C, L).permute(0, 2, 1)
        residual = residual_flat.reshape(B, C, L).permute(0, 2, 1)

        return residual, trend


# ----------------- AUTOCORRELATION MODULE -----------------
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

        # FFT-based auto-correlation
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=1).mean(dim=-1)   # [B, L, heads]

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


# ----------------- WAVELET AUTOFORMER ENCODER LAYER -----------------
class WaveletAutoformerEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with WaveletDecomp replacing Moving Average.
    """
    def __init__(self, d_model, d_ff, n_heads=1, dropout=0.3, wavelet='haar'):
        super().__init__()
        self.decomp1 = WaveletDecomp(wavelet=wavelet, level=1)
        self.decomp2 = WaveletDecomp(wavelet=wavelet, level=1)
        self.autocorr = AutoCorrelation(d_model, n_heads=n_heads, c=1)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Auto-Correlation block with Wavelet decomposition
        x_res, x_trend1 = self.decomp1(x)
        attn_out = self.autocorr(x_res)
        x = x + self.dropout(attn_out)

        # Feed-Forward block with Wavelet decomposition
        x_res, x_trend2 = self.decomp2(x)
        ff_out = self.ff(x_res)
        x = x + self.dropout(ff_out)
        return x


# ----------------- WAVELET AUTOFORMER -----------------
class WaveletAutoformer(nn.Module):
    """
    Autoformer variant where the Moving Average series_decomp is replaced
    by a lossless Wavelet-based decomposition (WaveletDecomp).
    
    This eliminates over-smoothing and preserves high-frequency shock information
    that traditional Moving Average would discard.
    """
    def __init__(
        self,
        seq_len=60,
        pred_len=21,
        d_model=32,
        e_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.3,
        wavelet='haar'
    ):
        super().__init__()
        self.scaler = PackedStdScaler()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            WaveletAutoformerEncoderLayer(d_model, d_ff, n_heads, dropout, wavelet)
            for _ in range(e_layers)
        ])

        self.head = VolatilityHead(seq_len * d_model, out_features=pred_len)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = self.scaler(x)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.flatten(1)    # [batch, seq_len * d_model]
        return self.head(x)  # [batch, pred_len]
