import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerGARCH(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, max_len=512, correction_scale=0.05):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.correction_scale = correction_scale

        self.raw_omega = nn.Parameter(torch.tensor(-5.0))
        self.raw_alpha = nn.Parameter(torch.tensor(-2.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda = nn.Parameter(torch.tensor(-2.0))

        self.raw_phi1 = nn.Parameter(torch.tensor(-1.0))
        self.raw_phi5 = nn.Parameter(torch.tensor(-1.0))
        self.raw_phi20 = nn.Parameter(torch.tensor(-1.0))

        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)

        self.raw_nu = nn.Parameter(torch.tensor(4.0))

    def garch_params(self):
        omega = F.softplus(self.raw_omega)
        alpha = torch.sigmoid(self.raw_alpha)
        beta = torch.sigmoid(self.raw_beta)

        scale = alpha + beta + 1e-6
        alpha = alpha / scale * 0.95
        beta = beta / scale * 0.95

        lambda_ = torch.sigmoid(self.raw_lambda)

        phi1 = torch.sigmoid(self.raw_phi1)
        phi5 = torch.sigmoid(self.raw_phi5)
        phi20 = torch.sigmoid(self.raw_phi20)

        phi_sum = phi1 + phi5 + phi20 + 1e-6
        phi1 = phi1 / phi_sum * 0.6
        phi5 = phi5 / phi_sum * 0.3
        phi20 = phi20 / phi_sum * 0.1

        return omega, alpha, beta, lambda_, phi1, phi5, phi20

    def student_nu(self):
        return torch.clamp(F.softplus(self.raw_nu) + 2, min=4)

    def _rolling_mean(self, squared_returns, end_idx, window):
        start_idx = max(0, end_idx - window + 1)
        window_slice = squared_returns[:, start_idx:end_idx + 1]
        return window_slice.mean(dim=1)

    def _garch_base_var(self, eps_prev, sigma2_prev, squared_returns, end_idx, omega, alpha, beta, lambda_, phi1, phi5, phi20):
        neg = (eps_prev < 0).float()
        leverage = lambda_ * eps_prev.pow(2) * neg

        rv1 = squared_returns[:, end_idx]
        rv5 = self._rolling_mean(squared_returns, end_idx, 5)
        rv20 = self._rolling_mean(squared_returns, end_idx, 20)

        return omega + alpha * eps_prev.pow(2) + beta * sigma2_prev + leverage + phi1 * rv1 + phi5 * rv5 + phi20 * rv20

    def _build_garch_path(self, returns):
        omega, alpha, beta, lambda_, phi1, phi5, phi20 = self.garch_params()

        _, sequence_length = returns.shape

        squared_returns = returns.pow(2)

        sigma2_t = squared_returns[:, 0] + 1e-6
        sigma2_path = [sigma2_t]
        features = [
            torch.stack(
                [
                    torch.zeros_like(sigma2_t),
                    torch.log(sigma2_t + 1e-8),
                ],
                dim=1,
            )
        ]

        for t in range(1, sequence_length):
            eps_prev = returns[:, t - 1]
            sigma2_prev = sigma2_path[-1]

            base_var = self._garch_base_var(
                eps_prev=eps_prev,
                sigma2_prev=sigma2_prev,
                squared_returns=squared_returns,
                end_idx=t - 1,
                omega=omega,
                alpha=alpha,
                beta=beta,
                lambda_=lambda_,
                phi1=phi1,
                phi5=phi5,
                phi20=phi20,
            )
            sigma2_t = torch.clamp(base_var, min=1e-8)
            sigma2_path.append(sigma2_t)
            features.append(
                torch.stack(
                    [
                        eps_prev / torch.sqrt(sigma2_prev + 1e-8),
                        torch.log(sigma2_prev + 1e-8),
                    ],
                    dim=1,
                )
            )

        sigma2_path = torch.stack(sigma2_path, dim=1)
        features = torch.stack(features, dim=1)
        return sigma2_path, features

    def _transformer_correction(self, features):
        seq_len = features.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")

        x = self.input_proj(features)
        x = x + self.pos_embedding[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=features.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)
        return self.output_layer(x[:, -1, :]).squeeze(-1)

    def forecast_next_variance(self, returns):
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)

        if returns.dim() != 2:
            raise ValueError("returns must have shape (batch_size, sequence_length) or (sequence_length,)")

        omega, alpha, beta, lambda_, phi1, phi5, phi20 = self.garch_params()
        sigma2_path, features = self._build_garch_path(returns)
        correction = self._transformer_correction(features)

        squared_returns = returns.pow(2)
        eps_prev = returns[:, -1]
        sigma2_prev = sigma2_path[:, -1]
        base_var_next = self._garch_base_var(
            eps_prev=eps_prev,
            sigma2_prev=sigma2_prev,
            squared_returns=squared_returns,
            end_idx=returns.size(1) - 1,
            omega=omega,
            alpha=alpha,
            beta=beta,
            lambda_=lambda_,
            phi1=phi1,
            phi5=phi5,
            phi20=phi20,
        )

        sigma2_next = base_var_next * (1.0 + self.correction_scale * torch.tanh(correction))
        return torch.clamp(sigma2_next, min=1e-8)

    def negative_log_likelihood(self, returns):
        sigma2_next = self.forecast_next_variance(returns)
        nu = self.student_nu()
        eps = returns[:, -1] if returns.dim() == 2 else returns[-1]
        eps = eps.unsqueeze(0) if eps.dim() == 0 else eps
        term1 = 0.5 * torch.log(sigma2_next)
        term2 = (nu + 1) / 2 * torch.log(1 + eps.pow(2) / ((nu - 2) * sigma2_next))
        const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log((nu - 2) * math.pi)
        return (term1 + term2 - const).mean()

    def forward(self, returns):
        omega, alpha, beta, lambda_, phi1, phi5, phi20 = self.garch_params()
        del omega, alpha, beta, lambda_, phi1, phi5, phi20
        return self.forecast_next_variance(returns)