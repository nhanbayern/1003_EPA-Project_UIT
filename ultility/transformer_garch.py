import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerGARCH(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, max_len=512):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

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

    def forward(self, returns):
        omega, alpha, beta, lambda_, phi1, phi5, phi20 = self.garch_params()
        nu = self.student_nu()

        batch_size, T = returns.shape

        sigma2_list = []
        sigma2_t = returns[:, 0] ** 2 + 1e-6
        sigma2_list.append(sigma2_t)

        squared_returns_history = [returns[:, 0] ** 2]
        features_seq = []

        for t in range(1, T):
            eps_prev = returns[:, t - 1]

            shock = eps_prev / torch.sqrt(sigma2_t + 1e-8)
            neg = (eps_prev < 0).float()
            leverage = lambda_ * eps_prev ** 2 * neg

            RV1 = eps_prev ** 2

            if len(squared_returns_history) >= 5:
                RV5 = torch.stack(squared_returns_history[-5:], dim=1).mean(dim=1)
            else:
                RV5 = torch.stack(squared_returns_history, dim=1).mean(dim=1)

            if len(squared_returns_history) >= 20:
                RV20 = torch.stack(squared_returns_history[-20:], dim=1).mean(dim=1)
            else:
                RV20 = torch.stack(squared_returns_history, dim=1).mean(dim=1)

            base_var = omega + alpha * eps_prev ** 2 + beta * sigma2_t + leverage + phi1 * RV1 + phi5 * RV5 + phi20 * RV20

            log_sigma = torch.log(base_var + 1e-8)

            feat = torch.stack([shock, log_sigma], dim=1)
            features_seq.append(feat)

            seq_tensor = torch.stack(features_seq, dim=1)

            x = self.input_proj(seq_tensor)
            x = x + self.pos_embedding[:, :x.size(1), :]

            mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
            x = self.transformer(x, mask=mask)

            correction = self.output_layer(x[:, -1, :]).squeeze(1)

            sigma2_t = base_var * (1 + 0.05 * torch.tanh(correction))
            sigma2_t = torch.clamp(sigma2_t, min=1e-8)
            sigma2_list.append(sigma2_t)

            squared_returns_history.append(eps_prev ** 2)

        sigma2 = torch.stack(sigma2_list, dim=1)

        eps = returns / torch.sqrt(sigma2)

        term1 = 0.5 * torch.log(sigma2)
        term2 = (nu + 1) / 2 * torch.log(1 + eps ** 2 / ((nu - 2) * sigma2))
        const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log((nu - 2) * math.pi)

        nll = term1 + term2 - const

        return nll.mean(), sigma2