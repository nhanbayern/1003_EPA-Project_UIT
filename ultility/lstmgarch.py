import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMGARCH(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.raw_omega = nn.Parameter(torch.tensor(-5.0))
        self.raw_alpha = nn.Parameter(torch.tensor(-2.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda = nn.Parameter(torch.tensor(-2.0))

        self.raw_phi1 = nn.Parameter(torch.tensor(-1.0))
        self.raw_phi5 = nn.Parameter(torch.tensor(-1.0))
        self.raw_phi20 = nn.Parameter(torch.tensor(-1.0))

        self.Wf = nn.Parameter(torch.randn(self.hidden_dim))
        self.Uf = nn.Parameter(torch.randn(self.hidden_dim))
        self.bf = nn.Parameter(torch.zeros(self.hidden_dim))

        self.Wi = nn.Parameter(torch.randn(self.hidden_dim))
        self.Ui = nn.Parameter(torch.randn(self.hidden_dim))
        self.bi = nn.Parameter(torch.zeros(self.hidden_dim))

        self.Wc = nn.Parameter(torch.randn(self.hidden_dim))
        self.Uc = nn.Parameter(torch.randn(self.hidden_dim))
        self.bc = nn.Parameter(torch.zeros(self.hidden_dim))

        self.v = nn.Parameter(torch.randn(self.hidden_dim))
        self.w = nn.Parameter(torch.tensor(0.0))

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
        c_t = torch.zeros(batch_size, self.hidden_dim, device=returns.device)
        sigma2_t = returns[:, 0] ** 2 + 1e-6
        sigma2_list.append(sigma2_t)
        squared_returns_history = [returns[:, 0] ** 2]
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
            shock = shock.unsqueeze(1)
            log_sigma = log_sigma.unsqueeze(1)
            f_t = torch.sigmoid(self.Wf * shock + self.Uf * log_sigma + self.bf)
            i_t = torch.sigmoid(self.Wi * shock + self.Ui * log_sigma + self.bi)
            c_hat = torch.tanh(self.Wc * shock + self.Uc * log_sigma + self.bc)
            c_t = f_t * c_t + i_t * c_hat
            correction = torch.tanh((c_t * self.v).sum(dim=1))
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