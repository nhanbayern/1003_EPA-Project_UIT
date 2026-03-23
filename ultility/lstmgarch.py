import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class LSTMGARCH(nn.Module):

    def __init__(self):
        super().__init__()

        # ===== GARCH parameters =====
        self.raw_omega = nn.Parameter(torch.tensor(-5.0))
        self.raw_alpha = nn.Parameter(torch.tensor(-2.0))
        self.raw_beta  = nn.Parameter(torch.tensor(0.0))

        # ===== LSTM gates =====
        self.Wf = nn.Parameter(torch.randn(1))
        self.Uf = nn.Parameter(torch.randn(1))
        self.bf = nn.Parameter(torch.zeros(1))

        self.Wi = nn.Parameter(torch.randn(1))
        self.Ui = nn.Parameter(torch.randn(1))
        self.bi = nn.Parameter(torch.zeros(1))

        self.Wc = nn.Parameter(torch.randn(1))
        self.Uc = nn.Parameter(torch.randn(1))
        self.bc = nn.Parameter(torch.zeros(1))

        self.w = nn.Parameter(torch.tensor(0.0))

        # ===== Student-t degrees of freedom =====
        self.raw_nu = nn.Parameter(torch.tensor(3.0))

    def garch_params(self):
        omega = F.softplus(self.raw_omega)
        alpha = torch.sigmoid(self.raw_alpha)
        beta  = torch.sigmoid(self.raw_beta)

        scale = alpha + beta + 1e-6
        alpha = alpha / scale * 0.98
        beta  = beta  / scale * 0.98

        return omega, alpha, beta

    def student_nu(self):
        # enforce nu > 2
        return F.softplus(self.raw_nu) + 2.0

    def forward(self, returns):

        omega, alpha, beta = self.garch_params()
        nu = self.student_nu()

        batch_size, T = returns.shape

        sigma2_list = []
        c_t = torch.zeros(batch_size, device=returns.device)

        sigma2_t = torch.var(returns, dim=1)
        sigma2_list.append(sigma2_t)

        for t in range(1, T):

            eps_prev = returns[:, t-1]

            o_t = omega + alpha * eps_prev**2 + beta * sigma2_t

            f_t = torch.sigmoid(self.Wf * eps_prev + self.Uf * sigma2_t + self.bf)
            i_t = torch.sigmoid(self.Wi * eps_prev + self.Ui * sigma2_t + self.bi)
            c_hat = torch.tanh(self.Wc * eps_prev + self.Uc * sigma2_t + self.bc)

            c_t = f_t * c_t + i_t * c_hat

            sigma2_t = o_t * (1 + self.w * torch.tanh(c_t))
            sigma2_t = torch.clamp(sigma2_t, min=1e-8)

            sigma2_list.append(sigma2_t)

        sigma2 = torch.stack(sigma2_list, dim=1)

        eps = returns

        # ===== Student-t negative log likelihood =====

        term1 = 0.5 * torch.log(sigma2)

        term2 = (nu + 1) / 2 * torch.log(
            1 + eps**2 / ((nu - 2) * sigma2)
        )

        # gamma constant (optional but more correct)
        const = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log((nu - 2) * math.pi)
        )

        nll = term1 + term2 - const

        return nll.mean(), sigma2