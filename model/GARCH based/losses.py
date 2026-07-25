import torch
import torch.nn as nn

class TLoss(nn.Module):
    def __init__(self, v=5.0):
        super(TLoss, self).__init__()
        self.v = v

    def forward(self, y_pred_var, y_true_returns):
        y_pred_var = torch.clamp(y_pred_var, min=1e-6)

        term1 = torch.log(y_pred_var) / 2.0
        term2 = ((self.v + 1) / 2.0) * torch.log(
            1 + (y_true_returns ** 2) / ((self.v - 2) * y_pred_var)
        )

        loss = term1 + term2
        return torch.mean(loss)
