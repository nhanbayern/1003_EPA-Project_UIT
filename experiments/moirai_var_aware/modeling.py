from __future__ import annotations

from contextlib import nullcontext
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_uni2ts_exports(base: str = "/root/uni2ts/src/uni2ts/model") -> None:
    os.makedirs(base + "/moirai", exist_ok=True)
    os.makedirs(base + "/moirai2", exist_ok=True)
    os.makedirs(base + "/moirai_moe", exist_ok=True)
    open(base + "/moirai/__init__.py", "w", encoding="utf-8").write(
        'from .module import MoiraiModule\n\n__all__ = ["MoiraiModule"]\n'
    )
    open(base + "/moirai2/__init__.py", "w", encoding="utf-8").write(
        'from .module import Moirai2Module\n\n__all__ = ["Moirai2Module"]\n'
    )
    open(base + "/moirai_moe/__init__.py", "w", encoding="utf-8").write(
        'from .module import MoiraiMoEModule\n\n__all__ = ["MoiraiMoEModule"]\n'
    )


class VolatilityFeatureExtractor(nn.Module):
    """Moirai-family backbone that emits pooled representations."""

    def __init__(
        self,
        model_type: str = "moirai",
        size: str = "small",
        device: str = "cuda",
        weights_dir: str = "/root/weights",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.freeze_backbone = freeze_backbone

        if model_type == "moirai":
            from uni2ts.common.torch_util import packed_attention_mask
            from uni2ts.model.moirai import MoiraiModule

            self.backbone = MoiraiModule.from_pretrained(os.path.join(weights_dir, f"moirai-1.1-R-{size}"))
            self.d_model = self.backbone.d_model
            self.packed_attention_mask = packed_attention_mask
            self.max_patch = max(self.backbone.patch_sizes)
        elif model_type == "moirai2":
            from uni2ts.common.torch_util import packed_causal_attention_mask
            from uni2ts.model.moirai2 import Moirai2Module

            self.backbone = Moirai2Module.from_pretrained(os.path.join(weights_dir, f"moirai-2.0-R-{size}"))
            self.d_model = self.backbone.d_model
            self.packed_causal_attention_mask = packed_causal_attention_mask
            self.max_patch = self.backbone.patch_size
        elif model_type == "moirai_moe":
            from uni2ts.common.torch_util import packed_causal_attention_mask
            from uni2ts.model.moirai_moe import MoiraiMoEModule

            self.backbone = MoiraiMoEModule.from_pretrained(os.path.join(weights_dir, f"moirai-moe-1.0-R-{size}"))
            self.d_model = self.backbone.d_model
            self.packed_causal_attention_mask = packed_causal_attention_mask
            self.max_patch = max(self.backbone.patch_sizes)
        else:
            raise ValueError("model_type must be moirai, moirai2, or moirai_moe")

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        if freeze_backbone:
            self.backbone.eval()
        self.backbone.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        patch_size_val = 16
        seq_len_patched = 4
        x_padded = F.pad(x, (64 - x.shape[1], 0))
        target_16 = x_padded.view(batch_size, seq_len_patched, patch_size_val).to(self.device)

        if self.max_patch > patch_size_val:
            target = F.pad(target_16, (0, self.max_patch - patch_size_val))
            observed_mask = torch.zeros_like(target, dtype=torch.bool)
            observed_mask[:, :, :patch_size_val] = True
        else:
            target = target_16
            observed_mask = torch.ones_like(target, dtype=torch.bool)

        sample_id = torch.zeros((batch_size, seq_len_patched), dtype=torch.long, device=self.device)
        time_id = torch.arange(seq_len_patched, dtype=torch.long, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        variate_id = torch.zeros((batch_size, seq_len_patched), dtype=torch.long, device=self.device)

        grad_context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with grad_context:
            if self.model_type == "moirai2":
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                tokens = torch.cat([scaled_target, observed_mask.to(torch.float32)], dim=-1)
                reprs = self.backbone.in_proj(tokens)
                attn_mask = self.packed_causal_attention_mask(sample_id, time_id)
                reprs = self.backbone.encoder(reprs, attn_mask, time_id=time_id, var_id=variate_id)
            elif self.model_type == "moirai":
                patch_size_tensor = torch.full((batch_size, seq_len_patched), patch_size_val, dtype=torch.long, device=self.device)
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                reprs = self.backbone.in_proj(scaled_target, patch_size_tensor)
                reprs = self.backbone.encoder(
                    reprs,
                    self.packed_attention_mask(sample_id),
                    time_id=time_id,
                    var_id=variate_id,
                )
            else:
                patch_size_tensor = torch.full((batch_size, seq_len_patched), patch_size_val, dtype=torch.long, device=self.device)
                loc, scale = self.backbone.scaler(target, observed_mask, sample_id, variate_id)
                scaled_target = (target - loc) / scale
                in_reprs = F.silu(self.backbone.in_proj(scaled_target, patch_size_tensor))
                in_reprs = self.backbone.feat_proj(in_reprs, patch_size_tensor)
                res_reprs = self.backbone.res_proj(scaled_target, patch_size_tensor)
                attn_mask = self.packed_causal_attention_mask(sample_id, time_id)
                reprs = self.backbone.encoder(in_reprs + res_reprs, attn_mask, time_id=time_id, var_id=variate_id)

        return torch.mean(reprs, dim=1)


class VolatilityRegressionModel(nn.Module):
    def __init__(self, extractor, hidden_dim: int = 256, output_dim: int = 5):
        super().__init__()
        self.extractor = extractor
        self.mlp = nn.Sequential(
            nn.Linear(extractor.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            # Volatility is a standard deviation and must be non-negative in
            # both the MSE and VaR terms, not only via a loss-side clamp.
            nn.Softplus(),
        )

    def forward(self, x):
        return self.mlp(self.extractor(x))
