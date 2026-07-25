from __future__ import annotations

import torch

from .losses import VarAwareVolatilityLoss


def _build_optimizer(model, tuning_mode: str, head_lr: float, backbone_lr: float, weight_decay: float):
    if tuning_mode == "head":
        return torch.optim.AdamW(model.mlp.parameters(), lr=head_lr, weight_decay=weight_decay)
    if tuning_mode == "full":
        return torch.optim.AdamW(
            [
                {"params": model.extractor.backbone.parameters(), "lr": backbone_lr},
                {"params": model.mlp.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
    raise ValueError("tuning_mode must be 'head' or 'full'")


def _state_module(model, tuning_mode: str):
    return model.mlp if tuning_mode == "head" else model


def train_model_var_aware(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    device: str = "cuda",
    alpha: float = 0.01,
    lambda_var: float = 0.2,
    distribution: str = "student_t",
    nu: float = 4.0,
    var_horizon_index: int = 0,
    tuning_mode: str = "head",
    max_grad_norm: float = 1.0,
):
    model.to(device)
    optimizer = _build_optimizer(model, tuning_mode, lr, backbone_lr, weight_decay=1e-4)
    criterion = VarAwareVolatilityLoss(alpha, lambda_var, distribution, nu, var_horizon_index=var_horizon_index)
    checkpoint_module = _state_module(model, tuning_mode)
    best_val_loss = float("inf")
    best_state = None
    patience = 7
    patience_counter = 0
    print(f"tuning_mode={tuning_mode} | head_lr={lr:g} | backbone_lr={backbone_lr:g}")

    for epoch in range(epochs):
        if tuning_mode == "head":
            model.extractor.eval()
            model.mlp.train()
        else:
            model.train()
        train_loss = 0.0
        for batch_x, batch_y, batch_return in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_return = batch_return.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss, _parts = criterion(pred, batch_y, batch_return)
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_loss += float(loss.item()) * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss = 0.0
        val_vol = 0.0
        val_var = 0.0
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_return in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_return = batch_return.to(device)
                pred = model(batch_x)
                loss, parts = criterion(pred, batch_y, batch_return)
                batch_n = batch_x.size(0)
                val_loss += float(loss.item()) * batch_n
                val_vol += float(parts["volatility_loss"].item()) * batch_n
                val_var += float(parts["var_quantile_loss"].item()) * batch_n
        val_loss /= len(val_loader.dataset)
        val_vol /= len(val_loader.dataset)
        val_var /= len(val_loader.dataset)

        print(
            "Epoch %02d | Train %.6f | Val %.6f | ValVol %.6f | ValVaR %.6f"
            % (epoch + 1, train_loss, val_loss, val_vol, val_var)
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in checkpoint_module.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("-> Early stopping!")
                break

    if best_state is not None:
        checkpoint_module.load_state_dict(best_state)
    return model


def predict(model, test_loader, device: str = "cuda"):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _batch_return in test_loader:
            pred = model(batch_x.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(batch_y.numpy())
    import numpy as np

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)
