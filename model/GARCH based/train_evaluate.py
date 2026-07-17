import torch
import copy
import numpy as np

def train_garch_lstm_hybrid(
    model,
    criterion,
    train_loader,
    val_loader,
    device="cpu",
    epochs=100,
    learning_rate=1e-2,
    lr_factor=0.5,
    lr_patience=5,
    early_stopping_patience=20,
    min_lr=1e-6,
):
    device = torch.device(device)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for enc_r, dec_v, target_r, _ in train_loader:
            enc_r = enc_r.to(device)
            dec_v = dec_v.to(device)
            target_r = target_r.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            # Predict only horizon=1 during training for GARCH-LSTM density estimation
            pred_var = model.forecast_multi_variance(enc_r, dec_v, max_horizon=1)
            pred_var = pred_var.squeeze(-1) # shape: (batch_size,)
            
            loss = criterion(pred_var, target_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_r, dec_v, target_r, _ in val_loader:
                enc_r = enc_r.to(device)
                dec_v = dec_v.to(device)
                target_r = target_r.to(device)

                pred_var = model.forecast_multi_variance(enc_r, dec_v, max_horizon=1)
                pred_var = pred_var.squeeze(-1)
                val_loss = criterion(pred_var, target_r)
                val_losses.append(float(val_loss.item()))

        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
