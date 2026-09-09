from __future__ import annotations

import io
import os
import sys
import zipfile


def run_training_to_zip(
    lambda_var: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
    tuning_mode: str = "head",
    backbone_lr: float = 1e-5,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    dataset_dir: str = "/root/dataset",
    weights_dir: str = "/root/weights",
) -> bytes:
    if tuning_mode not in {"head", "full"}:
        raise ValueError("tuning_mode must be 'head' or 'full'")

    sys.path.insert(0, "/root/uni2ts/src")
    sys.path.insert(0, "/root")

    import torch
    from torch.utils.data import DataLoader

    from moirai_var_aware.config import SPLIT_INFO
    from moirai_var_aware.data import load_and_split_dataset
    from moirai_var_aware.export import build_prediction_frame
    from moirai_var_aware.modeling import (
        VolatilityFeatureExtractor,
        VolatilityRegressionModel,
        patch_uni2ts_exports,
    )
    from moirai_var_aware.train import predict, train_model_var_aware

    patch_uni2ts_exports()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("cuda:", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    selected_models = models or ["moirai", "moirai2", "moirai_moe"]
    selected_datasets = datasets or list(SPLIT_INFO.keys())
    csv_outputs: dict[str, str] = {}

    for index_name in selected_datasets:
        csv_path = os.path.join(dataset_dir, f"{index_name}.csv")
        if not os.path.exists(csv_path):
            print("missing dataset:", csv_path)
            continue

        train_ds, val_ds, test_ds = load_and_split_dataset(csv_path, index_name)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

        import numpy as np
        from scipy.stats import kurtosis
        train_returns = np.array([train_ds.dataset.samples[i]["log_return"] for i in train_ds.indices])
        k = kurtosis(train_returns, fisher=True, nan_policy='omit')
        if k <= 0:
            dynamic_nu = 30.0
        else:
            dynamic_nu = float(np.clip(4.0 + 6.0 / k, 2.1, 30.0))
        print(f"[{index_name}] Kurtosis: {k:.4f} | Calculated Dynamic nu: {dynamic_nu:.4f}")

        for model_type in selected_models:
            print(f"=== {index_name} | {model_type} | lambda_var={lambda_var} | tuning_mode={tuning_mode} | nu={dynamic_nu:.4f} ===")
            extractor = VolatilityFeatureExtractor(
                model_type=model_type,
                size="small",
                device=device,
                weights_dir=weights_dir,
                freeze_backbone=(tuning_mode == "head"),
            )
            model = VolatilityRegressionModel(extractor=extractor)
            model = train_model_var_aware(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=1e-3,
                backbone_lr=backbone_lr,
                device=device,
                alpha=0.01,
                lambda_var=lambda_var,
                distribution="student_t",
                nu=dynamic_nu,
                var_horizon_index=0,
                tuning_mode=tuning_mode,
            )
            val_preds, val_targets = predict(model, val_loader, device=device)
            preds, targets = predict(model, test_loader, device=device)
            val_frame = build_prediction_frame(index_name, model_type, lambda_var, val_ds, val_preds, val_targets, tuning_mode=tuning_mode, split="validation")
            frame = build_prediction_frame(index_name, model_type, lambda_var, test_ds, preds, targets, tuning_mode=tuning_mode, split="test")
            mode_suffix = "" if tuning_mode == "head" else f"_{tuning_mode}"
            filename = f"{index_name}_{model_type}{mode_suffix}_lambda_{lambda_var:g}_predictions.csv"
            csv_outputs[filename] = frame.to_csv(index=False)
            csv_outputs[filename.replace("_predictions.csv", "_validation_predictions.csv")] = val_frame.to_csv(index=False)

            del model
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in csv_outputs.items():
            zf.writestr(filename, content)
    print("created zip files:", len(csv_outputs))
    return buffer.getvalue()
