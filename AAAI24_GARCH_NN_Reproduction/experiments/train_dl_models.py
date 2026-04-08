from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from AAAI24_GARCH_NN_Reproduction.core.data_processor import (
    DEFAULT_SEQ_LEN,
    create_sliding_windows,
    get_default_dataset_dir,
    load_close_series,
    prepare_aaai24_data,
    print_split_report,
)
from AAAI24_GARCH_NN_Reproduction.models.dl_baselines import (
    SUPPORTED_DL_MODELS,
    build_dataloaders,
    build_dl_model,
    train_dl_model,
)


def resolve_device(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return dev


def _build_val_windows(train_r, train_v, val_r, val_v, seq_len, horizon):
    try:
        return create_sliding_windows(val_r, val_v, seq_len=seq_len, horizon=horizon)
    except ValueError:
        context_r = pd.concat([train_r.iloc[-seq_len:], val_r], axis=0)
        context_v = pd.concat([train_v.iloc[-seq_len:], val_v], axis=0)
        return create_sliding_windows(context_r, context_v, seq_len=seq_len, horizon=horizon)


def train_dl_models_for_dataset(
    dataset_csv,
    model_names=None,
    seq_len=DEFAULT_SEQ_LEN,
    horizon=1,
    epochs=120,
    batch_size=64,
    device=None,
    num_workers=None,
    output_dir=None,
):
    dataset_csv = Path(dataset_csv)
    close = load_close_series(dataset_csv)

    train_split, val_split, test_split = prepare_aaai24_data(close)
    print_split_report(train_split, val_split, test_split, seq_len=seq_len)

    train_r, train_v = train_split
    val_r, val_v = val_split

    train_windows = create_sliding_windows(train_r, train_v, seq_len=seq_len, horizon=horizon)
    val_windows = _build_val_windows(train_r, train_v, val_r, val_v, seq_len=seq_len, horizon=horizon)

    runtime_device = resolve_device(device)
    if num_workers is None:
        num_workers = 2 if runtime_device.type == "cuda" else 0

    train_loader, val_loader = build_dataloaders(
        train_windows=train_windows,
        val_windows=val_windows,
        batch_size=batch_size,
        device=str(runtime_device),
        num_workers=num_workers,
    )

    if model_names is None:
        model_names = list(SUPPORTED_DL_MODELS)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "checkpoints"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in model_names:
        model = build_dl_model(model_name=model_name, seq_len=seq_len)
        model, history = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=str(runtime_device),
            epochs=epochs,
            learning_rate=1e-2,
            lr_factor=0.5,
            lr_patience=5,
            early_stopping_patience=20,
            min_lr=1e-6,
        )

        ckpt_path = output_dir / f"{dataset_csv.stem}_{model_name}.pt"
        torch.save(model.state_dict(), ckpt_path)

        best_val_loss = float(np.min(history["val_loss"])) if history["val_loss"] else np.nan
        rows.append(
            {
                "dataset": dataset_csv.stem,
                "model": model_name,
                "epochs_ran": len(history["train_loss"]),
                "best_val_loss": best_val_loss,
                "checkpoint": str(ckpt_path),
            }
        )

        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train DL baselines for AAAI24 reproduction")
    parser.add_argument("--dataset", type=str, required=True, help="CSV file path or dataset file name")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    dataset_arg = Path(args.dataset)
    if not dataset_arg.exists():
        dataset_arg = get_default_dataset_dir() / args.dataset

    result_df = train_dl_models_for_dataset(
        dataset_csv=dataset_arg,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
