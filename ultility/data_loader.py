
import pandas as pd
import numpy as np

def get_splits_for_dataset(series, splits_df, dataset_name, seq_len=60):
    if dataset_name not in splits_df.index:
        raise ValueError(f"Dataset '{dataset_name}' not found in splits_df.")

    s = series.dropna().values.astype(np.float32)

    train_len = int(splits_df.loc[dataset_name, "Train"])
    val_len = int(splits_df.loc[dataset_name, "Val"])
    test_len = int(splits_df.loc[dataset_name, "Test"])

    if train_len + val_len + test_len > len(s):
        # Adjust test_len to use available data
        test_len = len(s) - train_len - val_len

    if train_len <= seq_len or test_len <= seq_len:
         raise ValueError(f"Train ({train_len}) or test ({test_len}) length is too short for seq_len={seq_len}.")

    train_data = s[:train_len]
    val_data = s[train_len : train_len + val_len]
    test_data = s[train_len + val_len : train_len + val_len + test_len]

    return train_data, val_data, test_data
