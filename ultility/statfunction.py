import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def compute_rank_ic(prediction):
    if "rank_ic" not in prediction.columns:
        prediction["rank_ic"] = np.nan

    for _, df_t in prediction.groupby("ticker"):
        if df_t["y_pred"].nunique() < 2 or df_t["y_true"].nunique() < 2:
            continue

        ic = spearmanr(
            df_t["y_pred"].values,
            df_t["y_true"].values
        ).correlation

        if np.isnan(ic):
            continue

        prediction.loc[df_t.index, "rank_ic"] = ic
    
    return prediction

