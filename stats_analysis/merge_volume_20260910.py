"""Build a canonical merged benchmark CSV from volume_20260910_000255 test predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "dataset", "branch", "tier", "model", "time", "horizon",
    "log_return", "true_volatility", "predict_volatility",
]
CONFIG = ["branch", "tier", "model"]
KEY = ["dataset", "time", "horizon"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canon_ds(name: object) -> str:
    s = str(name).upper().replace("_MOIRAI", "").replace("-", "_").strip()
    if "DAX" in s:
        return "DAX_40"
    if "EURONEXT" in s:
        return "EURONEXT_100"
    if "IBEX" in s:
        return "IBEX_35"
    if "KOSPI" in s:
        return "KOSPI_INDEX"
    if "NIKKEI" in s:
        return "NIKKEI_225"
    if "SMI" in s:
        return "SMI"
    if "SP500" in s or "SNP500" in s:
        return "SNP500"
    if "VN30" in s:
        return "VN30_INDEX"
    if "VN" in s and "30" not in s:
        return "VN_INDEX"
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume-dir",
        type=Path,
        default=ROOT / "output" / "volume_20260910_000255" / "volume_20260910_000255",
        help="Root path to the volume extraction directory",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "output" / "merged_predictions_volume_20260910.csv",
        help="Path for final merged prediction CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volume_dir = args.volume_dir.resolve()
    output_csv = args.output_csv.resolve()

    norm_dirs = sorted(volume_dir.rglob("normalized_predictions"))
    if not norm_dirs:
        raise FileNotFoundError(f"No normalized_predictions directories found in {volume_dir}")

    all_files: list[Path] = []
    for d in norm_dirs:
        for f in d.glob("*.csv"):
            if "val" not in f.name.lower():
                all_files.append(f)

    print(f"Found {len(all_files)} test prediction CSVs in {volume_dir.name}")
    if len(all_files) != 369:
        print(f"Warning: expected 369 test CSVs, found {len(all_files)}")

    # Group files by configuration (branch, tier, model)
    configs: dict[tuple[str, str, str], list[Path]] = {}
    for f in all_files:
        head = pd.read_csv(f, nrows=1)
        b = str(head["branch"].iloc[0])
        t = str(head["tier"].iloc[0])
        m = str(head["model"].iloc[0])
        if m == "moe":
            m = "moirai_moe"
        key = (b, t, m)
        configs.setdefault(key, []).append(f)

    print(f"Identified {len(configs)} unique model configurations.")

    # Load and combine the 9 market files for each configuration
    config_frames: list[pd.DataFrame] = []
    for (b, t, m), files in sorted(configs.items()):
        market_dfs = []
        for f in files:
            df = pd.read_csv(f, usecols=REQUIRED)
            df["dataset"] = df["dataset"].apply(canon_ds)
            df["branch"] = b
            df["tier"] = t
            df["model"] = m
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
            df["horizon"] = df["horizon"].astype(int)
            for col in ("log_return", "true_volatility", "predict_volatility"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            market_dfs.append(df)
        full_df = pd.concat(market_dfs, ignore_index=True)
        config_frames.append(full_df)

    # Compute intersection of (dataset, time, horizon) across all configurations
    print("Computing common key intersection across all configurations...")
    common = set(map(tuple, config_frames[0][KEY].itertuples(index=False, name=None)))
    for cf in config_frames[1:]:
        common &= set(map(tuple, cf[KEY].itertuples(index=False, name=None)))

    print(f"Common key count: {len(common):,} across all {len(config_frames)} configurations")
    if not common:
        raise ValueError("No common keys across configurations")

    common_index = pd.MultiIndex.from_tuples(sorted(common), names=KEY)
    aligned_frames: list[pd.DataFrame] = []
    reference_truth: pd.DataFrame | None = None

    for cf in config_frames:
        indexed = cf.set_index(KEY).reindex(common_index).reset_index()
        current_truth = indexed[["log_return", "true_volatility"]]
        if reference_truth is None:
            reference_truth = current_truth
        elif not np.allclose(reference_truth.to_numpy(), current_truth.to_numpy(), rtol=1e-5, atol=1e-7):
            b_val = indexed["branch"].iloc[0]
            m_val = indexed["model"].iloc[0]
            raise ValueError(f"Ground truth disagreement in configuration: {b_val} {m_val}")
        aligned_frames.append(indexed)

    merged = pd.concat(aligned_frames, ignore_index=True).sort_values(CONFIG + KEY).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Successfully saved merged predictions to: {output_csv}")
    print(f"Total merged rows: {len(merged):,}")
    print(f"SHA-256: {sha256(output_csv)}")


if __name__ == "__main__":
    main()
