"""Build an auditable, common-key benchmark from canonical prediction CSVs.

Every input row must already use the project schema and the same causal
definition: origin time t, log_return=r[t+1], and future realized population
standard-deviation target ``std(r[t+1:t+h+1], ddof=0)``.
The merger intentionally retains only keys observed for *every* configuration;
it never pads or imputes a missing forecast.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
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
HORIZONS = {1, 3, 5, 10, 21}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, type=Path,
                        help="Directory recursively containing canonical prediction CSVs; repeat as needed.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", default=[],
                        help="Declared seed(s) for the source runs; repeat if family-specific.")
    parser.add_argument("--command", default="", help="Exact training command(s), for manifest provenance.")
    return parser.parse_args()


def discover_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for folder in input_dirs:
        if not folder.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {folder}")
        files.extend(sorted(folder.rglob("*_predictions.csv")))
    if not files:
        raise FileNotFoundError("No *_predictions.csv files found in supplied input directories")
    return sorted(set(files))


def load_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    frame = frame.loc[:, REQUIRED].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise").astype(int)
    if not set(frame["horizon"]).issubset(HORIZONS):
        raise ValueError(f"{path} contains unsupported horizons")
    for column in ("log_return", "true_volatility", "predict_volatility"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["log_return", "true_volatility", "predict_volatility"]].to_numpy()).all():
        raise ValueError(f"{path} contains non-finite canonical values")
    if (frame["true_volatility"] < 0).any() or (frame["predict_volatility"] < 0).any():
        raise ValueError(f"{path} contains negative volatility values")
    if frame[CONFIG].drop_duplicates().shape[0] != 1:
        raise ValueError(f"{path} contains more than one model configuration")
    if frame.duplicated(KEY).any():
        raise ValueError(f"{path} contains duplicate dataset/time/horizon keys")
    return frame


def main() -> None:
    args = parse_args()
    files = discover_files(args.input_dir)
    frames = [load_file(path) for path in files]
    configurations = [tuple(frame.loc[frame.index[0], CONFIG]) for frame in frames]
    if len(set(configurations)) != len(configurations):
        raise ValueError("Duplicate configuration supplied; each branch/tier/model requires one CSV")

    common = set(map(tuple, frames[0][KEY].itertuples(index=False, name=None)))
    for frame in frames[1:]:
        common &= set(map(tuple, frame[KEY].itertuples(index=False, name=None)))
    if not common:
        raise ValueError("No common dataset/time/horizon keys across all configurations")

    common_index = pd.MultiIndex.from_tuples(sorted(common), names=KEY)
    aligned: list[pd.DataFrame] = []
    reference: pd.DataFrame | None = None
    for frame in frames:
        indexed = frame.set_index(KEY).reindex(common_index).reset_index()
        current_truth = indexed[["log_return", "true_volatility"]]
        if reference is None:
            reference = current_truth
        elif not np.allclose(reference.to_numpy(), current_truth.to_numpy(), rtol=1e-7, atol=1e-9):
            raise ValueError("Configurations disagree on log_return or true_volatility at common keys")
        aligned.append(indexed)

    merged = pd.concat(aligned, ignore_index=True).sort_values(CONFIG + KEY).reset_index(drop=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    merged_path = output_dir / "merged_all_predictions.csv"
    merged.to_csv(merged_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": REQUIRED,
        "key": KEY,
        "common_key_policy": "intersection across every supplied configuration; no imputation",
        "source_files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files],
        "configuration_count": len(configurations),
        "common_key_count": len(common),
        "merged_row_count": len(merged),
        "merged_sha256": sha256(merged_path),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "declared_seeds": args.seed,
        "declared_command": args.command,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("configuration_count", "common_key_count", "merged_row_count", "merged_sha256")}, indent=2))


if __name__ == "__main__":
    main()
