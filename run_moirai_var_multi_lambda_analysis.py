import os
import sys
import glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stats_analysis.config import StatsAnalysisConfig
from stats_analysis.analyzer import StatsAnalysisPipeline
from stats_analysis.run_statistical_tests import (
    run_forecast_tests,
    run_var_tests,
    run_dm_for_var_predictions,
    VAR_CASES
)

def merge_predictions():
    pred_dir = PROJECT_ROOT / "results_v2" / "moirai_var_multi_lambda_predictions"
    all_files = glob.glob(str(pred_dir / "*.csv"))
    
    print(f"Found {len(all_files)} CSV files in {pred_dir}")
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
        
    merged = pd.concat(dfs, ignore_index=True)
    merged_path = pred_dir / "merged_predictions.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Merged predictions saved to {merged_path}")
    return merged_path

def run_stats_pipeline(input_csv):
    output_base = PROJECT_ROOT / "output" / "moirai_var_multi_lambda"
    
    for var_case, alpha in VAR_CASES.items():
        out_dir = output_base / var_case
        out_dir.mkdir(parents=True, exist_ok=True)
        
        config = StatsAnalysisConfig(
            input_csv=input_csv,
            output_dir=out_dir,
            alpha=alpha
        )
        print(f"Running pipeline for {var_case} (alpha={alpha})...")
        StatsAnalysisPipeline(config).run(save=True)
        print(f"Finished pipeline for {var_case}.")

def run_stats_tests():
    output_base = PROJECT_ROOT / "output" / "moirai_var_multi_lambda"
    stats_dir = output_base / "statistical_tests"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    friedman_frames = []
    nemenyi_frames = []
    dm_frames = []
    
    for var_case, alpha in VAR_CASES.items():
        case_dir = output_base / var_case
        print(f"Running statistical tests for {var_case}...")
        
        forecast_friedman, forecast_nemenyi = run_forecast_tests(case_dir, var_case, alpha)
        friedman_frames.append(forecast_friedman)
        nemenyi_frames.append(forecast_nemenyi)
        
        var_predictions = pd.read_csv(case_dir / "var_predictions.csv", low_memory=False)
        var_friedman, var_nemenyi = run_var_tests(var_predictions, var_case, alpha)
        dm_results = run_dm_for_var_predictions(var_predictions, alpha, var_case)
        
        friedman_frames.append(var_friedman)
        nemenyi_frames.append(var_nemenyi)
        dm_frames.append(dm_results)
        
    friedman = pd.concat(friedman_frames, ignore_index=True)
    nemenyi = pd.concat(nemenyi_frames, ignore_index=True)
    dm = pd.concat(dm_frames, ignore_index=True)
    
    friedman.to_csv(stats_dir / "friedman_results.csv", index=False)
    nemenyi.to_csv(stats_dir / "nemenyi_pairwise_results.csv", index=False)
    dm.to_csv(stats_dir / "dm_var_pairwise_results.csv", index=False)
    
    print(f"Saved: {stats_dir / 'friedman_results.csv'}")
    print(f"Saved: {stats_dir / 'nemenyi_pairwise_results.csv'}")
    print(f"Saved: {stats_dir / 'dm_var_pairwise_results.csv'}")

if __name__ == "__main__":
    merged_csv = merge_predictions()
    run_stats_pipeline(merged_csv)
    run_stats_tests()
    print("All tasks completed successfully.")
