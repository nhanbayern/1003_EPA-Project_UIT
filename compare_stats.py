import pandas as pd
from pathlib import Path

def compare():
    base_dir = Path(r"D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\output")
    new_stats_dir = base_dir / "moirai_var_multi_lambda"
    old_stats_dir = base_dir / "stats_analysis"
    
    for var_case in ["var_5pct", "var_1pct"]:
        print(f"\n==============================")
        print(f" Comparison for {var_case}")
        print(f"==============================")
        new_df = pd.read_csv(new_stats_dir / var_case / "stats_by_model.csv")
        old_df = pd.read_csv(old_stats_dir / var_case / "stats_by_model.csv")
        
        new_df["source"] = "new (moirai_var_multi_lambda)"
        old_df["source"] = "old (baseline)"
        
        combined = pd.concat([new_df, old_df], ignore_index=True)
        # Sort by pass_rate desc, qlike asc
        combined = combined.sort_values(["pass_rate", "qlike"], ascending=[False, True])
        
        # Print top 15 models
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        cols = ["source", "branch", "tier", "model", "pass_rate", "qlike", "violation_rate", "passed_cases"]
        print(combined[cols].head(15).to_string(index=False))

if __name__ == "__main__":
    compare()
