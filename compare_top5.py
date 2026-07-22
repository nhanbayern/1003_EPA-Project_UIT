import pandas as pd
from pathlib import Path

def extract_top_5():
    base_dir = Path(r"D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\output")
    new_stats_dir = base_dir / "moirai_var_multi_lambda"
    old_stats_dir = base_dir / "stats_analysis"
    
    for var_case, alpha in [("var_5pct", 0.05), ("var_1pct", 0.01)]:
        print(f"\n==============================")
        print(f" Top 5 cho {var_case.upper()}")
        print(f"==============================")
        
        try:
            new_df = pd.read_csv(new_stats_dir / var_case / "stats_by_model.csv")
            old_df = pd.read_csv(old_stats_dir / var_case / "stats_by_model.csv")
        except Exception as e:
            print(f"Error reading CSVs: {e}")
            continue
            
        new_df["source"] = "New (moirai_multi_lambda)"
        old_df["source"] = "Old (baseline)"
        
        combined = pd.concat([new_df, old_df], ignore_index=True)
        
        # Calculate Absolute Violation Error
        combined["abs_violation_error"] = (combined["violation_rate"] - alpha).abs()
        
        # Sort by pass_rate desc, qlike asc
        combined = combined.sort_values(["pass_rate", "qlike"], ascending=[False, True])
        
        top5 = combined.head(5)
        
        # Select 8 quantitative metrics:
        # Forecast: mse, mae, qlike
        # Risk: violation_rate, abs_violation_error, kupiec_p, lr_ind_p, pass_rate
        cols_to_show = [
            "source", "branch", "tier", "model",
            "mse", "mae", "qlike", 
            "violation_rate", "abs_violation_error", "kupiec_p", "lr_ind_p", "pass_rate"
        ]
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(top5[cols_to_show].to_string(index=False))

if __name__ == "__main__":
    extract_top_5()
