import pandas as pd
from pathlib import Path

def get_top1():
    base_dir = Path(r"D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\output")
    new_stats_dir = base_dir / "moirai_var_multi_lambda"
    old_stats_dir = base_dir / "stats_analysis"
    
    metrics_config = {
        "mse": "min",
        "mae": "min",
        "qlike": "min",
        "pass_rate": "max",
        "abs_violation_error": "min",
        "kupiec_p": "max",
        "lr_ind_p": "max"
    }
    
    for var_case, alpha in [("var_5pct", 0.05), ("var_1pct", 0.01)]:
        print(f"\n==============================")
        print(f" TOP 1 CHO TUNG METRIC - {var_case.upper()}")
        print(f"==============================")
        
        try:
            new_df = pd.read_csv(new_stats_dir / var_case / "stats_by_model.csv")
            old_df = pd.read_csv(old_stats_dir / var_case / "stats_by_model.csv")
        except Exception as e:
            continue
            
        new_df["source"] = "New"
        old_df["source"] = "Old"
        
        df = pd.concat([new_df, old_df], ignore_index=True)
        df["abs_violation_error"] = (df["violation_rate"] - alpha).abs()
        
        for metric, direction in metrics_config.items():
            if direction == "min":
                best_idx = df[metric].idxmin()
            else:
                best_idx = df[metric].idxmax()
                
            best_row = df.loc[best_idx]
            model_name = f"{best_row['branch']} ({best_row['tier']}) - {best_row['model']}"
            val = best_row[metric]
            source = best_row['source']
            print(f"- {metric.upper():<20} : {val:<10.6f} | {source:^5} | {model_name}")

if __name__ == "__main__":
    get_top1()
