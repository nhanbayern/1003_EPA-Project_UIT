import os
import json
import re
import pandas as pd

def main():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(notebook_dir, "1003-moirai-based-models-2nd-run.ipynb")
    
    # Define output files path (saved to Morai based/ directory)
    morai_based_dir = os.path.dirname(notebook_dir)
    output_wide_path = os.path.join(morai_based_dir, "comparison_metrics.csv")
    output_tidy_path = os.path.join(morai_based_dir, "comparison_metrics_tidy.csv")
    
    print(f"Reading notebook from: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Search for execution_count 13 or the cell containing `run_comparison_pipeline`
    stdout_text = ""
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            # Check either execution_count == 13 or source text contains pipeline run
            is_cell_13 = cell.get('execution_count') == 13
            source_code = "".join(cell.get('source', []))
            if is_cell_13 or "run_comparison_pipeline" in source_code:
                for out in cell.get('outputs', []):
                    if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
                        stdout_text += out.get('text', '')

    if not stdout_text:
        print("Error: Could not find cell 13 stdout output in the notebook.")
        return

    # Parse metrics from stdout text
    lines = stdout_text.split('\n')
    current_index = None
    
    # We will build list of dictionaries for Option 1 (Tidy) and Option 2 (Wide)
    tidy_rows = []
    
    # Temporary dict to group for Option 2 (Wide)
    # key: (index, metric, model) -> value: dict of {h1, h3, h5, h10, h21}
    wide_groups = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse Index name (e.g., "Bắt đầu huấn luyện cho: DAX_40")
        if "Bắt đầu huấn luyện cho:" in line:
            current_index = line.split("Bắt đầu huấn luyện cho:")[1].strip()
            # Clean up potential leading/trailing equals signs or spaces
            current_index = current_index.replace('=', '').strip()
            i += 1
            continue
        
        # Check for start of metric section (MSE:, MAE:, Q-LIKE:)
        if line in ["MSE:", "MAE:", "Q-LIKE:"]:
            metric_name = line[:-1]  # Remove trailing colon
            i += 1
            # Parse the next 5 lines for horizons h=1, 3, 5, 10, 21
            for _ in range(5):
                if i >= len(lines):
                    break
                h_line = lines[i].strip()
                # Match format: h=1  moirai: 0.019868  moirai2: 0.002866  moirai_moe: 0.006961
                match = re.search(r"h=(\d+)\s+moirai:\s+([\d\.]+)\s+moirai2:\s+([\d\.]+)\s+moirai_moe:\s+([\d\.]+)", h_line)
                if match:
                    h_val = int(match.group(1))
                    moirai_val = float(match.group(2))
                    moirai2_val = float(match.group(3))
                    moirai_moe_val = float(match.group(4))
                    
                    if current_index is None:
                        # Fallback index if parsing header was missed
                        current_index = "DAX_40"
                    
                    # Store for Option 1: Tidy Flat format
                    tidy_rows.append({
                        'index': current_index,
                        'metric': metric_name,
                        'horizon': h_val,
                        'moirai': moirai_val,
                        'moirai2': moirai2_val,
                        'moirai_moe': moirai_moe_val
                    })
                    
                    # Store for Option 2: Wide format
                    # We group values for each model
                    models = ['moirai', 'moirai2', 'moirai_moe']
                    vals = [moirai_val, moirai2_val, moirai_moe_val]
                    for model, val in zip(models, vals):
                        group_key = (current_index, metric_name, model)
                        if group_key not in wide_groups:
                            wide_groups[group_key] = {}
                        wide_groups[group_key][f"h{h_val}"] = val
                        
                i += 1
            continue
        
        i += 1

    if not tidy_rows:
        print("Error: Parsed 0 rows. Please verify cell 13 stdout format.")
        return

    # Write Option 1: Tidy format CSV
    df_tidy = pd.DataFrame(tidy_rows)
    df_tidy.to_csv(output_tidy_path, index=False)
    print(f"Successfully saved Option 1 (Tidy) to: {output_tidy_path}")

    # Write Option 2: Wide format CSV
    wide_rows = []
    for (idx, metric, model), h_dict in wide_groups.items():
        wide_rows.append({
            'index': idx,
            'metric': metric,
            'model': model,
            'h1': h_dict.get('h1', None),
            'h3': h_dict.get('h3', None),
            'h5': h_dict.get('h5', None),
            'h10': h_dict.get('h10', None),
            'h21': h_dict.get('h21', None)
        })
    
    # Sort wide rows to keep indices and metrics grouped logically
    df_wide = pd.DataFrame(wide_rows)
    df_wide.sort_values(by=['index', 'metric', 'model'], inplace=True)
    df_wide.to_csv(output_wide_path, index=False)
    print(f"Successfully saved Option 2 (Wide) to: {output_wide_path}")
    
    # Show preview
    print("\nOption 2 (Wide Pivot) preview:")
    print(df_wide.to_string(index=False))

if __name__ == "__main__":
    main()
