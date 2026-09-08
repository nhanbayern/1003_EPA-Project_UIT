import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_pareto():
    out_dir = r"d:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\paper - Copy"
    
    # Data from Table 3
    data = {
        "Configuration": ["Moirai 2", "Moirai-MoE", "MoiraiVaR-MoE", "GJR-GARCH"],
        "MSE": [0.02275, 0.02476, 0.02526, 0.19112],
        "PR_05": [0.289, 0.304, 0.393, 0.526]
    }
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#2c3e50"]
    markers = ["o", "s", "D", "^"]
    
    # Plot each point
    for i, row in df.iterrows():
        ax.scatter(row["MSE"], row["PR_05"], color=colors[i], marker=markers[i], s=80, label=row["Configuration"], zorder=3)
        ax.annotate(row["Configuration"], (row["MSE"], row["PR_05"]), 
                    xytext=(6, -2), textcoords='offset points', fontsize=8.5)

    # Draw Pareto frontier (Moirai 2 -> MoiraiVaR -> GJR-GARCH)
    # They are non-dominated. Moirai-MoE is dominated by Moirai 2.
    pareto_x = [0.02275, 0.02526, 0.19112]
    pareto_y = [0.289, 0.393, 0.526]
    
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, zorder=2, label="Pareto Frontier")
    
    ax.set_xlabel("Mean Squared Error (Lower is Better)")
    ax.set_ylabel("VaR 5% Pass Rate (Higher is Better)")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_pareto_frontier.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_pareto()
