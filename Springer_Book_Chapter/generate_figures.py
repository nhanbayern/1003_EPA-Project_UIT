r"""
generate_figures.py
Generates custom figures for the Springer LNCS Book Chapter paper.
Output: fig1.png, fig2.png, fig3.png in the same directory as this script.

Usage:
    cd D:\UIT\1003_EPA_PROJECT\1.0.0\1003_EPA-Project_UIT\Springer_Book_Chapter
    python generate_figures.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.dirname(SCRIPT_DIR)

MCDM_RUN = os.path.join(BASE, "output", "mcdm_results", "MCDM20260727104251")
MCDM_55  = os.path.join(MCDM_RUN, "5,5")
MCDM_37  = os.path.join(MCDM_RUN, "3,7")
OUT_DIR  = SCRIPT_DIR   # all PNGs go directly into Springer_Book_Chapter/

DPI = 300

# Colours — B&W legible
C_PROPOSED   = "#1a5276"
C_FOUNDATION = "#2980b9"
C_GARCH      = "#7f8c8d"
C_TRANS      = "#bdc3c7"
C_EXCLUDED   = "#c0392b"
C_THRESH     = "#e74c3c"

# ─────────────────────────────────────────────
# Helper: draw a rounded box
# ─────────────────────────────────────────────
def _box(ax, cx, cy, w, h, text, fc, fontsize=7.5, bold=False):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.1",
                       facecolor=fc, edgecolor="#2c3e50", lw=0.8, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")


def _arr(ax, src, dst, color="#2c3e50", label=""):
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9), zorder=2)
    if label:
        mx = (src[0] + dst[0]) / 2 + 0.15
        my = (src[1] + dst[1]) / 2
        ax.text(mx, my, label, fontsize=7, color=color, zorder=5)


# ═══════════════════════════════════════════════════════════
# FIG 1 — MSE rank vs VaR 1% Pass Rate rank scatter
# ═══════════════════════════════════════════════════════════
def fig1_rank_scatter():
    print("[Fig 1] MSE rank vs VaR Pass Rate rank …")
    fpath = os.path.join(MCDM_55, "MCDMInputMetrics.csv")
    if not os.path.exists(fpath):
        print(f"  SKIP: {fpath} not found.")
        return

    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    cmap = {c.lower(): c for c in df.columns}

    # Find columns
    mse_col   = next((cmap[k] for k in cmap if k == "mse"), None)
    var1_col  = next((cmap[k] for k in cmap if "pass" in k and "1" in k), None)
    model_col = next((cmap[k] for k in cmap if "display" in k or "model" in k), None)

    if mse_col is None or var1_col is None:
        print(f"  SKIP: needed columns not found. Columns: {list(df.columns)}")
        return

    df = df.dropna(subset=[mse_col, var1_col])
    df["mse_rank"] = df[mse_col].rank(ascending=True)
    df["var_rank"] = df[var1_col].rank(ascending=False)

    def classify(name):
        n = str(name).lower()
        if "garch" in n and ("autoformer" in n or "hybrid" in n):
            return "GARCH-Autoformer (proposed)", C_PROPOSED, "o", 70
        if "moiraivar" in n or "moirai_var" in n or ("moirai" in n and "var" in n):
            return "MoiraiVaR (proposed)", C_FOUNDATION, "s", 70
        if "moirai" in n:
            return "Moirai Baseline", "#5dade2", "D", 50
        if any(x in n for x in ["garch", "figarch", "gjr"]):
            return "GARCH Baseline", C_GARCH, "^", 50
        return "Transformer Baseline", C_TRANS, "v", 40

    labels, cols, markers, sizes = zip(*[classify(r) for r in df[model_col]]) if model_col else (
        ["?"] * len(df), [C_TRANS]*len(df), ["o"]*len(df), [40]*len(df))
    df["label"] = labels
    df["col"]   = cols
    df["mkr"]   = markers
    df["sz"]    = sizes

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    seen = set()
    for _, row in df.iterrows():
        lbl = row["label"] if row["label"] not in seen else "_"
        seen.add(row["label"])
        ax.scatter(row["mse_rank"], row["var_rank"],
                   c=row["col"], marker=row["mkr"], s=row["sz"],
                   alpha=0.88, label=lbl, zorder=3)

    maxr = max(df["mse_rank"].max(), df["var_rank"].max()) + 1
    ax.plot([1, maxr], [1, maxr], "k--", lw=0.7, alpha=0.4, label="Perfect alignment")

    ax.set_xlabel("MSE Rank  (1 = lowest MSE)", fontsize=9)
    ax.set_ylabel("VaR 1% Pass Rate Rank  (1 = highest)", fontsize=9)
    ax.set_title("Forecast Accuracy vs. VaR Calibration Rank", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8, borderpad=0.4)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════
# FIG 2 — 3-tier MCDM Pipeline Flowchart
# ═══════════════════════════════════════════════════════════
def fig2_pipeline():
    print("[Fig 2] MCDM pipeline flowchart …")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    C1 = "#d6eaf8"; C2 = "#fef9e7"; C3 = "#e9f7ef"; CG = "#fadbd8"

    # ── Tier 1 ──
    ax.text(5, 9.65, "Tier 1 — Per-Case Metric Computation",
            ha="center", fontsize=8.5, fontweight="bold", color="#1a5276")
    _box(ax, 5, 9.05, 8.5, 0.75,
         "Forecast & Observation  (model × market × horizon)", C1, fontsize=7.5)
    _box(ax, 5, 8.1, 8.5, 0.75,
         "9 Criteria: MSE · MAE · QLIKE · Std-Ratio Err · Tracking Corr\n"
         "VaR 1% Pass · AVE · VaR 5% Pass · AVE", C1, fontsize=7.2)
    _arr(ax, (5, 7.73), (5, 7.1))

    # ── Gate ──
    ax.text(5, 6.95, "Forecast Sanity Gate",
            ha="center", fontsize=8, fontweight="bold", color="#922b21")
    # Diamond
    gx, gy, gw, gh = 5, 6.25, 3.8, 0.95
    pts = np.array([[gx, gy+gh/2], [gx+gw/2, gy], [gx, gy-gh/2], [gx-gw/2, gy]])
    ax.add_patch(plt.Polygon(pts, closed=True, fc=CG, ec="black", lw=0.8, zorder=3))
    ax.text(gx, gy, "e_σ ≤ 0.9  AND  ρ > 0 ?", ha="center", va="center", fontsize=7.5, zorder=4)

    # Fail
    _arr(ax, (gx - gw/2, gy), (1.5, gy), color=C_EXCLUDED, label="No")
    _box(ax, 1.0, 5.0, 2.1, 1.3,
         "Excluded\n5 configs\n(flat /\nwrong-dir.)", "#fadbd8", fontsize=7)

    # Pass
    _arr(ax, (gx, gy - gh/2), (gx, 4.6), label="  Yes")

    # ── Tier 2 ──
    ax.text(5, 4.45, "Tier 2 — MCDM Ranking  (21 valid configs)",
            ha="center", fontsize=8.5, fontweight="bold", color="#1e8449")

    for xc, label in [(3.2, "50:50\nAccuracy = Risk"), (6.8, "30:70\nRisk-priority")]:
        _box(ax, xc, 3.75, 3.1, 0.78, label, C2, fontsize=7.5)
        _arr(ax, (xc, 3.36), (xc, 2.78))
        _box(ax, xc, 2.4, 3.1, 0.62, "SAW + TOPSIS", C3, fontsize=7.5)
        _arr(ax, (xc, 2.09), (xc, 1.6))

    # ── Tier 3 ──
    _box(ax, 5, 1.25, 8.5, 0.65,
         "Tier 3 — Exact Binomial Dominance Test (n=15 families)   H₁: P(win)>0.5", C3, fontsize=7.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════
# FIG 3 — 2-panel Architecture Diagram
# ═══════════════════════════════════════════════════════════
def fig3a_architecture():
    print("[Fig 3a] Architecture diagram GARCH-Autoformer (Horizontal) …")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis("off")

    def B(cx, cy, w, h, txt, fc, fs=7.2, bold=False):
        _box(ax, cx, cy, w, h, txt, fc, fontsize=fs, bold=bold)

    def A(src, dst, lbl=""):
        _arr(ax, src, dst, label=lbl)

    CI="#d6eaf8"; CS1="#d5f5e3"; CS2="#fef9e7"; CS3="#fdebd0"; CO="#e8daef"

    B(1.5, 3.0, 2.4, 0.8, "60-Day Log-Returns $r_t$", CI, bold=True)
    
    A((2.7, 3.2), (3.6, 4.5))
    A((2.7, 2.8), (3.6, 1.5))

    ax.text(5.0, 5.3, "Stage 1", fontsize=7, fontweight="bold", color="#1e8449", ha="center")
    B(5.0, 4.5, 2.8, 0.8, "GJR-GARCH(1,1,1)\n$\\to \hat{\sigma}_{GARCH}$", CS1, fs=7)
    
    ax.text(5.0, 2.3, "Stage 2", fontsize=7, fontweight="bold", color="#7d6608", ha="center")
    B(5.0, 1.5, 2.8, 0.8, "$z_t = r_t / \hat{\sigma}_{GARCH}$\n(Standardised Residual)", CS2, fs=7)
    
    A((6.4, 1.5), (7.0, 1.5))
    B(8.3, 1.5, 2.6, 0.8, "Series Decomposition\n(Moving Average, $k=5$)", CS2, fs=7)
    
    A((9.6, 1.5), (10.2, 1.5))
    B(11.5, 1.5, 2.6, 0.9, "Auto-Correlation\nDimension: 32 / 128\nHeads: 4 / 8\nLayers: 2 / 3", CS2, fs=6.7)
    
    A((12.8, 1.5), (13.4, 1.5))
    B(14.7, 1.5, 2.6, 0.8, "Residual Head (GELU)\n$\\to \hat{e}_{pred}$", CS2, fs=7)
    
    ax.text(14.7, 5.3, "Stage 3", fontsize=7, fontweight="bold", color="#922b21", ha="center")
    B(14.7, 4.5, 4.5, 0.8, "Ensemble: Softplus$(\hat{\sigma}_{GARCH}^2 + \hat{e}_{pred})$\n$\\to \hat{\sigma}_{final}$", CS3, fs=7)
    
    A((14.7, 1.9), (14.7, 4.1))
    A((6.4, 4.5), (12.45, 4.5))
    
    A((16.95, 4.5), (17.55, 4.5))
    B(18.7, 4.5, 2.3, 0.8, "$\\text{VaR}_{\\alpha} = \mu + \hat{\sigma}_{final}$\n$\\cdot q_{\\alpha}(\\text{Student-}t)$", CO, fs=7)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(OUT_DIR, "fig3a.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def fig3b_architecture():
    print("[Fig 3b] Architecture diagram MoiraiVaR (Horizontal) …")
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.set_xlim(0, 22); ax.set_ylim(0, 3); ax.axis("off")

    def B2(cx, cy, w, h, txt, fc, fs=7.2, bold=False):
        _box(ax, cx, cy, w, h, txt, fc, fontsize=fs, bold=bold)

    def A2(src, dst):
        _arr(ax, src, dst)

    CB="#d6eaf8"; CM="#d5f5e3"; CL="#fef9e7"; CO2="#e8daef"; CLOSS="#fdebd0"

    # Input
    B2(1.5, 1.8, 2.4, 0.8, "60-Day Log-Returns $r_t$", CB, bold=True)
    A2((2.7, 1.8), (3.3, 1.8))
    
    # Pad
    B2(4.5, 1.8, 2.4, 0.8, "Zero-Padding (60 $\\to$ 64)\n4 Patches $\\times$ 16 Steps", CB, fs=7.2)
    A2((5.7, 1.8), (6.3, 1.8))
    
    # Backbone
    B2(8.1, 1.8, 3.6, 0.8, "Pretrained Foundation Model\n(Moirai / Moirai 2 / Moirai-MoE)", CM, fs=7)
    A2((9.9, 1.8), (10.5, 1.8))
    
    # Pooling
    B2(11.8, 1.8, 2.6, 0.8, "Mean Pooling\n$\\to$ Feature Vector", CM, fs=7.2)
    A2((13.1, 1.8), (13.7, 1.8))
    
    # MLP
    B2(15.3, 1.8, 3.2, 0.9, "MLP Head:\nLinear $\\to$ ReLU\n$\\to$ Dropout $\\to$ Linear", CL, fs=7)
    A2((16.9, 1.8), (17.5, 1.8))
    
    # Loss/Output
    B2(19.3, 1.8, 3.6, 1.1, "Forecasts $\hat{\sigma}_h$ for $h \in \{1, 3, 5, 10, 21\}$\n\nLoss = MSE + $\lambda \cdot$ Pinball$(\\text{VaR}_{1\\%})$", CLOSS, fs=7)
    
    # Text below backbone
    ax.text(8.1, 0.8, "Head-Only or Full Fine-Tuning", ha="center", fontsize=7)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(OUT_DIR, "fig3b.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ═══════════════════════════════════════════════════════════
# FIG Extra — MAE and MSE Bar Charts
# ═══════════════════════════════════════════════════════════
def fig_extra_mae_mse():
    print("[Fig Extra] MAE and MSE bar charts from raw predictions …")
    raw_pred_path = os.path.join(BASE, "output", "merged_all_predictions_24_7.csv")
    if not os.path.exists(raw_pred_path):
        print(f"  SKIP: {raw_pred_path} not found.")
        return

    # Read raw predictions (might take a few seconds due to size)
    df_raw = pd.read_csv(raw_pred_path, usecols=['branch', 'tier', 'model', 'true_volatility', 'predict_volatility'])
    
    # Calculate errors
    df_raw['se'] = (df_raw['predict_volatility'] - df_raw['true_volatility']) ** 2
    df_raw['ae'] = (df_raw['predict_volatility'] - df_raw['true_volatility']).abs()
    
    # Group by branch, tier, model
    grouped = df_raw.groupby(['branch', 'tier', 'model'], dropna=False)[['se', 'ae']].mean().reset_index()
    grouped.rename(columns={'se': 'mse', 'ae': 'mae'}, inplace=True)
    
    # Create display names
    def make_display_name(row):
        m = str(row['model'])
        if m == "HybridGARCHAutoformer": m = "GARCH-Autoformer"
        elif m.lower() == "moirai": m = "Moirai"
        elif m.lower() == "moirai2": m = "Moirai 2"
        elif m.lower() in ["moirai_moe", "moirai-moe"]: m = "Moirai-MoE"
        else:
            # Fallback for any other model names: Replace underscores, capitalize
            m = m.replace('_', '-').title()
        
        b = str(row['branch'])
        if b == "Moirai_VAR":
            m = f"MoiraiVaR - {m}"
            
        t = str(row['tier'])
        if "Tier" in t:
            parts = t.split("_")
            if len(parts) >= 2:
                return f"{m} (Tier {parts[1]})"
        elif "lambda" in t:
            l_val = t.split("_")[-1] if "_" in t else t
            return f"{m} (lambda={l_val})"
            
        # Clean up any leftover snake case just in case
        return m.replace('_', '-')
        
    grouped['display_name'] = grouped.apply(make_display_name, axis=1)
    
    df_clean = grouped.dropna(subset=["mse", "mae", "display_name"])
    
    # Define colors based on grouping for a better look
    def get_color(name):
        n = str(name).lower()
        if "garch-autoformer" in n: return C_PROPOSED
        if "moiraivar" in n or "moirai" in n: return C_FOUNDATION
        if "garch" in n: return C_GARCH
        return C_TRANS
        
    df_clean["color"] = df_clean["display_name"].apply(get_color)
    
    # Plot MSE
    df_mse = df_clean.sort_values("mse", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df_mse["display_name"], df_mse["mse"], color=df_mse["color"])
    ax.set_xlabel("Mean Squared Error (MSE)", fontsize=10, fontweight="bold")
    ax.set_title("MSE Comparison across Models (Lower is Better)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_mse = os.path.join(OUT_DIR, "fig_mse_comparison.png")
    fig.savefig(out_mse, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_mse}")
    
    # Plot MAE
    df_mae = df_clean.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df_mae["display_name"], df_mae["mae"], color=df_mae["color"])
    ax.set_xlabel("Mean Absolute Error (MAE)", fontsize=10, fontweight="bold")
    ax.set_title("MAE Comparison across Models (Lower is Better)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_mae = os.path.join(OUT_DIR, "fig_mae_comparison.png")
    fig.savefig(out_mae, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_mae}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")
    fig1_rank_scatter()
    fig2_pipeline()
    fig3a_architecture()
    fig3b_architecture()
    fig_extra_mae_mse()
    print("\nAll custom figures generated.")
