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
    print(f"  Saved → {out}")


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
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════
# FIG 3 — 2-panel Architecture Diagram
# ═══════════════════════════════════════════════════════════
def fig3_architecture():
    print("[Fig 3] Architecture diagram (2-panel) …")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4))

    # ── (a) GARCH-Autoformer ──
    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("(a) GARCH-Autoformer", fontsize=9, fontweight="bold", pad=3)

    def B(cx, cy, w, h, txt, fc, fs=7.2, bold=False):
        _box(ax, cx, cy, w, h, txt, fc, fontsize=fs, bold=bold)

    def A(src, dst, lbl=""):
        _arr(ax, src, dst, label=lbl)

    CI="#d6eaf8"; CS1="#d5f5e3"; CS2="#fef9e7"; CS3="#fdebd0"; CO="#e8daef"

    B(3, 10.35, 5.0, 0.65, "60-day log-returns  rₜ", CI, bold=True)
    # Split to two branches
    ax.annotate("", xy=(1.5, 9.75), xytext=(3, 10.03),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=0.9))
    ax.annotate("", xy=(4.5, 9.75), xytext=(3, 10.03),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=0.9))

    # Stage 1 (left)
    ax.text(0.35, 9.35, "Stage 1", fontsize=7, fontweight="bold", color="#1e8449")
    B(1.5, 9.35, 2.4, 0.7, "GJR-GARCH(1,1,1)\n→ σ̂_GARCH", CS1, fs=7)
    A((1.5, 9.0), (1.5, 8.3))

    # Stage 2 (right)
    ax.text(3.35, 9.35, "Stage 2", fontsize=7, fontweight="bold", color="#7d6608")
    B(4.5, 9.35, 2.4, 0.7, "z_t = r_t / σ̂_GARCH\n(standardised residual)", CS2, fs=7)
    A((4.5, 9.0), (4.5, 8.3))
    B(4.5, 7.85, 2.4, 0.8, "Series Decompose\n(kernel = 5)", CS2, fs=7)
    A((4.5, 7.45), (4.5, 6.75))
    B(4.5, 6.3, 2.4, 0.85,
      "AutoCorrelation\nd_model 32/128\nn_heads 4/8\ne_layers 2/3", CS2, fs=6.7)
    A((4.5, 5.87), (4.5, 5.2))
    B(4.5, 4.75, 2.4, 0.8, "ResidualHead (GELU)\n→ ê_pred", CS2, fs=7)

    # Stage 3
    ax.text(0.35, 3.7, "Stage 3", fontsize=7, fontweight="bold", color="#922b21")
    B(3, 3.7, 5.0, 0.82,
      "Ensemble:  Softplus(σ̂²_GARCH + ê_pred)\n→ σ̂_final = √final_var", CS3, fs=7)
    A((1.5, 8.3), (2.3, 4.11))   # GARCH branch down
    A((4.5, 4.35), (3.7, 4.11))  # residual down

    # Output
    B(3, 2.55, 5.0, 0.7, "VaR_α = μ + σ̂_final · q_α(Student-t, ν=dyn.)", CO, fs=7)
    A((3, 3.29), (3, 2.9))

    # ── (b) MoiraiVaR ──
    ax = axes[1]
    ax.set_xlim(0, 6); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("(b) MoiraiVaR", fontsize=9, fontweight="bold", pad=3)

    def B2(cx, cy, w, h, txt, fc, fs=7.2, bold=False):
        _box(ax, cx, cy, w, h, txt, fc, fontsize=fs, bold=bold)

    def A2(src, dst):
        _arr(ax, src, dst)

    CB="#d6eaf8"; CM="#d5f5e3"; CL="#fef9e7"; CO2="#e8daef"; CLOSS="#fdebd0"

    B2(3, 10.35, 5.0, 0.65, "60-day log-returns  rₜ", CB, bold=True)
    A2((3, 10.03), (3, 9.4))
    B2(3, 9.05, 5.0, 0.65, "Pad 60→64  →  4 patches × 16 values", CB, fs=7.2)
    A2((3, 8.73), (3, 8.05))
    B2(3, 7.65, 5.0, 0.75,
       "Moirai / Moirai2 / Moirai-MoE  (size=small)\n[pretrained backbone]", CM, fs=7)
    A2((3, 7.27), (3, 6.65))
    B2(3, 6.3, 5.0, 0.65, "Mean Pooling  →  feature vector (d_model)", CM, fs=7.2)
    A2((3, 5.98), (3, 5.35))
    B2(3, 4.95, 5.0, 0.75,
       "MLP Head:\nLinear(d_model→256) → ReLU → Dropout(0.2)\n→ Linear(256→5)", CL, fs=7)
    A2((3, 4.57), (3, 3.95))
    B2(3, 3.6, 5.0, 0.65, "σ̂  for  h ∈ {1, 3, 5, 10, 21} days", CO2, fs=7.2)
    A2((3, 3.27), (3, 2.6))
    B2(3, 2.1, 5.2, 1.0,
       "Loss = MSE(σ̂, σ_true)\n+ λ · Pinball(VaR₁%, h=1)\nλ=0.2,  α=0.01,  ν=dynamic", CLOSS, fs=7)
    A2((3, 1.6), (3, 0.95))
    B2(3, 0.65, 5.0, 0.55,
       "head-only (frozen)  or  full fine-tune (lr=1e-5)", CB, fs=7)

    plt.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(OUT_DIR, "fig3.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")
    fig1_rank_scatter()
    fig2_pipeline()
    fig3_architecture()
    print("\nAll custom figures generated.")
