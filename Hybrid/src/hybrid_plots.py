#!/usr/bin/env python3
"""Generate hybrid model plots and bound-coverage visualization."""
import os, sys, warnings, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "Lasso", "src"))
sys.path.insert(0, os.path.join(REPO, "Hybrid", "src"))

from plot_config import (apply_style, setup_figure, save_figure, style_axes,
                         SENSORY_COLORS, SENSORY_ORDER, FONT_SIZE_ANNOTATION,
                         FONT_SIZE_LEGEND, sensory_legend_handles)
from hybrid_analysis import build_analysis_df, hs_bounds, loo_evaluate, metrics, load_attr_from_py

RAW_PATH = os.path.join(REPO, "data", "raw_recipes.py")
OUT_DIR = os.path.join(REPO, "results", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

TASTES = ['sweet', 'sour', 'bitter', 'umami', 'salty']
TASTE_DISPLAY = {'sweet':'Sweet','sour':'Sour','bitter':'Bitter','umami':'Umami','salty':'Salt'}
CHEM_COLS = ['protein_frac','sugar_frac','maillard_potential','salt_frac',
             'water_frac','conc_factor','allium_frac','fermented_frac']
ALPHAS = np.logspace(-3, 1, 30)


def main():
    apply_style()
    raw_recipes = load_attr_from_py(RAW_PATH, "raw_recipes")
    df, _ = build_analysis_df(raw_recipes)
    n = len(df)

    # ── Compute hybrid LOO predictions ────────────────────────────────
    hybrid_preds = {}
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        feat = [f'{t}_hs_mid', f'{t}_voigt'] + CHEM_COLS
        X = df[feat].values.astype(float)
        hybrid_preds[t] = loo_evaluate(X, actual, ALPHAS)

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 1: Hybrid Predicted vs Actual (same style as HS/RV/Lasso)
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Generating hybrid predicted-vs-actual plot...")
    all_pred, all_actual, all_labels = [], [], []
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        pred = hybrid_preds[t]
        all_pred.extend(pred.tolist())
        all_actual.extend(actual.tolist())
        all_labels.extend([t] * len(actual))

    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    p = np.clip(np.array(all_pred), 0, 100)
    a = np.clip(np.array(all_actual), 0, 100)
    r_val, _ = pearsonr(p, a)
    r2 = r2_score(a, p)

    fig, ax = setup_figure(size="single_sq")
    style_axes(ax)
    colors = [SENSORY_COLORS[l] for l in all_labels]
    ax.scatter(p, a, c=colors, alpha=0.85, s=25, linewidths=0.3, edgecolors="#333333")
    ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8)
    ax.text(0.03, 0.97, f"PCC = {r_val:.2f}\n$R^2$ = {r2:.2f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=FONT_SIZE_ANNOTATION,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    handles = sensory_legend_handles(markersize=4)
    handles.append(plt.Line2D([0],[0], color='k', linestyle='--', linewidth=0.8, label='Ideal'))
    ax.legend(handles=handles, loc='lower right', fontsize=FONT_SIZE_LEGEND, frameon=True)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Predicted vs Actual (Hybrid HS + Chemistry)")
    save_figure(fig, os.path.join(OUT_DIR, "hybrid_predicted_vs_actual.png"))
    print(f"    PCC={r_val:.3f}, R2={r2:.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 2: Bound coverage bar chart (for each taste dimension)
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Generating bound coverage chart...")
    fig, ax = setup_figure(size="double")
    style_axes(ax)

    taste_labels = ['Sweet','Sour','Bitter','Umami','Salt']
    below_pcts, within_pcts, above_pcts = [], [], []
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        lo = df[f'{t}_hs_lo'].values
        up = df[f'{t}_hs_up'].values
        below_pcts.append(np.sum(actual < lo) / n * 100)
        within_pcts.append(np.sum((actual >= lo) & (actual <= up)) / n * 100)
        above_pcts.append(np.sum(actual > up) / n * 100)

    x = np.arange(len(taste_labels))
    width = 0.55
    b1 = ax.bar(x, below_pcts, width, label='Below lower bound', color='#4477AA', alpha=0.85)
    b2 = ax.bar(x, within_pcts, width, bottom=below_pcts, label='Within bounds', color='#228833', alpha=0.85)
    b3 = ax.bar(x, above_pcts, width, bottom=[b+w for b,w in zip(below_pcts, within_pcts)],
                label='Above upper bound', color='#EE6677', alpha=0.85)

    # Add percentage labels on the "above" segment
    for i, (b, w, a) in enumerate(zip(below_pcts, within_pcts, above_pcts)):
        if a > 15:
            ax.text(i, b + w + a/2, f'{a:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
        if w > 15:
            ax.text(i, b + w/2, f'{w:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(taste_labels)
    ax.set_ylabel('Percentage of recipes')
    ax.set_title('HS Bound Coverage by Taste Dimension')
    ax.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND, frameon=True)
    ax.set_ylim(0, 105)
    save_figure(fig, os.path.join(OUT_DIR, "bound_coverage.png"))

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 3: Model comparison bar chart (MAE by dimension, HS vs Hybrid)
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Generating model comparison chart...")
    fig, ax = setup_figure(size="double")
    style_axes(ax)

    hs_maes, hyb_maes, voigt_maes, lasso_maes = [], [], [], []
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        hs_maes.append(np.mean(np.abs(df[f'{t}_hs_mid'].values - actual)))
        voigt_maes.append(np.mean(np.abs(df[f'{t}_voigt'].values - actual)))
        hyb_maes.append(np.mean(np.abs(hybrid_preds[t] - actual)))
        # Lasso 5D
        X5 = df[[f'{tt}_voigt' for tt in TASTES]].values
        lp = loo_evaluate(X5, actual, ALPHAS)
        lasso_maes.append(np.mean(np.abs(lp - actual)))

    x = np.arange(len(taste_labels))
    w = 0.2
    ax.bar(x - 1.5*w, hs_maes, w, label='HS midpoint', color='#4477AA', alpha=0.85)
    ax.bar(x - 0.5*w, voigt_maes, w, label='Voigt', color='#CCBB44', alpha=0.85)
    ax.bar(x + 0.5*w, lasso_maes, w, label='Lasso 5D', color='#EE6677', alpha=0.85)
    ax.bar(x + 1.5*w, hyb_maes, w, label='Hybrid HS+chem', color='#228833', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(taste_labels)
    ax.set_ylabel('MAE (0–100 scale)')
    ax.set_title('Mean Absolute Error by Taste Dimension and Model')
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=True)
    save_figure(fig, os.path.join(OUT_DIR, "model_comparison_mae.png"))

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 4: Bias elimination chart (HS bias vs Hybrid bias per dimension)
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Generating bias comparison chart...")
    fig, ax = setup_figure(size="double")
    style_axes(ax)

    hs_biases, hyb_biases = [], []
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        hs_biases.append(np.mean(df[f'{t}_hs_mid'].values - actual))
        hyb_biases.append(np.mean(hybrid_preds[t] - actual))

    x = np.arange(len(taste_labels))
    w = 0.35
    ax.bar(x - w/2, hs_biases, w, label='HS midpoint', color='#4477AA', alpha=0.85)
    ax.bar(x + w/2, hyb_biases, w, label='Hybrid HS+chem', color='#228833', alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(taste_labels)
    ax.set_ylabel('Bias (predicted − actual)')
    ax.set_title('Systematic Bias: HS Midpoint vs Hybrid Model')
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=True)
    save_figure(fig, os.path.join(OUT_DIR, "bias_comparison.png"))

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 5: Updated RMSE boxplot with 4 methods (HS, RV, Lasso, Hybrid)
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Generating 4-method RMSE boxplot...")
    fig, ax = setup_figure(size="double_wide" if hasattr(plt, '_') else "double")
    try:
        fig, ax = setup_figure(size="double_wide")
    except:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.5))
    style_axes(ax)

    method_colors = ['#4477AA', '#CCBB44', '#EE6677', '#228833']
    method_names = ['HS', 'Voigt', 'Lasso', 'Hybrid']

    data_boxes, tick_labels = [], []
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        hs_err = np.abs(df[f'{t}_hs_mid'].values - actual)
        voigt_err = np.abs(df[f'{t}_voigt'].values - actual)

        # Lasso 5D LOO
        X5 = df[[f'{tt}_voigt' for tt in TASTES]].values
        lp = loo_evaluate(X5, actual, ALPHAS)
        lasso_err = np.abs(lp - actual)

        hyb_err = np.abs(hybrid_preds[t] - actual)

        data_boxes.extend([hs_err, voigt_err, lasso_err, hyb_err])
        tick_labels.extend([
            "HS\n",
            "Voigt\n" + TASTE_DISPLAY.get(t, t),
            "Lasso\n",
            "Hybrid\n"
        ])

    bp = ax.boxplot(data_boxes, patch_artist=True, widths=0.6, showfliers=False)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(method_colors[i % 4])
        box.set_alpha(0.85)
        box.set_edgecolor('black')
        box.set_linewidth(0.6)
    for median in bp['medians']:
        median.set_linewidth(1.2)
        median.set_color('black')

    ax.set_ylabel("Absolute Error")
    ax.set_title("Prediction Error by Taste Dimension and Model")
    positions = np.arange(1, len(tick_labels) + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    handles = [plt.Line2D([0],[0], marker='s', linestyle='', markersize=8,
               color=c, label=n) for c, n in zip(method_colors, method_names)]
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZE_LEGEND)
    save_figure(fig, os.path.join(OUT_DIR, "rmse_boxplot_4methods.png"))

    print(f"\n[+] All plots saved to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(OUT_DIR, f))
            print(f"    {f} ({size//1024} KB)")


if __name__ == "__main__":
    main()
