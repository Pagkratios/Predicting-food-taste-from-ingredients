#!/usr/bin/env python3
"""
Generate the right-column evaluation panels for Figure 1 (overview).
Top: Hybrid predicted-vs-actual scatter
Bottom: Bound coverage bar chart (the paper's central result)

Styled to match the existing figure's aesthetic (clean, minimal, publication).
"""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "Lasso", "src"))
sys.path.insert(0, os.path.join(REPO, "Hybrid", "src"))

from hybrid_analysis import build_analysis_df, loo_evaluate, load_attr_from_py

RAW_PATH = os.path.join(REPO, "data", "raw_recipes.py")
OUT_DIR = os.path.join(REPO, "results", "publication_figures")
os.makedirs(OUT_DIR, exist_ok=True)

TASTES = ['sweet', 'sour', 'bitter', 'umami', 'salty']
TASTE_DISPLAY = ['Sweet', 'Sour', 'Bitter', 'Umami', 'Salt']
ALPHAS = np.logspace(-3, 1, 30)

COL = {
    'sweet': '#4477AA', 'sour': '#AA3377', 'bitter': '#EE6677',
    'umami': '#CCBB44', 'salty': '#228833',
}
COL_BELOW = '#4477AA'
COL_WITHIN = '#228833'
COL_ABOVE = '#EE6677'

DPI = 600


def main():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.5, 'figure.dpi': 150,
        'savefig.dpi': DPI, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    raw_recipes = load_attr_from_py(RAW_PATH, "raw_recipes")
    df, _ = build_analysis_df(raw_recipes)
    n = len(df)

    # Compute hybrid predictions
    hybrid_preds = {}
    chem_cols = ['protein_frac','sugar_frac','maillard_potential','salt_frac',
                 'water_frac','conc_factor','allium_frac','fermented_frac']
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        feat = [f'{t}_hs_mid', f'{t}_voigt'] + chem_cols
        hybrid_preds[t] = loo_evaluate(df[feat].values.astype(float), actual, ALPHAS)

    # ── Two-panel figure: scatter on top, coverage below ──
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.8, 6.5),
                                          gridspec_kw={'height_ratios': [1.1, 0.9], 'hspace': 0.35})

    # ═══════════════════════════════════════════════════════
    # TOP: Hybrid Predicted vs Actual
    # ═══════════════════════════════════════════════════════
    ax = ax_top
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_p, all_a, all_l = [], [], []
    for t in TASTES:
        all_p.extend(hybrid_preds[t].tolist())
        all_a.extend(df[f'{t}_actual'].values.tolist())
        all_l.extend([t] * n)

    all_p = np.clip(all_p, 0, 100)
    all_a = np.clip(all_a, 0, 100)
    pcc, _ = pearsonr(all_p, all_a)
    r2 = r2_score(all_a, all_p)

    colors = [COL[l] for l in all_l]
    ax.scatter(all_p, all_a, c=colors, alpha=0.8, s=22, linewidths=0.3, edgecolors='#444')
    ax.plot([0, 100], [0, 100], 'k--', lw=0.8, alpha=0.5)

    ax.text(0.04, 0.96, f'PCC = {pcc:.2f}\n$R^2$ = {r2:.2f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='gray', lw=0.5))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Hybrid: Predicted vs Actual', fontsize=11)

    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=5,
               color=COL[t], markeredgecolor='#444', markeredgewidth=0.3,
               label=td) for t, td in zip(TASTES, TASTE_DISPLAY)]
    handles.append(plt.Line2D([0], [0], color='k', linestyle='--', lw=0.8, label='Ideal'))
    ax.legend(handles=handles, loc='lower right', fontsize=7, frameon=True, framealpha=0.9)

    # ═══════════════════════════════════════════════════════
    # BOTTOM: Bound coverage stacked bars
    # ═══════════════════════════════════════════════════════
    ax = ax_bot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    below_p, within_p, above_p = [], [], []
    for t in TASTES:
        act = df[f'{t}_actual'].values
        lo = df[f'{t}_hs_lo'].values
        up = df[f'{t}_hs_up'].values
        below_p.append(np.sum(act < lo) / n * 100)
        within_p.append(np.sum((act >= lo) & (act <= up)) / n * 100)
        above_p.append(np.sum(act > up) / n * 100)

    x = np.arange(5)
    w = 0.6
    ax.bar(x, below_p, w, color=COL_BELOW, alpha=0.85, label='Below bounds')
    ax.bar(x, within_p, w, bottom=below_p, color=COL_WITHIN, alpha=0.85, label='Within bounds')
    ax.bar(x, above_p, w, bottom=[b + wi for b, wi in zip(below_p, within_p)],
           color=COL_ABOVE, alpha=0.85, label='Above bounds')

    # Percentage labels
    for i, (b, wi, a) in enumerate(zip(below_p, within_p, above_p)):
        if a > 18:
            ax.text(i, b + wi + a/2, f'{a:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
        if wi > 18:
            ax.text(i, b + wi/2, f'{wi:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(TASTE_DISPLAY)
    ax.set_ylabel('Recipes (%)')
    ax.set_title('HS Bound Coverage', fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=7, loc='upper left', frameon=True, framealpha=0.9)

    # Save
    path = os.path.join(OUT_DIR, 'overview_right_panel.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[+] {path} ({os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    main()
