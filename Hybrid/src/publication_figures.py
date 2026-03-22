#!/usr/bin/env python3
"""
Generate three publication-quality composite figures for the paper.

Figure 1: The analogy and its failure (coverage, HS scatter, bias)
Figure 2: Model performance (hybrid scatter, MAE bars, boxplot)
Figure 3: Data context (ingredient usage, taste distributions, t-SNE)

Run from repo root:
    python3 Hybrid/src/publication_figures.py
"""
import os, sys, warnings, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "Lasso", "src"))
sys.path.insert(0, os.path.join(REPO, "Hybrid", "src"))

from hybrid_analysis import build_analysis_df, hs_bounds, loo_evaluate, load_attr_from_py

RAW_PATH = os.path.join(REPO, "data", "raw_recipes.py")
OUT_DIR = os.path.join(REPO, "results", "publication_figures")
os.makedirs(OUT_DIR, exist_ok=True)

TASTES = ['sweet', 'sour', 'bitter', 'umami', 'salty']
TASTE_DISPLAY = ['Sweet', 'Sour', 'Bitter', 'Umami', 'Salt']
ALPHAS = np.logspace(-3, 1, 30)

# ── Tol Bright palette (colorblind-safe) ──
COL = {
    'sweet': '#4477AA', 'sour': '#AA3377', 'bitter': '#EE6677',
    'umami': '#CCBB44', 'salty': '#228833',
}
METHOD_COL = {
    'HS': '#4477AA', 'Voigt': '#CCBB44', 'Lasso': '#EE6677', 'Hybrid': '#228833',
}
COL_BELOW = '#4477AA'
COL_WITHIN = '#228833'
COL_ABOVE = '#EE6677'

DPI = 600
FONT_FAMILY = 'serif'


def setup_style():
    plt.rcParams.update({
        'font.family': FONT_FAMILY, 'font.size': 8,
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7, 'figure.dpi': 150,
        'savefig.dpi': DPI, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False,
    })


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_label(ax, letter, x=-0.12, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left', fontfamily=FONT_FAMILY)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [+] {name} ({os.path.getsize(path)//1024} KB)")


def main():
    setup_style()
    print("[*] Loading data and computing predictions...")
    raw_recipes = load_attr_from_py(RAW_PATH, "raw_recipes")
    df, _ = build_analysis_df(raw_recipes)
    n = len(df)

    # Compute all predictions
    hybrid_preds, lasso5d_preds = {}, {}
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        # Hybrid
        feat_h = [f'{t}_hs_mid', f'{t}_voigt'] + [
            'protein_frac','sugar_frac','maillard_potential','salt_frac',
            'water_frac','conc_factor','allium_frac','fermented_frac']
        hybrid_preds[t] = loo_evaluate(df[feat_h].values.astype(float), actual, ALPHAS)
        # Lasso 5D
        feat_l = [f'{tt}_voigt' for tt in TASTES]
        lasso5d_preds[t] = loo_evaluate(df[feat_l].values.astype(float), actual, ALPHAS)

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1: The analogy and its failure
    # (A) Bound coverage  (B) HS scatter  (C) Bias comparison
    # ═══════════════════════════════════════════════════════════════════
    print("\n[*] Figure 1: Analogy and failure...")
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.1, 1], wspace=0.38)

    # ── (A) Bound coverage ──
    ax_a = fig.add_subplot(gs[0])
    style_ax(ax_a)
    panel_label(ax_a, 'A')

    below_p, within_p, above_p = [], [], []
    for t in TASTES:
        act = df[f'{t}_actual'].values
        lo, up = df[f'{t}_hs_lo'].values, df[f'{t}_hs_up'].values
        below_p.append(np.sum(act < lo) / n * 100)
        within_p.append(np.sum((act >= lo) & (act <= up)) / n * 100)
        above_p.append(np.sum(act > up) / n * 100)

    x = np.arange(5)
    w = 0.55
    ax_a.bar(x, below_p, w, color=COL_BELOW, alpha=0.85, label='Below')
    ax_a.bar(x, within_p, w, bottom=below_p, color=COL_WITHIN, alpha=0.85, label='Within')
    ax_a.bar(x, above_p, w, bottom=[b+wi for b,wi in zip(below_p, within_p)],
             color=COL_ABOVE, alpha=0.85, label='Above')
    for i, (b, wi, a) in enumerate(zip(below_p, within_p, above_p)):
        if a > 20:
            ax_a.text(i, b + wi + a/2, f'{a:.0f}%', ha='center', va='center',
                      fontsize=6.5, fontweight='bold', color='white')
        if wi > 20:
            ax_a.text(i, b + wi/2, f'{wi:.0f}%', ha='center', va='center',
                      fontsize=6.5, fontweight='bold', color='white')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(TASTE_DISPLAY, fontsize=7)
    ax_a.set_ylabel('Recipes (%)')
    ax_a.set_title('HS bound coverage')
    ax_a.set_ylim(0, 108)
    ax_a.legend(fontsize=6, loc='upper left', frameon=True, framealpha=0.9)

    # ── (B) HS predicted vs actual ──
    ax_b = fig.add_subplot(gs[1])
    style_ax(ax_b)
    panel_label(ax_b, 'B')

    all_p, all_a, all_l = [], [], []
    for t in TASTES:
        p = df[f'{t}_hs_mid'].values
        a = df[f'{t}_actual'].values
        all_p.extend(p); all_a.extend(a); all_l.extend([t]*len(a))
    all_p = np.clip(all_p, 0, 100)
    all_a = np.clip(all_a, 0, 100)
    pcc, _ = pearsonr(all_p, all_a)
    r2 = r2_score(all_a, all_p)
    colors = [COL[l] for l in all_l]
    ax_b.scatter(all_p, all_a, c=colors, alpha=0.75, s=18, linewidths=0.3, edgecolors='#555')
    ax_b.plot([0,100],[0,100], 'k--', lw=0.7, alpha=0.5)
    ax_b.text(0.04, 0.96, f'PCC={pcc:.2f}\n$R^2$={r2:.2f}', transform=ax_b.transAxes,
              va='top', fontsize=7, bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8, ec='gray'))
    ax_b.set_xlim(0, 100); ax_b.set_ylim(0, 100)
    ax_b.set_aspect('equal', adjustable='box')
    ax_b.set_xlabel('Predicted (HS midpoint)')
    ax_b.set_ylabel('Actual (SVT panel)')
    ax_b.set_title('HS predicted vs. actual')
    # Legend
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', markersize=4,
               color=COL[t], label=td) for t, td in zip(TASTES, TASTE_DISPLAY)]
    ax_b.legend(handles=handles, fontsize=5.5, loc='lower right', frameon=True, framealpha=0.9, ncol=1)

    # ── (C) Bias comparison ──
    ax_c = fig.add_subplot(gs[2])
    style_ax(ax_c)
    panel_label(ax_c, 'C')

    hs_bias, hyb_bias = [], []
    for t in TASTES:
        act = df[f'{t}_actual'].values
        hs_bias.append(np.mean(df[f'{t}_hs_mid'].values - act))
        hyb_bias.append(np.mean(hybrid_preds[t] - act))

    w2 = 0.35
    ax_c.bar(x - w2/2, hs_bias, w2, color=METHOD_COL['HS'], alpha=0.85, label='HS midpoint')
    ax_c.bar(x + w2/2, hyb_bias, w2, color=METHOD_COL['Hybrid'], alpha=0.85, label='Hybrid')
    ax_c.axhline(0, color='black', lw=0.7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(TASTE_DISPLAY, fontsize=7)
    ax_c.set_ylabel('Bias (pred. − actual)')
    ax_c.set_title('Systematic bias')
    ax_c.legend(fontsize=6, frameon=True, framealpha=0.9)

    save(fig, 'Figure_1.png')
    save(fig, 'Figure_1.pdf') if False else None  # PDF needs different backend

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 2: Model performance
    # (A) Hybrid scatter  (B) MAE bars  (C) Error boxplot
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Figure 2: Model performance...")
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1, 1.3], wspace=0.38)

    # ── (A) Hybrid predicted vs actual ──
    ax_a = fig.add_subplot(gs[0])
    style_ax(ax_a)
    panel_label(ax_a, 'A')

    all_p, all_a, all_l = [], [], []
    for t in TASTES:
        all_p.extend(hybrid_preds[t].tolist())
        all_a.extend(df[f'{t}_actual'].values.tolist())
        all_l.extend([t]*len(df))
    all_p = np.clip(all_p, 0, 100)
    all_a = np.clip(all_a, 0, 100)
    pcc, _ = pearsonr(all_p, all_a)
    r2 = r2_score(all_a, all_p)
    colors = [COL[l] for l in all_l]
    ax_a.scatter(all_p, all_a, c=colors, alpha=0.75, s=18, linewidths=0.3, edgecolors='#555')
    ax_a.plot([0,100],[0,100], 'k--', lw=0.7, alpha=0.5)
    ax_a.text(0.04, 0.96, f'PCC={pcc:.2f}\n$R^2$={r2:.2f}', transform=ax_a.transAxes,
              va='top', fontsize=7, bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8, ec='gray'))
    ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 100)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.set_xlabel('Predicted (Hybrid)')
    ax_a.set_ylabel('Actual (SVT panel)')
    ax_a.set_title('Hybrid predicted vs. actual')
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', markersize=4,
               color=COL[t], label=td) for t, td in zip(TASTES, TASTE_DISPLAY)]
    ax_a.legend(handles=handles, fontsize=5.5, loc='lower right', frameon=True, framealpha=0.9)

    # ── (B) MAE grouped bars ──
    ax_b = fig.add_subplot(gs[1])
    style_ax(ax_b)
    panel_label(ax_b, 'B')

    methods_mae = {'HS': [], 'Voigt': [], 'Lasso': [], 'Hybrid': []}
    for t in TASTES:
        act = df[f'{t}_actual'].values
        methods_mae['HS'].append(np.mean(np.abs(df[f'{t}_hs_mid'].values - act)))
        methods_mae['Voigt'].append(np.mean(np.abs(df[f'{t}_voigt'].values - act)))
        methods_mae['Lasso'].append(np.mean(np.abs(lasso5d_preds[t] - act)))
        methods_mae['Hybrid'].append(np.mean(np.abs(hybrid_preds[t] - act)))

    bw = 0.18
    for j, (mname, maes) in enumerate(methods_mae.items()):
        offset = (j - 1.5) * bw
        ax_b.bar(x + offset, maes, bw, color=METHOD_COL[mname], alpha=0.85, label=mname)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(TASTE_DISPLAY, fontsize=7)
    ax_b.set_ylabel('MAE')
    ax_b.set_title('Error by dimension')
    ax_b.legend(fontsize=5.5, frameon=True, framealpha=0.9, ncol=2)

    # ── (C) Error boxplot (4 methods) ──
    ax_c = fig.add_subplot(gs[2])
    style_ax(ax_c)
    panel_label(ax_c, 'C')

    box_data, box_labels = [], []
    mcols = [METHOD_COL['HS'], METHOD_COL['Voigt'], METHOD_COL['Lasso'], METHOD_COL['Hybrid']]
    for ti, t in enumerate(TASTES):
        act = df[f'{t}_actual'].values
        errs = [
            np.abs(df[f'{t}_hs_mid'].values - act),
            np.abs(df[f'{t}_voigt'].values - act),
            np.abs(lasso5d_preds[t] - act),
            np.abs(hybrid_preds[t] - act),
        ]
        box_data.extend(errs)
        box_labels.extend(['', TASTE_DISPLAY[ti], '', ''])

    bp = ax_c.boxplot(box_data, patch_artist=True, widths=0.55, showfliers=False)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(mcols[i % 4])
        box.set_alpha(0.8)
        box.set_edgecolor('#333')
        box.set_linewidth(0.5)
    for med in bp['medians']:
        med.set_linewidth(1.0); med.set_color('black')
    for wh in bp['whiskers']:
        wh.set_linewidth(0.5)
    for cap in bp['caps']:
        cap.set_linewidth(0.5)

    # X-axis: label every group of 4 with the taste name
    positions = np.arange(1, len(box_data)+1)
    group_centers = [2.5 + i*4 for i in range(5)]
    ax_c.set_xticks(group_centers)
    ax_c.set_xticklabels(TASTE_DISPLAY, fontsize=7)
    # Add vertical separators
    for sep in [4.5, 8.5, 12.5, 16.5]:
        ax_c.axvline(sep, color='#ddd', lw=0.5, zorder=0)
    ax_c.set_ylabel('|Error|')
    ax_c.set_title('Error distribution')
    ax_c.grid(True, axis='y', linestyle='--', alpha=0.2)
    handles = [plt.Line2D([0],[0], marker='s', linestyle='', markersize=6,
               color=c, label=n) for c, n in zip(mcols, ['HS','Voigt','Lasso','Hybrid'])]
    ax_c.legend(handles=handles, fontsize=5.5, loc='upper right', frameon=True, framealpha=0.9, ncol=2)

    save(fig, 'Figure_2.png')

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 3: Data context
    # (A) Ingredient usage  (B) Taste distributions  (C) t-SNE
    # ═══════════════════════════════════════════════════════════════════
    print("[*] Figure 3: Data context...")
    fig = plt.figure(figsize=(7.0, 3.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # ── (A) Ingredient usage (top ingredients by frequency) ──
    ax_a = fig.add_subplot(gs[0])
    style_ax(ax_a)
    panel_label(ax_a, 'A')

    # Count ingredient occurrences
    from collections import Counter
    ing_counts = Counter()
    for r in raw_recipes:
        for ing in r['ingredients']:
            ing_counts[ing['name']] += 1
    # Top 20
    top = ing_counts.most_common(20)
    names_top = [t[0][:25] for t in reversed(top)]
    counts_top = [t[1] for t in reversed(top)]
    y_pos = np.arange(len(names_top))
    ax_a.barh(y_pos, counts_top, color='#4477AA', alpha=0.8, height=0.7)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names_top, fontsize=5.5)
    ax_a.set_xlabel('Recipes')
    ax_a.set_title('Ingredient frequency (top 20)')

    # ── (B) Taste distributions ──
    ax_b = fig.add_subplot(gs[1])
    style_ax(ax_b)
    panel_label(ax_b, 'B')

    for t, td in zip(TASTES, TASTE_DISPLAY):
        vals = df[f'{t}_actual'].values
        if np.std(vals) > 0.5:
            xgrid = np.linspace(0, max(vals)*1.2, 200)
            kde = gaussian_kde(vals, bw_method=0.3)
            ax_b.plot(xgrid, kde(xgrid), color=COL[t], lw=1.3, label=td)
            ax_b.fill_between(xgrid, kde(xgrid), alpha=0.15, color=COL[t])
    ax_b.set_xlabel('Score (0–100)')
    ax_b.set_ylabel('Density')
    ax_b.set_title('Taste distributions')
    ax_b.legend(fontsize=6, frameon=True, framealpha=0.9)
    ax_b.set_xlim(0, 80)

    # ── (C) t-SNE ──
    ax_c = fig.add_subplot(gs[2])
    style_ax(ax_c)
    panel_label(ax_c, 'C')

    # Build feature matrix for t-SNE
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    feat_mat = []
    recipe_names = []
    for r in raw_recipes:
        total_w = sum(ing['weight'] for ing in r['ingredients'])
        if total_w == 0: continue
        vec = np.zeros(5)
        for ing in r['ingredients']:
            w = ing['weight'] / total_w
            scores = ing['sensory_scores']
            vec += w * np.array([scores.get(t, 0) for t in TASTES])
        feat_mat.append(vec)
        recipe_names.append(r.get('recipe_name', ''))
    feat_mat = np.array(feat_mat)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(15, len(feat_mat)-1),
                random_state=42, max_iter=1000)
    emb = tsne.fit_transform(feat_mat)

    # K-means (K=2 as in paper)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(feat_mat)
    cluster_colors = ['#4477AA', '#EE6677']

    for k in range(2):
        mask = labels == k
        ax_c.scatter(emb[mask, 0], emb[mask, 1], c=cluster_colors[k],
                     alpha=0.7, s=20, linewidths=0.3, edgecolors='#555',
                     label=f'Cluster {k+1}')
    ax_c.set_xlabel('t-SNE 1')
    ax_c.set_ylabel('t-SNE 2')
    ax_c.set_title('Recipe clusters (K=2)')
    ax_c.legend(fontsize=6, frameon=True, framealpha=0.9)

    save(fig, 'Figure_3.png')

    print(f"\n[+] All publication figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
