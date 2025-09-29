#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-ready data analysis plots for the 29-recipe dataset (PNG only).

Outputs (ALL at <= 5456x4096 px, 4:3 aspect):
  1) ingredient_usage.png – pie, ≥3-use ingredients, no text on wedges, pro colors, legend top-right
  2) tsne_recipes_ingredients_all_labels.png – t-SNE of recipes (squares) + unique ingredients (circles), ALL labels
  3) tsne_recipes_ingredients_no_labels.png  – same embedding as (2), NO labels
     (both colored by one K-Means model; legend top-right inside axes)
  4) gaussian_<key>.png for each of [sweet, bitter, salty, umami, sour]:
     per-key KDE (SciPy gaussian_kde, Scott’s rule) + Gaussian fit; top-3 farthest (|z|) annotated
  5) combined_distribution.png – pooled KDE of all (excluding sour) + single Gaussian fit
"""


import matplotlib.patches as mpatches


import importlib.util
import random
import os
import numpy as np
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm, gaussian_kde

# ---------------- Paths & constants ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "gaussian_data", "plots_data")
RAW_FILE     = os.path.join(DATA_DIR, "raw_recipes.py")
os.makedirs(PLOTS_DIR, exist_ok=True)

SENSORY_ORDER = ["sweet", "bitter", "salty", "umami", "sour"]

# ---------------- Determinism ----------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------------- Output size control (<= 5456x4096 px, 4:3) ----------------
TARGET_PIX_W = 5456
TARGET_PIX_H = 4096
SAVE_DPI     = 600
FIGSIZE_W    = TARGET_PIX_W / SAVE_DPI
FIGSIZE_H    = TARGET_PIX_H / SAVE_DPI
# --- t-SNE figure sizing (match example, landscape 4:3) ---
TSNE_PIX_W = 4096       # <= 8192x4096 requirement respected
TSNE_PIX_H = 3072
TSNE_FIG_W = TSNE_PIX_W / SAVE_DPI
TSNE_FIG_H = TSNE_PIX_H / SAVE_DPI

# Markers a bit smaller for clarity at this size
TSNE_RECIPE_MARKER = 70
TSNE_ING_MARKER    = 55

# ---------------- Visibility controls ----------------
FIG_DPI        = 200
TICK_SIZE      = 10
AXES_LINEWIDTH = 4.0
TICK_WIDTH     = 3.2
FONT_SIZE      = 18
AX_TITLE_SIZE  = 30
AX_LABEL_SIZE  = 26
LEGEND_SIZE    = 16
LINE_WIDTH     = 4.0
GRID_ALPHA     = 0.22

# Marker sizes (t-SNE) — smaller for clarity
RECIPE_MARKER_SIZE = 90
ING_MARKER_SIZE    = 70

# Global styling
mpl.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": SAVE_DPI,

    # Typography
    "font.size":          FONT_SIZE,
    "axes.titlesize":     AX_TITLE_SIZE,
    "axes.labelsize":     AX_LABEL_SIZE,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "legend.fontsize":    LEGEND_SIZE,

    # Axes/ticks emphasis
    "axes.linewidth":     AXES_LINEWIDTH,
    "axes.edgecolor":     "black",
    "axes.labelcolor":    "black",
    "xtick.color":        "black",
    "ytick.color":        "black",
    "xtick.major.width":  TICK_WIDTH,
    "ytick.major.width":  TICK_WIDTH,

    # Lines
    "lines.linewidth":    LINE_WIDTH,

    # Grid
    "grid.linestyle":     "--",
    "grid.alpha":         GRID_ALPHA,

    # Spines: only left/bottom
    "axes.spines.top":    False,
    "axes.spines.right":  False,

    # Draw order: keep grid below data; spines/ticks above
    "axes.axisbelow":     True,
})

sns.set_theme(style="whitegrid", rc=mpl.rcParams)

# Professional palettes
PALETTE_QUAL = sns.color_palette("tab20")  # categorical (pie / clusters)
PALETTE_SET2 = sns.color_palette("Set2")   # smooth fills

# ---------------- Utils ----------------
def load_attr_from_py(filepath, variable_name):
    spec = importlib.util.spec_from_file_location(variable_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, variable_name)

def normalize_weights(ingredients):
    total = sum(ing['weight'] for ing in ingredients)
    return [{**ing, 'weight': (ing['weight'] / total if total > 0 else 0.0)} for ing in ingredients]

def compute_x_vector(ingredients):
    v = np.zeros(len(SENSORY_ORDER), dtype=float)
    for ing in ingredients:
        w = float(ing['weight']); s = ing['sensory_scores']
        v += w * np.array([float(s[k]) for k in SENSORY_ORDER], dtype=float)
    return v

def _style_axes(ax):
    ax.spines['left'].set_linewidth(AXES_LINEWIDTH)
    ax.spines['bottom'].set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(width=TICK_WIDTH, labelsize=TICK_SIZE, colors="black")
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA, linewidth=max(1.0, AXES_LINEWIDTH - 2))
    ax.margins(x=0.08, y=0.12)  # breathing room

# ---------------- Pie (≥3 uses, no text on wedges) ----------------

def plot_ingredient_usage_pie(raw_recipes, outdir, threshold=3):
    # Count unique recipes per ingredient
    counts = defaultdict(int)
    for r in raw_recipes:
        seen = set()
        for ing in r["ingredients"]:
            name = ing["name"]
            if name not in seen:
                counts[name] += 1
                seen.add(name)

    total_recipes = len(raw_recipes)

    # Filter ingredients by threshold
    items = [(n, c) for n, c in counts.items() if c >= threshold] or list(counts.items())
    # Sort by descending percentage
    items.sort(key=lambda x: x[1], reverse=True)

    labels = [n for n, _ in items]
    sizes  = [100.0 * c / total_recipes for _, c in items]
    counts_only = [c for _, c in items]

    # Strong color palette (avoiding black, neon, washed-out colors)
    from matplotlib.cm import get_cmap
    cmap = get_cmap("tab20")  # strong qualitative palette
    colors = [cmap(i % 20) for i in range(len(labels))]

    # Create figure
    fig, ax = plt.subplots(figsize=(FIGSIZE_W, FIGSIZE_H))
    _style_axes(ax)

    # Pie chart (empty inside, wedges only)
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 0.6, "antialiased": True}
    )
    ax.axis("equal")

    # Build legend-like side panel
    legend_handles = []
    for i, (label, size, count) in enumerate(zip(labels, sizes, counts_only)):
        handle = mpatches.Patch(color=colors[i], label=f"{label}: {size:.1f}% ({count})")
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Ingredients",
        frameon=True,
        fancybox=True
    )

    ax.set_title(f"Ingredient Usage Across {total_recipes} Recipes (≥{threshold} occurrences)")

    plt.tight_layout(pad=2.0)
    out_path = os.path.join(outdir, "ingredient_usage.png")
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")

    return dict(counts)


# ---------------- KDE utilities ----------------
def _kde_vals(vals, x_grid):
    if vals.size > 1 and np.any(vals != vals[0]):
        kde = gaussian_kde(vals, bw_method='scott')
        return kde, kde(x_grid)
    return None, None

def _annotate_top_farthest(ax, vals, names, kde, top_k=3):
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    if sd <= 0 or vals.size == 0:
        return
    z = (vals - mu) / sd
    order = np.argsort(np.abs(z))[::-1][:top_k]
    for i in order:
        x0 = float(vals[i])
        y0 = float(kde([x0])[0]) if kde is not None else ax.get_ylim()[1]*0.05
        ax.axvline(x0, color='crimson', linestyle=':', alpha=0.9, linewidth=LINE_WIDTH, zorder=3)
        ax.annotate(
            names[i],
            xy=(x0, y0), xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', fontsize=FONT_SIZE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=1.2),
            arrowprops=dict(arrowstyle='-', color='crimson', lw=AXES_LINEWIDTH*0.9),
            zorder=4, clip_on=True
        )

# ---------------- Per-sensory Gaussian/KDE ----------------
def plot_gaussian_per_key(raw_recipes, outdir, top_k=3):
    # Dark, professional custom palette (5 keys)
    # sweet, bitter, salty, umami, sour
    custom_palette = {
        "sweet": "#264653",  # deep blue-green
        "bitter": "#e76f51", # brick orange-red
        "salty": "#2a9d8f",  # teal
        "umami": "#e9c46a",  # muted mustard
        "sour":  "#8ab17d",  # soft olive
    }
    color_map = {k: custom_palette[k] for k in SENSORY_ORDER}
    x_grid = np.linspace(0, 100, 1000)

    # How high to lift labels (fraction of final y-top); tweak if needed
    LABEL_LIFT_FRAC = 0.22   # lift labels this much above curve
    Y_HEADROOM_FRAC = 0.35   # extra top headroom so lifted labels don't clip

    # Text halo for readability
    import matplotlib.patheffects as pe
    text_pe = [pe.withStroke(linewidth=2.2, foreground="white")]

    for key in SENSORY_ORDER:
        vals = np.array([float(r['food_sensory_scores'][key]) for r in raw_recipes], dtype=float)
        names = [r['recipe_name'] for r in raw_recipes]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

        fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=350)

        # Heavy/dark axes; tiny labels
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.tick_params(axis='both', which='major',
                       colors='black', width=2.2, length=8, labelsize=7, direction='out')
        ax.grid(True, color='black', alpha=0.12, linestyle='--', linewidth=0.8)

        kde, kde_y = _kde_vals(vals, x_grid)

        if kde_y is not None:
            ax.fill_between(x_grid, kde_y, alpha=0.30, color=color_map[key],
                            label=f'{key.capitalize()} KDE', zorder=1)
            ax.plot(x_grid, kde_y, alpha=0.95, color=color_map[key],
                    linewidth=1.7, zorder=2)
            y_curve_max = float(np.max(kde_y))
        else:
            ax.axvline(mu, color=color_map[key], alpha=0.95, linewidth=1.8,
                       label=f'{key.capitalize()} (degenerate)', zorder=2)
            y_curve_max = 0.0

        if sd > 0:
            gauss_y = norm.pdf(x_grid, mu, sd)
            ax.plot(x_grid, gauss_y, linestyle='-', color='black',
                    linewidth=2.1, label=f'Gaussian (μ={mu:.1f}, σ={sd:.1f})', zorder=3)
            y_curve_max = max(y_curve_max, float(np.max(gauss_y)))

        # --- Raise top-k farthest labels well above to avoid overlap ---
        if vals.size and sd > 0:
            z = (vals - mu) / sd
            order = np.argsort(np.abs(z))[::-1][:top_k]
            # stagger a tiny bit horizontally so labels don't stack perfectly
            x_jitter = np.linspace(-1.5, 1.5, num=len(order)) if len(order) > 1 else [0.0]
            for rank, (i, xoff) in enumerate(zip(order, x_jitter)):
                x0 = float(vals[i])
                y0 = float(kde([x0])[0]) if kde is not None else 0.0
                y_top = y_curve_max * (1.0 + Y_HEADROOM_FRAC)
                y_label = y0 + y_curve_max * LABEL_LIFT_FRAC
                # Clamp label inside top headroom
                y_label = min(y_label, y_top * 0.98)

                ax.annotate(
                    names[i],
                    xy=(x0, y0),
                    xytext=(x0 + xoff, y_label),
                    textcoords='data',
                    ha='center', va='bottom',
                    fontsize=7, color='black', fontweight='bold',
                    path_effects=text_pe,
                    arrowprops=dict(arrowstyle='-', color='black', lw=1.6, alpha=0.9),
                    zorder=4, annotation_clip=False
                )
                # guide line at the value
                ax.axvline(x0, color='black', linestyle=':', alpha=0.7, linewidth=1.6, zorder=2.5)

        # NO horizontal padding; sit on baseline
        ax.set_xlim(0, 100)
        # Keep baseline at 0; add only top headroom for lifted labels
        y_min, y_max_now = ax.get_ylim()
        y_top_needed = max(y_curve_max * (1.0 + Y_HEADROOM_FRAC), y_max_now)
        ax.set_ylim(0, y_top_needed)

        # Small labels/titles
        ax.set_title(f"{key.capitalize()} Distribution + Gaussian Fit", fontsize=8, weight='bold', color='black')
        ax.set_xlabel('Score (0–100)', fontsize=7, color='black')
        ax.set_ylabel('Density', fontsize=7, color='black')

        # Compact legend
        leg = ax.legend(
            frameon=True, loc='upper right', fancybox=False, fontsize=6,
            handlelength=1.2, handletextpad=0.35, borderpad=0.25, labelspacing=0.25
        )
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(0.8)

        plt.tight_layout(pad=0.5)
        out_path = os.path.join(outdir, f"gaussian_{key}.png")
        plt.savefig(out_path, dpi=350, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"[✓] Saved: {out_path}")


def plot_combined_kde_with_gaussian(raw_recipes, outdir):
    # Dark, rich, professional colors from ColorBrewer (manually hex-coded)
    custom_palette = [
        "#4e79a7",  # deep blue
        "#f28e2b",  # burnt orange
        "#e15759",  # brick red
        "#76b7b2",  # teal
        "#59a14f",  # forest green
        "#edc949",  # mustard
        "#af7aa1",  # plum
        "#ff9da7",  # muted pink
        "#9c755f",  # coffee brown
        "#bab0ab"   # stone gray
    ]
    base_palette = custom_palette[:len(SENSORY_ORDER)]
    color_map = dict(zip(SENSORY_ORDER, base_palette))

    x_grid = np.linspace(0, 100, 1000)

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

    # Axes styling
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)
    ax.tick_params(axis='both', which='major', colors='black', width=1.2, labelsize=8)
    ax.grid(True, color='black', alpha=0.15, linestyle='--', linewidth=0.8)

    pooled = []

    # KDEs per attribute (excluding sour)
    for key in SENSORY_ORDER:
        if key == 'sour':
            continue
        vals = np.array([float(r['food_sensory_scores'][key]) for r in raw_recipes], dtype=float)
        pooled.extend(vals.tolist())
        kde, kde_y = _kde_vals(vals, x_grid)
        if kde_y is not None:
            ax.fill_between(x_grid, kde_y, alpha=0.3, color=color_map[key], label=f'{key.capitalize()} KDE')
            ax.plot(x_grid, kde_y, alpha=0.95, color=color_map[key], linewidth=1.4)

    pooled = np.array(pooled, dtype=float)
    mu = float(np.mean(pooled))
    sd = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0

    # Gaussian fit
    if sd > 0:
        gauss_y = norm.pdf(x_grid, mu, sd)
        ax.plot(x_grid, gauss_y, linestyle='-', color='black', linewidth=1.8,
                label=f'Pooled Gaussian (μ={mu:.1f}, σ={sd:.1f})')
    else:
        ax.axvline(mu, color='black', linestyle='--', linewidth=1.6,
                   label=f'Pooled Gaussian (μ={mu:.1f})')

    # X-axis: 0 to slightly beyond 100
    ax.set_xlim(0, 110)

    # Keep density baseline on X-axis & add top breathing room
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max * 1.3)

    ax.set_title('Sensory distribution across recipes', fontsize=9, weight='bold', color='black')
    ax.set_xlabel('Score (0–100)', fontsize=8, color='black')
    ax.set_ylabel('Density', fontsize=8, color='black')

    # Smaller, compact legend
    leg = ax.legend(
        frameon=True,
        loc='upper right',
        fancybox=False,
        fontsize=6,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.3
    )
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.8)
    out_path = os.path.join(outdir, "combined_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")

# === tsne_bestk_pipeline.py ===
# Saves everything under results/t-sne_new

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal, mannwhitneyu

# You must have these in your environment:
# SENSORY_ORDER: e.g. ["sweet", "sour", "salty", "bitter", "umami"]
# SEED: e.g. 42
# normalize_weights(ingredients) -> list of (name, weight, sensory_scores)
# compute_x_vector(normalized_ingredients) -> feature vector for embedding

OUTDIR = "results/t-sne_new"
os.makedirs(OUTDIR, exist_ok=True)

# --- helpers ---------------------------------------------------------------

def _safe_perplexity(n_samples, perplexity):
    return min(perplexity, max(2, (n_samples - 1) // 3))

def _pick_k_silhouette(X, k_min=2, k_max=12, plot_path=None, seed=42):
    k_max_eff = max(k_min, min(k_max, X.shape[0] - 1))
    K_range = list(range(k_min, k_max_eff + 1))

    scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels_k = km.fit_predict(X)
        try:
            s = silhouette_score(X, labels_k, metric='euclidean')
        except Exception:
            s = np.nan
        scores.append(s)

    # choose best valid k
    valid = [(k, s) for k, s in zip(K_range, scores) if s == s]  # filter NaNs
    best_k = valid[np.argmax([s for _, s in valid])][0] if valid else K_range[0]

    if plot_path:
        plt.figure(figsize=(7,5))
        if valid:
            ks, ss = zip(*valid)
            plt.plot(ks, ss, marker="o")
            # highlight best
            best_s = ss[ks.index(best_k)]
            plt.scatter([best_k], [best_s], s=160, facecolors="none", edgecolors="k", linewidths=2)
            plt.annotate(f"k={best_k}\n{s:.3f}" if valid else f"k={best_k}",
                         (best_k, best_s), textcoords="offset points", xytext=(8,8))
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette (KMeans)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
    return best_k

def _render_tsne(Z, labels, recipe_names, ingredient_names, n_rec, out_path):
    cmap = plt.get_cmap('tab10')
    Z_rec = Z[:n_rec]
    Z_ing = Z[n_rec:] if Z.shape[0] > n_rec else np.empty((0, 2))

    plt.figure(figsize=(8,6))
    # ingredients
    if Z_ing.size:
        colors_ing = [cmap(int(l)%10) for l in labels[n_rec:]]
        plt.scatter(Z_ing[:,0], Z_ing[:,1], s=20, c=colors_ing, marker='o', edgecolors='black', linewidths=0.5, alpha=0.9, label="Ingredients")
    # recipes
    colors_rec = [cmap(int(l)%10) for l in labels[:n_rec]]
    plt.scatter(Z_rec[:,0], Z_rec[:,1], s=35, c=colors_rec, marker='s', edgecolors='black', linewidths=0.6, alpha=0.98, label="Recipes")

    # legend by cluster id
    cluster_ids = sorted(set(int(x) for x in labels))
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='o', color='black', label=f'Cluster {cid}',
                      markerfacecolor=cmap(cid%10), linestyle='', markersize=6)
               for cid in cluster_ids]
    handles += [
        Line2D([0],[0], marker='s', color='black', linestyle='', label='Recipes (■)', markerfacecolor='none', markersize=6),
        Line2D([0],[0], marker='o', color='black', linestyle='', label='Ingredients (●)', markerfacecolor='none', markersize=6)
    ]
    plt.legend(handles=handles, loc="best", fontsize=8, frameon=True)
    plt.title("t-SNE (best k)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def _save_cluster_members_json(labels, recipe_names, ingredient_names, n_rec, outdir):
    cluster_map = defaultdict(lambda: {"recipes": [], "ingredients": []})
    for idx, cl in enumerate(labels):
        cid = int(cl)
        if idx < n_rec:
            cluster_map[cid]["recipes"].append(recipe_names[idx])
        else:
            cluster_map[cid]["ingredients"].append(ingredient_names[idx - n_rec])
    best_k = len(set(int(x) for x in labels))
    json_path   = os.path.join(outdir, f"tsne_cluster_members_bestk_{best_k}.json")
    latest_path = os.path.join(outdir, "cluster_members_latest_tsne.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, ensure_ascii=False, indent=2)

def _extract_recipe_sensory(raw_recipes, sensory_keys):
    """
    Compute weighted-average sensory scores for each recipe.
    raw_recipes[i]['ingredients'][j]['sensory_scores'] must be a dict with sensory_keys.
    """
    import numpy as np
    names = [r['recipe_name'] for r in raw_recipes]
    S = np.zeros((len(raw_recipes), len(sensory_keys)), dtype=float)

    for i, r in enumerate(raw_recipes):
        total_w = 0.0
        accum = np.zeros(len(sensory_keys), dtype=float)
        for ing in r['ingredients']:
            w = ing.get('weight', 0.0)
            sens = ing.get('sensory_scores', {}) or {}
            if not isinstance(sens, dict):
                sens = {}
            vec = np.array([float(sens.get(k, 0.0)) for k in sensory_keys], dtype=float)
            accum += w * vec
            total_w += w
        if total_w > 0:
            S[i, :] = accum / total_w
        else:
            S[i, :] = 0.0
    return names, S

def save_cluster_pvalues_json(labels_rec, S_rec, sensation_names=None,
                              out_path="results/t-sne_new/p_values.json"):
    """
    Saves JSON with p-values for each cluster:
      - Mann–Whitney U (non-parametric median test)
      - Welch's t-test (mean test)
      - KS-test (distribution difference)
    """
    import os, json, numpy as np
    from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels_rec = np.asarray(labels_rec).astype(int)
    clusters = sorted(set(labels_rec.tolist()))
    results = {}

    if sensation_names is None:
        sensation_names = [f"sens_{i}" for i in range(S_rec.shape[1])]

    for cid in clusters:
        results[cid] = {}
        mask_in = (labels_rec == cid)
        mask_out = ~mask_in

        for j, sens in enumerate(sensation_names):
            x_in = S_rec[mask_in, j]
            x_out = S_rec[mask_out, j]

            entry = {}

            try:
                _, entry["mannwhitney_p"] = mannwhitneyu(x_in, x_out, alternative="two-sided")
            except Exception:
                entry["mannwhitney_p"] = np.nan

            try:
                _, entry["ttest_p"] = ttest_ind(x_in, x_out, equal_var=False, nan_policy="omit")
            except Exception:
                entry["ttest_p"] = np.nan

            try:
                _, entry["ks_p"] = ks_2samp(x_in, x_out)
            except Exception:
                entry["ks_p"] = np.nan


            results[cid][sens] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# --- main entry ------------------------------------------------------------

def run_tsne_bestk_and_stats(raw_recipes,
                             perplexity=5,
                             k_min=2, k_max=12,
                             outdir=OUTDIR):
    """
    Produces:
      - {outdir}/silhouette.png
      - {outdir}/tsne_bestk.png
      - {outdir}/tsne_cluster_members_bestk_<k>.json (+ latest alias)
      - {outdir}/p_values.csv
    """
    os.makedirs(outdir, exist_ok=True)

    # --- build embedding dataset (recipes + unique ingredients’ sensory vectors) ---
    R = np.array([compute_x_vector(normalize_weights(r['ingredients'])) for r in raw_recipes])
    recipe_names = [r['recipe_name'] for r in raw_recipes]

    # unique ingredient sensory vectors by name (if present)
    ing_map = {}
    for r in raw_recipes:
        for ing in r['ingredients']:
            name = ing['name']
            if name not in ing_map:
                s = ing.get('sensory_scores', {}) or {}
                ing_map[name] = np.array([float(s.get(k, 0.0)) for k in SENSORY_ORDER], dtype=float)
    ingredient_names = list(ing_map.keys())
    I = np.vstack([ing_map[n] for n in ingredient_names]) if ingredient_names else np.zeros((0, len(SENSORY_ORDER)))
    X = np.vstack([R, I]) if I.size else R

    # --- t-SNE ---
    n_samples = X.shape[0]
    tsne = TSNE(
        n_components=2,
        perplexity=_safe_perplexity(n_samples, perplexity),
        random_state=SEED,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
    )
    Z = tsne.fit_transform(X)

    # small jitter to stabilize clustering ties
    rng = np.random.RandomState(SEED)
    Z_ = Z + 1e-6 * rng.normal(size=Z.shape)

    # --- choose k by silhouette ---
    sil_path = os.path.join(outdir, "silhouette.png")
    best_k = _pick_k_silhouette(Z_, k_min=k_min, k_max=k_max, plot_path=sil_path, seed=SEED)

    # --- fit best-k clustering ---
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    labels = km.fit_predict(Z_)

    # --- render ONE t-SNE plot for best k ---
    tsne_path = os.path.join(outdir, "tsne_bestk.png")
    _render_tsne(Z, labels, recipe_names, ingredient_names, n_rec=len(recipe_names), out_path=tsne_path)

    # --- save cluster membership JSON ---
    _save_cluster_members_json(labels, recipe_names, ingredient_names, n_rec=len(recipe_names), outdir=outdir)

    # --- p-values for five sensations per cluster (and global) ---
    # Map any typos to standard five: sweet, sour, salty, bitter, umami
    # We’ll read them from SENSORY_ORDER but ensure these five exist; if not, we’ll use what’s present.
    FIVE = ["sweet", "sour", "salty", "bitter", "umami"]
    sensory_keys = [k for k in FIVE if k in SENSORY_ORDER] or list(SENSORY_ORDER[:5])

    rec_names_check, S_rec = _extract_recipe_sensory(raw_recipes, sensory_keys)
    assert rec_names_check == recipe_names, "Recipe order mismatch while extracting sensory"

    # Take labels for recipes only (labels include ingredients too)
    labels_rec = labels[:len(recipe_names)]
    p_csv = os.path.join(outdir, "p_values.csv")
    save_cluster_pvalues_json(labels_rec, S_rec, sensation_names=["sweet", "sour", "salty", "bitter", "umami"],
                              out_path="results/t-sne_new/p_values.json")

    print(f"[✓] Best k = {best_k}")
    print(f"[✓] Saved silhouette: {sil_path}")
    print(f"[✓] Saved t-SNE: {tsne_path}")
    print(f"[✓] Saved p-values: {p_csv}")
    print(f"[✓] Output dir: {os.path.abspath(outdir)}")

def pca_clusters_minimal(
    raw_recipes,
    n_components=2,
    standardize=False,
    outdir="results/plot_pca",
    k_min=2,
    k_max=12,
    seed=42
):
    """
    Minimal PCA + KMeans pipeline that:
      1) Plots silhouette vs k (pca_silhouette.png)
      2) Picks best k (highest silhouette), then plots 2D PCA with legend outside (pca_bestk_scatter.png)
      3) Saves JSON mapping each cluster -> {recipes, ingredients} (clusters_bestk.json)
      4) Adds a bar plot of PC1 loadings/weights by SENSORY_ORDER (pc1_loadings.png)
      5) Annotates axes with % variance for PC1 and PC2

    Assumes the following are defined elsewhere in your codebase:
      - SENSORY_ORDER: List[str]  (feature names / order)
      - compute_x_vector(weights_dict) -> np.ndarray (len = len(SENSORY_ORDER))
      - normalize_weights(ingredients_list) -> dict[name -> float]
    """
    import os, json
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    # ---------- Build feature matrices ----------
    # Recipes matrix (R): one row per recipe, columns in SENSORY_ORDER
    R = np.array([
        compute_x_vector(normalize_weights(r["ingredients"]))
        for r in raw_recipes
    ], dtype=float)
    recipe_names = [r["recipe_name"] for r in raw_recipes]

    # Ingredients matrix (I): one row per unique ingredient, columns in SENSORY_ORDER
    ing_map = {}
    for r in raw_recipes:
        for ing in r["ingredients"]:
            name = ing["name"]
            if name not in ing_map:
                s = ing["sensory_scores"]
                ing_map[name] = np.array([float(s[k]) for k in SENSORY_ORDER], dtype=float)
    ingredient_names = list(ing_map.keys())
    I = np.vstack([ing_map[n] for n in ingredient_names]) if ingredient_names else np.zeros((0, len(SENSORY_ORDER)))

    # Combined matrix X (stack recipes over ingredients)
    X = np.vstack([R, I]) if I.size else R
    n_rec = R.shape[0]

    # ---------- Optional standardization ----------
    if standardize:
        from sklearn.preprocessing import StandardScaler
        X0 = StandardScaler().fit_transform(X)
    else:
        X0 = X

    # ---------- PCA ----------
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False, random_state=seed)
    Z = pca.fit_transform(X0)  # shape: (n_samples, n_components)
    evr = pca.explained_variance_ratio_
    var_pc1 = float(evr[0]) if len(evr) > 0 else 0.0
    var_pc2 = float(evr[1]) if len(evr) > 1 else 0.0

    # ---------- PC1 loadings bar plot ----------
    # pca.components_[0] has shape (n_features,)
    os.makedirs(outdir or ".", exist_ok=True)
    pc1 = pca.components_[0]
    order = np.argsort(-np.abs(pc1))  # sort by absolute weight, descending
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(pc1)), pc1[order])
    plt.xticks(np.arange(len(pc1)), [SENSORY_ORDER[i] for i in order], rotation=60, ha="right")
    plt.ylabel("PC1 Loading (weight)")
    plt.title("PC1 Loadings by Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pc1_loadings.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Tiny jitter to stabilize KMeans ties
    rng = np.random.RandomState(seed)
    Z_jit = Z + 1e-6 * rng.normal(size=Z.shape)

    # ---------- Pick k by silhouette & plot curve ----------
    # ---------- Pick k by silhouette & plot curve (robust) ----------
    n_samples = Z_jit.shape[0]

    # Silhouette needs at least 2 clusters and at least 3 samples overall.
    if n_samples < 3:
        # Fall back: no silhouette possible
        K_range = [max(1, min(k_max, n_samples))]
        scores = [np.nan]
        best_k = K_range[0]
    else:
        # Enforce k >= 2, and k <= n_samples - 1
        k_min_eff = max(2, k_min)
        k_max_eff = max(k_min_eff, min(k_max, n_samples - 1))
        K_range = list(range(k_min_eff, k_max_eff + 1))

        scores = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels_k = km.fit_predict(Z_jit)

            # Guard against degenerate labelings (e.g., empty cluster or effectively <2 unique labels)
            unique_labels = np.unique(labels_k)
            if unique_labels.size < 2 or unique_labels.size > n_samples - 1:
                scores.append(np.nan)
                continue

            # Also guard against singleton clusters which make per-sample silhouette undefined
            # (sklearn will raise in such cases in some versions)
            counts = np.bincount(labels_k)
            if counts.min() <= 1:
                scores.append(np.nan)
                continue

            try:
                s_val = silhouette_score(Z_jit, labels_k, metric="euclidean")
            except Exception:
                s_val = np.nan
            scores.append(s_val)

        # Pick best k among valid scores; if all nan, fall back to smallest k in range
        if np.all(np.isnan(scores)):
            best_k = K_range[0]
            best_s = np.nan
        else:
            best_idx = int(np.nanargmax(scores))
            best_k = K_range[best_idx]
            best_s = float(scores[best_idx])

    # Silhouette plot
    plt.figure(figsize=(8, 5))
    y_vals = [s if (s is not None and not np.isnan(s)) else np.nan for s in scores]
    plt.plot(K_range, y_vals, marker="o")

    # Highlight best point, if any valid score exists
    if not np.all(np.isnan(y_vals)):
        plt.scatter([best_k], [best_s], s=160, facecolors="none", edgecolors="k", linewidths=2)
        # <-- bug fix: use best_s, not the stale `s`
        plt.annotate(f"k = {best_k}\n{best_s:.3f}", (best_k, best_s),
                     textcoords="offset points", xytext=(10, -25))

    plt.xlabel("Clusters (k)")
    plt.ylabel("Silhouette score (higher is better)")
    plt.title("Silhouette scores for KMeans (PCA space)")
    plt.grid(True, linestyle="--", alpha=0.4)
    if len(K_range) > 0:
        plt.xticks(K_range)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_silhouette.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- Final clustering at best_k ----------
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
    labels = km.fit_predict(Z_jit)

    # ---------- PCA scatter for best_k (legend OUTSIDE axes) ----------
    # Split back to recipes vs ingredients for markers
    Z_rec = Z[:, :2][:n_rec]
    Z_ing = Z[:, :2][n_rec:] if Z.shape[0] > n_rec else np.empty((0, 2))
    labels_rec = labels[:n_rec]
    labels_ing = labels[n_rec:] if Z_ing.size else np.array([], dtype=int)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    if Z_ing.size:
        ax.scatter(Z_ing[:, 0], Z_ing[:, 1],
                   s=24, alpha=0.9, edgecolors="black", linewidths=0.6,
                   marker="o",
                   c=[cmap(int(c) % 10) for c in labels_ing],
                   label="Ingredients")

    ax.scatter(Z_rec[:, 0], Z_rec[:, 1],
               s=32, alpha=0.95, edgecolors="black", linewidths=0.6,
               marker="s",
               c=[cmap(int(c) % 10) for c in labels_rec],
               label="Recipes")

    # Build legend entries: one per cluster + markers guide
    import matplotlib.lines as mlines
    handles = []
    cluster_ids = sorted(set(int(x) for x in labels))
    for cl in cluster_ids:
        handles.append(
            mlines.Line2D([0],[0], color='black', marker='o',
                          markerfacecolor=cmap(int(cl) % 10), linestyle='',
                          markersize=6, label=f'Cluster {cl}')
        )
    handles.append(mlines.Line2D([0],[0], color='black', marker='s', linestyle='',
                                 markersize=6, label='Recipes (■)', markerfacecolor='none'))
    handles.append(mlines.Line2D([0],[0], color='black', marker='o', linestyle='',
                                 markersize=6, label='Ingredients (●)', markerfacecolor='none'))

    # Legend outside (right side)
    leg = ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                    frameon=True, fontsize=9)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)

    ax.grid(True, color='black', alpha=0.25, linestyle='--', linewidth=1.0)
    ax.set_title(f"PCA (k={best_k})")
    ax.set_xlabel(f"PC1 ({var_pc1*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_pc2*100:.1f}% var)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_bestk_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- JSON: members by cluster ----------
    cluster_map = {}
    for idx, cl in enumerate(labels):
        cl = int(cl)
        if cl not in cluster_map:
            cluster_map[cl] = {"recipes": [], "ingredients": []}
        if idx < n_rec:
            cluster_map[cl]["recipes"].append(recipe_names[idx])
        else:
            cluster_map[cl]["ingredients"].append(ingredient_names[idx - n_rec])

    def _to_py(obj):
        """JSON-safe conversion for numpy types."""
        import numpy as np
        if isinstance(obj, dict):
            return { (int(k) if isinstance(k, (np.integer,)) else k): _to_py(v) for k, v in obj.items() }
        if isinstance(obj, (list, tuple)):
            return [_to_py(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    json_path = os.path.join(outdir, "clusters_bestk.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_py(cluster_map), f, ensure_ascii=False, indent=2)

    # Return a compact summary in case you want to use it programmatically
    return {
        "best_k": int(best_k),
        "explained_variance_ratio": {
            "pc1": var_pc1,
            "pc2": var_pc2
        },
        "paths": {
            "silhouette": os.path.join(outdir, "pca_silhouette.png"),
            "pca_bestk_scatter": os.path.join(outdir, "pca_bestk_scatter.png"),
            "pc1_loadings": os.path.join(outdir, "pc1_loadings.png"),
            "clusters_json": json_path
        }
    }

def main():
    raw_recipes = load_attr_from_py(RAW_FILE, 'raw_recipes')

    plot_ingredient_usage_pie(raw_recipes, PLOTS_DIR, threshold=3)
    plot_gaussian_per_key(raw_recipes, PLOTS_DIR, top_k=3)
    plot_combined_kde_with_gaussian(raw_recipes, PLOTS_DIR)
    pca_clusters_minimal(raw_recipes, n_components=2, standardize=False)
    print("[✓] All publication-ready PNGs generated (<= 5456x4096, 4:3).")
    run_tsne_bestk_and_stats(raw_recipes)
if __name__ == "__main__":
    main()
