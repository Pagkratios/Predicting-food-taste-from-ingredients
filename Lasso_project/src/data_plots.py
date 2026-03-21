#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-ready data analysis plots for the 29-recipe dataset (PNG only).

Outputs (ALL at <= 5456x4096 px, 4:3 aspect):
  1) ingredient_usage.png -- pie chart, >=3-use ingredients, legend side panel
  2) gaussian_<key>.png for each of [sweet, bitter, salty, umami, sour]:
     per-key KDE (SciPy gaussian_kde, Scott's rule) + Gaussian fit; top-3 farthest annotated
  3) combined_distribution.png -- pooled KDE without bitter + single Gaussian fit
  4) combined_distribution_with_bitter.png -- pooled KDE with all sensory keys + single Gaussian fit
  5) PCA cluster analysis: silhouette, scatter, PC1 loadings, cluster JSON
  6) t-SNE cluster analysis: silhouette, scatter, cluster JSON, p-values JSON

Usage examples:
  python3 Lasso_project/src/data_plots.py
  python3 Lasso_project/src/data_plots.py --only gaussians combined
  python3 Lasso_project/src/data_plots.py --only pca tsne
"""

import argparse
import importlib.util
import json
import os
import random
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.stats import (
    gaussian_kde, ks_2samp, mannwhitneyu, norm, ttest_ind,
)
from env_config import get_raw_recipes_path

# ---------------- Paths & constants ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RAW_FILE     = get_raw_recipes_path()

# New styled outputs under Lasso_project/results/
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots_data")
TSNE_DIR     = os.path.join(PROJECT_ROOT, "results", "t-sne")
PCA_DIR      = os.path.join(PROJECT_ROOT, "results", "pca")

for _d in (PLOTS_DIR, TSNE_DIR, PCA_DIR):
    os.makedirs(_d, exist_ok=True)

SENSORY_ORDER = ["sweet", "bitter", "salty", "umami", "sour"]
PLOT_GROUP_CHOICES = ["all", "ingredient_usage", "gaussians", "combined", "pca", "tsne", "clusters"]

# ---------------- Determinism ----------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

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

# ---------------- KDE utilities ----------------
def _kde_vals(vals, x_grid):
    if vals.size > 1 and np.any(vals != vals[0]):
        kde = gaussian_kde(vals, bw_method='scott')
        return kde, kde(x_grid)
    return None, None


def _get_ranked_outliers(vals, names, top_k=3):
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    if sd <= 0 or vals.size == 0:
        return []

    z = (vals - mu) / sd
    order = np.argsort(np.abs(z))[::-1][:top_k]
    return [
        {
            "rank": rank,
            "name": names[i],
            "value": float(vals[i]),
        }
        for rank, i in enumerate(order, start=1)
    ]


def _draw_outlier_guides(ax, ranked_outliers, guide_linewidth=0.8):
    for item in ranked_outliers:
        ax.axvline(
            item["value"],
            color='black',
            linestyle=':',
            alpha=0.7,
            linewidth=guide_linewidth,
            zorder=2.5,
        )


def _save_gaussian_dotted_line_report(assignments_by_key, outdir):
    out_path = os.path.join(outdir, "gaussian_dotted_line_recipes.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Gaussian dotted line recipe guide\n")
        f.write("Order is left to right in each plot.\n\n")
        for key in SENSORY_ORDER:
            f.write(f"{key.capitalize()}\n")
            items = assignments_by_key.get(key, [])
            if not items:
                f.write("  No dotted line assignments.\n\n")
                continue
            for line_pos, item in enumerate(sorted(items, key=lambda x: (x["value"], x["rank"])), start=1):
                f.write(f"  {line_pos}. {item['name']}\n")
            f.write("\n")
    print(f"[+] Saved: {out_path}")

def _safe_perplexity(n_samples, perplexity):
    return min(perplexity, max(2, (n_samples - 1) // 3))

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
    """Compute weighted-average sensory scores for each recipe."""
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

def save_cluster_pvalues_json(labels_rec, S_rec, sensation_names=None, out_path=None):
    """
    Saves JSON with p-values for each cluster:
      - Mann-Whitney U, Welch's t-test, KS-test
    """
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


# ════════════════════════════════════════════════════════════════════════
# NEW (v2) publication-quality plotting functions -- unified plot_config
# ════════════════════════════════════════════════════════════════════════

def plot_ingredient_usage_pie_v2(raw_recipes, outdir, threshold=3):
    """Ranked ingredient usage chart with unified styling."""
    from plot_config import (
        setup_figure, save_figure_fixed, style_axes,
        FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_ANNOTATION,
    )

    counts = defaultdict(int)
    for r in raw_recipes:
        seen = set()
        for ing in r["ingredients"]:
            name = ing["name"]
            if name not in seen:
                counts[name] += 1
                seen.add(name)

    total_recipes = len(raw_recipes)
    items = [(n, c) for n, c in counts.items() if c >= threshold] or list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    labels = [n for n, _ in items]
    counts_only = [c for _, c in items]
    pct = [100.0 * c / total_recipes for c in counts_only]

    cmap = get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(labels))]

    fig, ax = setup_figure(size=(7.0, 5.8))
    style_axes(ax, grid=True)

    y = np.arange(len(labels))
    bars = ax.barh(y, pct, color=colors, edgecolor="#333333", linewidth=0.5, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(pct) * 1.18)
    ax.set_xlabel("Recipes Containing Ingredient (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        f"Ingredient Usage Across Recipes (\u2265{threshold} uses)",
        fontsize=FONT_SIZE_TITLE,
        weight="bold",
    )

    for bar, count, pct_val in zip(bars, counts_only, pct):
        ax.text(
            bar.get_width() + 0.45,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count}/{total_recipes}",
            va="center",
            ha="left",
            fontsize=FONT_SIZE_ANNOTATION,
            color="#333333",
        )

    out_path = os.path.join(outdir, "ingredient_usage.png")
    save_figure_fixed(fig, out_path)
    return dict(counts)


def plot_gaussian_per_key_v2(raw_recipes, outdir, top_k=3):
    """Per-sensory KDE + Gaussian fit with unified styling."""
    from plot_config import (
        setup_figure, save_figure_fixed, style_axes, get_sensory_color,
        SENSORY_ORDER as SO, FONT_SIZE_ANNOTATION, FONT_SIZE_TITLE, FONT_SIZE_LABEL,
    )

    x_grid = np.linspace(0, 100, 1000)
    assignments_by_key = {}

    for key in SO:
        vals = np.array([float(r['food_sensory_scores'][key]) for r in raw_recipes], dtype=float)
        names = [r['recipe_name'] for r in raw_recipes]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        color = get_sensory_color(key)

        fig, ax = setup_figure(size="single_sq")
        style_axes(ax)

        kde, kde_y = _kde_vals(vals, x_grid)
        if kde_y is not None:
            ax.fill_between(x_grid, kde_y, alpha=0.30, color=color,
                            label=f'{key.capitalize()} KDE', zorder=1)
            ax.plot(x_grid, kde_y, alpha=0.95, color=color, linewidth=1.5, zorder=2)
            y_curve_max = float(np.max(kde_y))
        else:
            ax.axvline(mu, color=color, alpha=0.95, linewidth=1.5,
                       label=f'{key.capitalize()} (degenerate)', zorder=2)
            y_curve_max = 0.0

        if sd > 0:
            gauss_y = norm.pdf(x_grid, mu, sd)
            ax.plot(x_grid, gauss_y, linestyle='-', color='black', linewidth=1.8,
                    label=f'Gaussian (\u03bc={mu:.1f}, \u03c3={sd:.1f})', zorder=3)
            y_curve_max = max(y_curve_max, float(np.max(gauss_y)))

        if vals.size and sd > 0:
            ranked_outliers = _get_ranked_outliers(vals, names, top_k=top_k)
            _draw_outlier_guides(ax, ranked_outliers, guide_linewidth=0.8)
            assignments_by_key[key] = ranked_outliers

        ax.set_xlim(0, 100)
        y_top = max(y_curve_max * 1.35, ax.get_ylim()[1])
        ax.set_ylim(0, y_top)

        ax.set_title(f"{key.capitalize()} Score Distribution",
                     fontsize=FONT_SIZE_TITLE, weight='bold')
        ax.set_xlabel('Score (0\u2013100)', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Density', fontsize=FONT_SIZE_LABEL)

        leg = ax.legend(frameon=True, loc='upper right', fancybox=False,
                        fontsize=FONT_SIZE_ANNOTATION, handlelength=1.2)
        leg.get_frame().set_edgecolor('#333333')
        leg.get_frame().set_linewidth(0.6)

        out_path = os.path.join(outdir, f"gaussian_{key}.png")
        save_figure_fixed(fig, out_path)

    _save_gaussian_dotted_line_report(assignments_by_key, outdir)


def plot_combined_kde_v2(raw_recipes, outdir, exclude_keys=None, out_name="combined_distribution.png", title=None):
    """Combined KDE across selected sensory keys, plus pooled Gaussian."""
    from plot_config import (
        setup_figure, save_figure_fixed, style_axes, get_sensory_color,
        SENSORY_ORDER as SO, FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_ANNOTATION,
    )

    exclude_keys = set(exclude_keys or [])
    x_grid = np.linspace(0, 100, 1000)
    fig, ax = setup_figure(size="single_sq")
    style_axes(ax)

    pooled = []
    for key in SO:
        if key in exclude_keys:
            continue
        vals = np.array([float(r['food_sensory_scores'][key]) for r in raw_recipes], dtype=float)
        pooled.extend(vals.tolist())
        kde, kde_y = _kde_vals(vals, x_grid)
        if kde_y is not None:
            color = get_sensory_color(key)
            ax.fill_between(x_grid, kde_y, alpha=0.3, color=color, label=f'{key.capitalize()} KDE')
            ax.plot(x_grid, kde_y, alpha=0.95, color=color, linewidth=1.2)

    pooled = np.array(pooled, dtype=float)
    mu = float(np.mean(pooled))
    sd = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0

    if sd > 0:
        gauss_y = norm.pdf(x_grid, mu, sd)
        ax.plot(x_grid, gauss_y, linestyle='-', color='black', linewidth=1.5,
                label=f'Pooled Gaussian (\u03bc={mu:.1f}, \u03c3={sd:.1f})')

    ax.set_xlim(0, 100)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max * 1.3)

    ax.set_title(title or 'Combined Score Distribution', fontsize=FONT_SIZE_TITLE, weight='bold')
    ax.set_xlabel('Score (0\u2013100)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Density', fontsize=FONT_SIZE_LABEL)

    leg = ax.legend(frameon=True, loc='upper right', fancybox=False,
                    fontsize=FONT_SIZE_ANNOTATION, handlelength=1.5)
    leg.get_frame().set_edgecolor('#333333')
    leg.get_frame().set_linewidth(0.6)

    out_path = os.path.join(outdir, out_name)
    save_figure_fixed(fig, out_path)


def _pick_k_silhouette_v2(X, k_min=2, k_max=12, plot_path=None, seed=42):
    """Silhouette-based k selection with unified styling."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from plot_config import setup_figure, save_figure_fixed, style_axes, FONT_SIZE_TITLE, FONT_SIZE_LABEL

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

    valid = [(k, s) for k, s in zip(K_range, scores) if s == s]
    best_k = valid[np.argmax([s for _, s in valid])][0] if valid else K_range[0]

    if plot_path:
        fig, ax = setup_figure(size="single")
        style_axes(ax)
        if valid:
            ks, ss = zip(*valid)
            ax.plot(ks, ss, marker="o", linewidth=1.5, markersize=5)
            best_s = ss[ks.index(best_k)]
            ax.scatter([best_k], [best_s], s=120, facecolors="none",
                       edgecolors="k", linewidths=1.5, zorder=5)
            ax.annotate(f"k={best_k}\n{best_s:.3f}", (best_k, best_s),
                        textcoords="offset points", xytext=(10, -22), fontsize=8)
        ax.set_xlabel("Clusters (k)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Silhouette score", fontsize=FONT_SIZE_LABEL)
        ax.set_title("KMeans Silhouette (t-SNE space)", fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(K_range)
        save_figure_fixed(fig, plot_path)

    return best_k


def _render_tsne_v2(Z, labels, recipe_names, ingredient_names, n_rec, out_path):
    """t-SNE scatter with unified styling."""
    from plot_config import setup_figure, save_figure_fixed, style_axes, FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_LEGEND

    cmap = plt.get_cmap('tab10')
    Z_rec = Z[:n_rec]
    Z_ing = Z[n_rec:] if Z.shape[0] > n_rec else np.empty((0, 2))

    fig, ax = setup_figure(size="double")
    style_axes(ax)

    if Z_ing.size:
        colors_ing = [cmap(int(l) % 10) for l in labels[n_rec:]]
        ax.scatter(Z_ing[:, 0], Z_ing[:, 1], s=20, c=colors_ing, marker='o',
                   edgecolors='black', linewidths=0.4, alpha=0.9)

    colors_rec = [cmap(int(l) % 10) for l in labels[:n_rec]]
    ax.scatter(Z_rec[:, 0], Z_rec[:, 1], s=35, c=colors_rec, marker='s',
               edgecolors='black', linewidths=0.5, alpha=0.98)

    cluster_ids = sorted(set(int(x) for x in labels))
    handles = [
        mlines.Line2D([0], [0], marker='o', color='black',
                      label=f'Cluster {cid}', markerfacecolor=cmap(cid % 10),
                      linestyle='', markersize=5)
        for cid in cluster_ids
    ]
    handles += [
        mlines.Line2D([0], [0], marker='s', color='black', linestyle='',
                      label='Recipes', markerfacecolor='none', markersize=5),
        mlines.Line2D([0], [0], marker='o', color='black', linestyle='',
                      label='Ingredients', markerfacecolor='none', markersize=5),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=FONT_SIZE_LEGEND)
    ax.set_title(f"t-SNE Projection (k={len(cluster_ids)})", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("t-SNE 1", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("t-SNE 2", fontsize=FONT_SIZE_LABEL)

    save_figure_fixed(fig, out_path)


def run_tsne_v2(raw_recipes, perplexity=5, k_min=2, k_max=12, outdir=None):
    """t-SNE pipeline with unified plot styling."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    if outdir is None:
        outdir = TSNE_DIR
    os.makedirs(outdir, exist_ok=True)

    R = np.array([compute_x_vector(normalize_weights(r['ingredients'])) for r in raw_recipes])
    recipe_names = [r['recipe_name'] for r in raw_recipes]

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

    tsne = TSNE(
        n_components=2, perplexity=_safe_perplexity(X.shape[0], perplexity),
        random_state=SEED, init="pca", learning_rate="auto", max_iter=1000,
    )
    Z = tsne.fit_transform(X)

    rng = np.random.RandomState(SEED)
    Z_ = Z + 1e-6 * rng.normal(size=Z.shape)

    sil_path = os.path.join(outdir, "silhouette.png")
    best_k = _pick_k_silhouette_v2(Z_, k_min=k_min, k_max=k_max, plot_path=sil_path, seed=SEED)

    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    labels = km.fit_predict(Z_)

    tsne_path = os.path.join(outdir, "tsne_bestk.png")
    _render_tsne_v2(Z, labels, recipe_names, ingredient_names, n_rec=len(recipe_names), out_path=tsne_path)

    _save_cluster_members_json(labels, recipe_names, ingredient_names, n_rec=len(recipe_names), outdir=outdir)

    FIVE = ["sweet", "sour", "salty", "bitter", "umami"]
    sensory_keys = [k for k in FIVE if k in SENSORY_ORDER] or list(SENSORY_ORDER[:5])
    rec_names_check, S_rec = _extract_recipe_sensory(raw_recipes, sensory_keys)
    assert rec_names_check == recipe_names, "Recipe order mismatch"

    labels_rec = labels[:len(recipe_names)]
    pvalues_path = os.path.join(outdir, "p_values.json")
    save_cluster_pvalues_json(
        labels_rec, S_rec,
        sensation_names=["sweet", "sour", "salty", "bitter", "umami"],
        out_path=pvalues_path,
    )

    print(f"[+] t-SNE v2: best k = {best_k}, output dir: {os.path.abspath(outdir)}")


def pca_clusters_v2(raw_recipes, n_components=2, standardize=False, outdir=None, k_min=2, k_max=12, seed=42):
    """PCA + KMeans pipeline with unified plot styling."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from plot_config import (
        setup_figure, save_figure_fixed, style_axes, FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_LEGEND,
    )

    if outdir is None:
        outdir = PCA_DIR
    os.makedirs(outdir, exist_ok=True)

    R = np.array([compute_x_vector(normalize_weights(r["ingredients"])) for r in raw_recipes], dtype=float)
    recipe_names = [r["recipe_name"] for r in raw_recipes]

    ing_map = {}
    for r in raw_recipes:
        for ing in r["ingredients"]:
            name = ing["name"]
            if name not in ing_map:
                s = ing["sensory_scores"]
                ing_map[name] = np.array([float(s[k]) for k in SENSORY_ORDER], dtype=float)
    ingredient_names = list(ing_map.keys())
    I = np.vstack([ing_map[n] for n in ingredient_names]) if ingredient_names else np.zeros((0, len(SENSORY_ORDER)))
    X = np.vstack([R, I]) if I.size else R
    n_rec = R.shape[0]

    if standardize:
        from sklearn.preprocessing import StandardScaler
        X0 = StandardScaler().fit_transform(X)
    else:
        X0 = X

    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False, random_state=seed)
    Z = pca.fit_transform(X0)
    evr = pca.explained_variance_ratio_
    var_pc1 = float(evr[0]) if len(evr) > 0 else 0.0
    var_pc2 = float(evr[1]) if len(evr) > 1 else 0.0

    # PC1 loadings bar plot
    pc1 = pca.components_[0]
    order = np.argsort(-np.abs(pc1))
    fig, ax = setup_figure(size="onehalf")
    style_axes(ax, grid=False)
    ax.bar(np.arange(len(pc1)), pc1[order], color="#4477AA", edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(np.arange(len(pc1)))
    ax.set_xticklabels([SENSORY_ORDER[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("PC1 Loading", fontsize=FONT_SIZE_LABEL)
    ax.set_title("PC1 Feature Loadings", fontsize=FONT_SIZE_TITLE)
    save_figure_fixed(fig, os.path.join(outdir, "pc1_loadings.png"))

    # Silhouette
    rng = np.random.RandomState(seed)
    Z_jit = Z + 1e-6 * rng.normal(size=Z.shape)

    n_samples = Z_jit.shape[0]
    if n_samples < 3:
        best_k = max(1, min(k_max, n_samples))
        best_s = np.nan
    else:
        k_min_eff = max(2, k_min)
        k_max_eff = max(k_min_eff, min(k_max, n_samples - 1))
        K_range = list(range(k_min_eff, k_max_eff + 1))
        scores = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels_k = km.fit_predict(Z_jit)
            unique_labels = np.unique(labels_k)
            if unique_labels.size < 2 or unique_labels.size > n_samples - 1:
                scores.append(np.nan)
                continue
            counts = np.bincount(labels_k)
            if counts.min() <= 1:
                scores.append(np.nan)
                continue
            try:
                s_val = silhouette_score(Z_jit, labels_k, metric="euclidean")
            except Exception:
                s_val = np.nan
            scores.append(s_val)

        if np.all(np.isnan(scores)):
            best_k = K_range[0]
            best_s = np.nan
        else:
            best_idx = int(np.nanargmax(scores))
            best_k = K_range[best_idx]
            best_s = float(scores[best_idx])

        # Silhouette plot
        fig, ax = setup_figure(size="single")
        style_axes(ax)
        y_vals = [s if (s is not None and not np.isnan(s)) else np.nan for s in scores]
        ax.plot(K_range, y_vals, marker="o", linewidth=1.5, markersize=5)
        if not np.all(np.isnan(y_vals)):
            ax.scatter([best_k], [best_s], s=120, facecolors="none",
                       edgecolors="k", linewidths=1.5, zorder=5)
            ax.annotate(f"k = {best_k}\n{best_s:.3f}", (best_k, best_s),
                        textcoords="offset points", xytext=(10, -25), fontsize=8)
        ax.set_xlabel("Clusters (k)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Silhouette score", fontsize=FONT_SIZE_LABEL)
        ax.set_title("KMeans Silhouette (PCA space)", fontsize=FONT_SIZE_TITLE)
        if len(K_range) > 0:
            ax.set_xticks(K_range)
        save_figure_fixed(fig, os.path.join(outdir, "pca_silhouette.png"))

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
    labels = km.fit_predict(Z_jit)

    # PCA scatter
    Z_rec = Z[:, :2][:n_rec]
    Z_ing = Z[:, :2][n_rec:] if Z.shape[0] > n_rec else np.empty((0, 2))
    labels_rec = labels[:n_rec]
    labels_ing = labels[n_rec:] if Z_ing.size else np.array([], dtype=int)

    cmap = plt.get_cmap("tab10")
    fig, ax = setup_figure(size="double")
    style_axes(ax)

    if Z_ing.size:
        ax.scatter(Z_ing[:, 0], Z_ing[:, 1], s=24, alpha=0.9, edgecolors="black",
                   linewidths=0.4, marker="o",
                   c=[cmap(int(c) % 10) for c in labels_ing])

    ax.scatter(Z_rec[:, 0], Z_rec[:, 1], s=32, alpha=0.95, edgecolors="black",
               linewidths=0.5, marker="s",
               c=[cmap(int(c) % 10) for c in labels_rec])

    handles = []
    cluster_ids = sorted(set(int(x) for x in labels))
    for cl in cluster_ids:
        handles.append(
            mlines.Line2D([0], [0], color='black', marker='o',
                          markerfacecolor=cmap(int(cl) % 10), linestyle='',
                          markersize=5, label=f'Cluster {cl}'))
    handles.append(mlines.Line2D([0], [0], color='black', marker='s', linestyle='',
                                 markersize=5, label='Recipes', markerfacecolor='none'))
    handles.append(mlines.Line2D([0], [0], color='black', marker='o', linestyle='',
                                 markersize=5, label='Ingredients', markerfacecolor='none'))

    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZE_LEGEND)
    ax.set_title(f"PCA Projection (k={best_k})", fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel(f"PC1 ({var_pc1*100:.1f}% var)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(f"PC2 ({var_pc2*100:.1f}% var)", fontsize=FONT_SIZE_LABEL)

    save_figure_fixed(fig, os.path.join(outdir, "pca_bestk_scatter.png"))

    # Cluster membership JSON (reuse existing helper logic)
    cluster_map = {}
    for idx, cl in enumerate(labels):
        cl = int(cl)
        if cl not in cluster_map:
            cluster_map[cl] = {"recipes": [], "ingredients": []}
        if idx < n_rec:
            cluster_map[cl]["recipes"].append(recipe_names[idx])
        else:
            cluster_map[cl]["ingredients"].append(ingredient_names[idx - n_rec])

    json_path = os.path.join(outdir, "clusters_bestk.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_map, f, ensure_ascii=False, indent=2)

    print(f"[+] PCA v2: best k = {best_k}, output dir: {os.path.abspath(outdir)}")


def parse_args():
    """Parse command-line flags for selective plot generation."""
    parser = argparse.ArgumentParser(description="Generate selected EDA plots without retraining models.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PLOT_GROUP_CHOICES,
        default=["all"],
        help=(
            "Plot groups to generate. Use one or more of: "
            "ingredient_usage, gaussians, combined, pca, tsne, clusters, all"
        ),
    )
    return parser.parse_args()


def resolve_plot_groups(selected_groups):
    """Expand convenience groups into concrete plot tasks."""
    groups = set(selected_groups or ["all"])
    if "all" in groups:
        return {"ingredient_usage", "gaussians", "combined", "pca", "tsne"}
    if "clusters" in groups:
        groups.remove("clusters")
        groups.update({"pca", "tsne"})
    return groups


# ---------------- Main ----------------

def main():
    args = parse_args()
    selected_groups = resolve_plot_groups(args.only)
    raw_recipes = load_attr_from_py(RAW_FILE, 'raw_recipes')

    from plot_config import apply_style
    apply_style(use_seaborn=True)

    if "ingredient_usage" in selected_groups:
        plot_ingredient_usage_pie_v2(raw_recipes, PLOTS_DIR, threshold=3)

    if "gaussians" in selected_groups:
        plot_gaussian_per_key_v2(raw_recipes, PLOTS_DIR, top_k=3)

    if "combined" in selected_groups:
        plot_combined_kde_v2(
            raw_recipes,
            PLOTS_DIR,
            exclude_keys={"bitter"},
            out_name="combined_distribution.png",
            title="Combined Score Distribution (Without Bitter)",
        )
        plot_combined_kde_v2(
            raw_recipes,
            PLOTS_DIR,
            exclude_keys=set(),
            out_name="combined_distribution_with_bitter.png",
            title="Combined Score Distribution (With Bitter)",
        )

    if "pca" in selected_groups:
        pca_clusters_v2(raw_recipes, n_components=2, standardize=False, outdir=PCA_DIR)

    if "tsne" in selected_groups:
        run_tsne_v2(raw_recipes, outdir=TSNE_DIR)

    chosen = ", ".join(sorted(selected_groups))
    print(f"[+] Completed plot groups: {chosen}")

if __name__ == "__main__":
    main()
