#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-modal taste interaction analysis (reviewer response figures).

Addresses the concern that the linear mixture model ignores perceptual taste
interactions (sweetness suppressing bitterness, salt enhancing umami, etc.).

Outputs under results/interactions/:
  1) ingredient_sensory_profiles.png -- every ingredient x 5 sensory attributes
  2) sensory_interactions.pdf -- (A) ingredient-level taste co-occurrence,
     (B) recipe-level measured taste correlation, (C) LOO-residual correlation,
     (D) residual vs. pairwise cross-modal interaction terms, (E) does adding
     interaction terms improve out-of-sample error?
  plus CSV/TXT tables with every r, p, and BH-FDR q value quoted in the figures.

Everything here is computed from the raw recipe file only; it does not read the
trained-model artifacts, so the figures stand independent of the training run.

Usage:
  python3 Lasso/src/interaction_plots.py
  python3 Lasso/src/interaction_plots.py --only heatmap
  python3 Lasso/src/interaction_plots.py --only correlations
"""

import argparse
import importlib.util
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from env_config import get_raw_recipes_path
from plot_config import (
    FONT_SIZE_ANNOTATION, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TITLE,
    SENSORY_COLORS, SENSORY_ORDER, apply_style, save_figure_fixed,
)

# ---------------- Paths ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
OUT_DIR = os.path.join(REPO_ROOT, "results", "interactions")

SEED = 42
ALPHA = 0.05

# Diverging / sequential maps built from the project's Tol Bright palette.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "tol_div", ["#4477AA", "#8CB0D0", "#FFFFFF", "#F3A3AC", "#EE6677"])
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "tol_seq", ["#FFFFFF", "#C9DCEA", "#7FA8CB", "#4477AA", "#1F3B5C"])

PAIRS = list(itertools.combinations(range(len(SENSORY_ORDER)), 2))
PAIR_LABELS = [f"{SENSORY_ORDER[a]}×{SENSORY_ORDER[b]}" for a, b in PAIRS]


# ---------------- Data ----------------
def load_attr_from_py(filepath, variable_name):
    spec = importlib.util.spec_from_file_location(variable_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, variable_name)


def build_matrices(raw_recipes):
    """Return ingredient profiles, usage counts, mixture features and targets.

    ingredients : (n_ing, 5) raw sensory scores, one row per unique ingredient
    counts      : recipes each ingredient appears in
    mixture     : (n_rec, 5) weight-normalised average of ingredient scores --
                  this is the additive, interaction-free prediction
    measured    : (n_rec, 5) the recipe's own sensory scores
    """
    profiles, counts = {}, {}
    mixture, measured = [], []

    for r in raw_recipes:
        total = sum(ing["weight"] for ing in r["ingredients"]) or 1.0
        vec = np.zeros(len(SENSORY_ORDER))
        for ing in r["ingredients"]:
            scores = np.array([float(ing["sensory_scores"][k]) for k in SENSORY_ORDER])
            vec += (ing["weight"] / total) * scores
            profiles.setdefault(ing["name"], scores)
            counts[ing["name"]] = counts.get(ing["name"], 0) + 1
        mixture.append(vec)
        measured.append([float(r["food_sensory_scores"][k]) for k in SENSORY_ORDER])

    names = list(profiles)
    return (names,
            np.vstack([profiles[n] for n in names]),
            np.array([counts[n] for n in names]),
            np.vstack(mixture),
            np.vstack(measured))


# ---------------- Statistics ----------------
def corr_matrix(data):
    """Pearson r and p for every column pair of `data`."""
    k = data.shape[1]
    r = np.eye(k)
    p = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            r[i, j] = r[j, i] = pearsonr(data[:, i], data[:, j])[0]
            p[i, j] = p[j, i] = pearsonr(data[:, i], data[:, j])[1]
    return r, p


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, float).ravel()
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        running = min(running, p[idx] * n / rank)
        q[idx] = running
    return q.reshape(np.shape(pvals))


def loo_residuals(features, targets):
    """Out-of-sample residuals of the additive (interaction-free) linear model."""
    loo = LeaveOneOut()
    resid = np.empty_like(targets, dtype=float)
    for i in range(targets.shape[1]):
        pred = cross_val_predict(LinearRegression(), features, targets[:, i], cv=loo)
        resid[:, i] = targets[:, i] - pred
    return resid


def interaction_gain(features, interactions, targets):
    """LOO RMSE with and without the 10 pairwise cross-modal interaction terms."""
    loo = LeaveOneOut()
    both = np.hstack([features, interactions])
    rows = []
    for i, key in enumerate(SENSORY_ORDER):
        y = targets[:, i]
        base = cross_val_predict(LinearRegression(), features, y, cv=loo)
        full = cross_val_predict(LinearRegression(), both, y, cv=loo)
        rmse_base = float(np.sqrt(np.mean((base - y) ** 2)))
        rmse_full = float(np.sqrt(np.mean((full - y) ** 2)))
        rows.append({
            "target": key,
            "rmse_additive": rmse_base,
            "rmse_with_interactions": rmse_full,
            "delta_rmse": rmse_full - rmse_base,
        })
    return rows


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA else ""


def panel_label(ax, letter, dx=-17, dy=9):
    """Bold panel letter, offset outside the axes so it reads as a label.

    Offsets are in points rather than axes fractions so the letter sits the same
    distance from every panel regardless of how wide that panel is. Panel D
    overrides dx because its title is centred over a very wide axes and would
    otherwise crowd the letter.
    """
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=FONT_SIZE_ANNOTATION + 2.5, weight="bold",
                ha="left", va="bottom")


# ---------------- Figure 1: ingredient x sensory ----------------
def plot_ingredient_profiles(names, profiles, out_path, n_panels=2):
    """Heatmap of every ingredient's sensory profile, grouped by dominant taste."""
    peak = profiles.max(axis=1)
    dominant = np.where(peak > 0, profiles.argmax(axis=1), len(SENSORY_ORDER))
    order = sorted(range(len(names)), key=lambda i: (dominant[i], -peak[i], names[i]))

    # Flag the sugar+salt ingredients the reviewer singles out.
    q75 = np.percentile(profiles, 75, axis=0)
    i_sweet, i_salty = SENSORY_ORDER.index("sweet"), SENSORY_ORDER.index("salty")
    dual = {i for i in range(len(names))
            if profiles[i, i_sweet] >= q75[i_sweet] and profiles[i, i_salty] >= q75[i_salty]}

    chunks = np.array_split(np.array(order), n_panels)
    # Sugar/salt/acid saturate at 100 while most ingredients sit below 20, so a
    # square-root norm is needed to keep the low-intensity structure visible.
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=float(profiles.max()))

    fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 9.4), constrained_layout=False)
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, chunks):
        block = profiles[idx]
        ax.imshow(block, cmap=CMAP_SEQ, norm=norm, aspect="auto", interpolation="nearest")

        ax.set_xticks(range(len(SENSORY_ORDER)))
        ax.set_xticklabels([k.capitalize() for k in SENSORY_ORDER],
                           rotation=90, fontsize=6)
        for tick, key in zip(ax.get_xticklabels(), SENSORY_ORDER):
            tick.set_color(SENSORY_COLORS[key])
            tick.set_fontweight("bold")
        ax.xaxis.set_ticks_position("top")

        labels = [f"{'◆ ' if i in dual else ''}{names[i][:34]}" for i in idx]
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(labels, fontsize=4.6)
        for tick, i in zip(ax.get_yticklabels(), idx):
            if dominant[i] < len(SENSORY_ORDER):
                tick.set_color(SENSORY_COLORS[SENSORY_ORDER[dominant[i]]])

        # Faint separators between dominant-taste groups.
        for row in range(1, len(idx)):
            if dominant[idx[row]] != dominant[idx[row - 1]]:
                ax.axhline(row - 0.5, color="#333333", linewidth=0.7)

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#888888")

    fig.subplots_adjust(left=0.195, right=0.985, top=0.900, bottom=0.070, wspace=0.62)

    cax = fig.add_axes([0.30, 0.048, 0.42, 0.010])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_SEQ),
                        cax=cax, orientation="horizontal",
                        ticks=[0, 5, 20, 45, 80, 100])
    cbar.set_label("Sensory score (0–100, √-scaled)", fontsize=FONT_SIZE_ANNOTATION)
    cbar.ax.tick_params(labelsize=FONT_SIZE_ANNOTATION - 1)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle("Figure R1.  Ingredient Sensory Profiles", fontsize=FONT_SIZE_TITLE,
                 weight="bold", y=0.978)
    fig.text(0.5, 0.952,
             "Rows grouped and coloured by dominant taste.  "
             "◆ = upper-quartile in both sweet and salty.",
             ha="center", fontsize=FONT_SIZE_ANNOTATION - 0.5, color="#555555")

    save_figure_fixed(fig, out_path)
    return dual


def _draw_corr_panel(ax, r, p, letter, title, n, show_ylabels=True):
    ax.imshow(r, cmap=CMAP_DIV, vmin=-1, vmax=1, aspect="equal")
    k = len(SENSORY_ORDER)
    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=FONT_SIZE_ANNOTATION, color="#777777")
                continue
            ax.text(j, i, f"{r[i, j]:.2f}\n{stars(p[i, j])}", ha="center", va="center",
                    fontsize=FONT_SIZE_ANNOTATION - 1.2,
                    color="white" if abs(r[i, j]) > 0.55 else "#222222")

    ax.set_xticks(range(k))
    ax.set_xticklabels([s.capitalize() for s in SENSORY_ORDER], rotation=45,
                       ha="right", fontsize=FONT_SIZE_ANNOTATION)
    ax.set_yticks(range(k))
    if show_ylabels:
        ax.set_yticklabels([s.capitalize() for s in SENSORY_ORDER],
                           fontsize=FONT_SIZE_ANNOTATION)
    else:
        ax.set_yticklabels([])
    for tick, key in zip(ax.get_xticklabels(), SENSORY_ORDER):
        tick.set_color(SENSORY_COLORS[key])
    if show_ylabels:
        for tick, key in zip(ax.get_yticklabels(), SENSORY_ORDER):
            tick.set_color(SENSORY_COLORS[key])
    ax.set_title(f"{title}  (n = {n})", fontsize=FONT_SIZE_ANNOTATION + 0.5,
                 weight="bold", pad=7)
    panel_label(ax, letter)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------- Figure 2: sensory-sensory correlation ----------------
def plot_sensory_interactions(profiles, mixture, measured, out_path):
    """Taste-taste correlation at ingredient, recipe and residual level."""
    z = (mixture - mixture.mean(axis=0)) / mixture.std(axis=0)
    interactions = np.column_stack([z[:, a] * z[:, b] for a, b in PAIRS])
    resid = loo_residuals(z, measured)

    r_ing, p_ing = corr_matrix(profiles)
    r_rec, p_rec = corr_matrix(measured)
    r_res, p_res = corr_matrix(resid)

    # Residual vs. every pairwise interaction term.
    r_int = np.zeros((len(SENSORY_ORDER), len(PAIRS)))
    p_int = np.zeros_like(r_int)
    for i in range(len(SENSORY_ORDER)):
        for j in range(len(PAIRS)):
            r_int[i, j], p_int[i, j] = pearsonr(resid[:, i], interactions[:, j])
    q_int = bh_fdr(p_int)

    gains = interaction_gain(z, interactions, measured)

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=False)
    gs_top = fig.add_gridspec(1, 3, left=0.085, right=0.965,
                              top=0.900, bottom=0.600, wspace=0.30)
    gs_bot = fig.add_gridspec(1, 3, left=0.085, right=0.965,
                              top=0.465, bottom=0.205, wspace=0.55,
                              width_ratios=[1.0, 1.0, 0.72])

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])
    _draw_corr_panel(ax_a, r_ing, p_ing, "A", "Ingredient profiles", profiles.shape[0])
    _draw_corr_panel(ax_b, r_rec, p_rec, "B", "Measured recipes", measured.shape[0], False)
    _draw_corr_panel(ax_c, r_res, p_res, "C", "Residuals (LOO)", measured.shape[0], False)

    # Panel D -- residual vs interaction terms
    ax_d = fig.add_subplot(gs_bot[0, 0:2])
    lim = float(np.abs(r_int).max())
    im = ax_d.imshow(r_int, cmap=CMAP_DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(r_int.shape[0]):
        for j in range(r_int.shape[1]):
            mark = "●" if q_int[i, j] < ALPHA else ("*" if p_int[i, j] < ALPHA else "")
            ax_d.text(j, i, f"{r_int[i, j]:+.2f}{mark}", ha="center", va="center",
                      fontsize=FONT_SIZE_ANNOTATION - 2.2,
                      color="white" if abs(r_int[i, j]) > 0.7 * lim else "#222222")
    ax_d.set_xticks(range(len(PAIR_LABELS)))
    ax_d.set_xticklabels(PAIR_LABELS, rotation=45, ha="right",
                         fontsize=FONT_SIZE_ANNOTATION - 1)
    ax_d.set_yticks(range(len(SENSORY_ORDER)))
    ax_d.set_yticklabels([s.capitalize() for s in SENSORY_ORDER],
                         fontsize=FONT_SIZE_ANNOTATION)
    for tick, key in zip(ax_d.get_yticklabels(), SENSORY_ORDER):
        tick.set_color(SENSORY_COLORS[key])
    ax_d.set_ylabel("Residual", fontsize=FONT_SIZE_ANNOTATION + 1)
    ax_d.set_xlabel("Cross-modal interaction term", fontsize=FONT_SIZE_ANNOTATION + 1)
    n_nominal = int((p_int < ALPHA).sum())
    n_fdr = int((q_int < ALPHA).sum())
    ax_d.set_title(
        f"Residual × cross-modal interaction term   "
        f"({n_nominal}/{r_int.size} at p<0.05,  {n_fdr}/{r_int.size} at FDR q<0.05)",
        fontsize=FONT_SIZE_ANNOTATION + 0.5, weight="bold", pad=7)
    panel_label(ax_d, "D", dx=-34)
    ax_d.tick_params(length=0)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(im, ax=ax_d, fraction=0.025, pad=0.015)
    cb.set_label("Pearson r", fontsize=FONT_SIZE_ANNOTATION - 1)
    cb.ax.tick_params(labelsize=FONT_SIZE_ANNOTATION - 2)
    cb.outline.set_linewidth(0.5)

    # Panel E -- does adding interaction terms pay off out-of-sample?
    ax_e = fig.add_subplot(gs_bot[0, 2])
    deltas = [g["delta_rmse"] for g in gains]
    colors = [SENSORY_COLORS[g["target"]] for g in gains]
    ypos = np.arange(len(gains))
    ax_e.barh(ypos, deltas, color=colors, edgecolor="#333333", linewidth=0.5, height=0.65)
    ax_e.axvline(0, color="#333333", linewidth=0.8)
    ax_e.set_yticks(ypos)
    ax_e.set_yticklabels([g["target"].capitalize() for g in gains],
                         fontsize=FONT_SIZE_ANNOTATION)
    for tick, g in zip(ax_e.get_yticklabels(), gains):
        tick.set_color(SENSORY_COLORS[g["target"]])
    ax_e.invert_yaxis()
    pad = max(abs(min(deltas)), abs(max(deltas))) * 0.45
    ax_e.set_xlim(min(deltas) - pad, max(deltas) + pad)
    for y, d in zip(ypos, deltas):
        ax_e.text(d + (pad * 0.12 if d >= 0 else -pad * 0.12), y, f"{d:+.2f}",
                  va="center", ha="left" if d >= 0 else "right",
                  fontsize=FONT_SIZE_ANNOTATION - 1.5, color="#333333")
    ax_e.set_xlabel("Δ LOO RMSE\n(with − without interactions)",
                    fontsize=FONT_SIZE_ANNOTATION)
    ax_e.set_title("Cost of adding\ninteraction terms",
                   fontsize=FONT_SIZE_ANNOTATION + 0.5, weight="bold", pad=7)
    panel_label(ax_e, "E")
    ax_e.grid(True, axis="x", linestyle="--", alpha=0.3, linewidth=0.6, color="#CCCCCC")
    ax_e.tick_params(labelsize=FONT_SIZE_ANNOTATION - 1, length=3)
    ax_e.text(0.5, -0.40, "positive ⇒ interactions hurt", transform=ax_e.transAxes,
              ha="center", fontsize=FONT_SIZE_ANNOTATION - 1.5, color="#777777")

    fig.suptitle("Cross-Modal Taste Correlation and Interaction Diagnostics",
                 fontsize=FONT_SIZE_TITLE, weight="bold", y=0.962)
    fig.text(0.5, 0.032,
             "A–C: * p<0.05, ** p<0.01, *** p<0.001.    "
             "D: * nominal p<0.05, ● Benjamini–Hochberg q<0.05.",
             ha="center", fontsize=FONT_SIZE_ANNOTATION - 1, color="#555555")

    save_figure_fixed(fig, out_path)
    return dict(r_ing=r_ing, p_ing=p_ing, r_rec=r_rec, p_rec=p_rec,
                r_res=r_res, p_res=p_res, r_int=r_int, p_int=p_int,
                q_int=q_int, gains=gains, resid=resid)


# ---------------- Tables ----------------
def export_tables(names, profiles, counts, stats, out_dir):
    import csv

    with open(os.path.join(out_dir, "ingredient_sensory_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ingredient", "n_recipes"] + SENSORY_ORDER)
        for i, name in enumerate(names):
            w.writerow([name, int(counts[i])] + [f"{v:g}" for v in profiles[i]])

    with open(os.path.join(out_dir, "sensory_correlations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "taste_a", "taste_b", "pearson_r", "p_value"])
        for level, (r, p) in (("ingredient", (stats["r_ing"], stats["p_ing"])),
                              ("recipe_measured", (stats["r_rec"], stats["p_rec"])),
                              ("additive_residual", (stats["r_res"], stats["p_res"]))):
            for i, j in PAIRS:
                w.writerow([level, SENSORY_ORDER[i], SENSORY_ORDER[j],
                            f"{r[i, j]:.4f}", f"{p[i, j]:.4g}"])

    with open(os.path.join(out_dir, "interaction_tests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residual_taste", "interaction_term", "pearson_r", "p_value", "bh_q"])
        for i, key in enumerate(SENSORY_ORDER):
            for j, lab in enumerate(PAIR_LABELS):
                w.writerow([key, lab, f"{stats['r_int'][i, j]:.4f}",
                            f"{stats['p_int'][i, j]:.4g}", f"{stats['q_int'][i, j]:.4g}"])

    with open(os.path.join(out_dir, "loo_interaction_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "rmse_additive",
                                          "rmse_with_interactions", "delta_rmse"])
        w.writeheader()
        for row in stats["gains"]:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    n_nominal = int((stats["p_int"] < ALPHA).sum())
    n_fdr = int((stats["q_int"] < ALPHA).sum())
    worse = [g["target"] for g in stats["gains"] if g["delta_rmse"] > 0]
    lines = [
        "Cross-modal interaction diagnostics",
        "=" * 42,
        f"Recipes: {stats['resid'].shape[0]}   Unique ingredients: {len(names)}",
        "",
        "Residual x interaction-term tests "
        f"({stats['r_int'].size} tests, Pearson):",
        f"  nominal p < {ALPHA}: {n_nominal}",
        f"  Benjamini-Hochberg q < {ALPHA}: {n_fdr}",
        f"  max |r|: {np.abs(stats['r_int']).max():.3f}",
        "",
        "Leave-one-out RMSE, additive model vs. + 10 pairwise interaction terms:",
    ]
    for g in stats["gains"]:
        lines.append(f"  {g['target']:<7} {g['rmse_additive']:7.3f} -> "
                     f"{g['rmse_with_interactions']:7.3f}   "
                     f"delta = {g['delta_rmse']:+.3f}")
    lines += [
        "",
        f"Interaction terms increased out-of-sample RMSE for "
        f"{len(worse)}/{len(stats['gains'])} attributes ({', '.join(worse)}).",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", nargs="+", default=["all"],
                        choices=["all", "heatmap", "correlations"],
                        help="Which figures to generate.")
    args = parser.parse_args()
    groups = {"heatmap", "correlations"} if "all" in args.only else set(args.only)

    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    apply_style()

    raw_recipes = load_attr_from_py(get_raw_recipes_path(), "raw_recipes")
    names, profiles, counts, mixture, measured = build_matrices(raw_recipes)
    print(f"[INFO] {len(raw_recipes)} recipes, {len(names)} unique ingredients")

    if "heatmap" in groups:
        dual = plot_ingredient_profiles(
            names, profiles,
            os.path.join(OUT_DIR, "ingredient_sensory_profiles.png"))
        print(f"[INFO] {len(dual)} ingredients upper-quartile in both sweet and salty")

    if "correlations" in groups:
        stats = plot_sensory_interactions(
            profiles, mixture, measured,
            os.path.join(OUT_DIR, "sensory_interactions.pdf"))
        export_tables(names, profiles, counts, stats, OUT_DIR)

    print(f"[+] Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
