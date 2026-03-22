#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite figure: Panels A–E
  Row 1: Predicted vs Actual scatter — A=HS, B=RV
  Row 2: Predicted vs Actual scatter — C=Lasso, D=Hybrid
  Row 3: RMSE grouped boxplot spanning full width — E
         (4 boxes per sensory attribute: HS, RV, Lasso, Hybrid)

Prerequisites:
  Run python3 Lasso/src/train.py first to populate
  data/data_predictions.py with Lasso predictions.
  Run python3 HS_RV/src/compute_bounds.py first to populate
  data/hs_predictions.py and data/rv_predictions.py.

Usage (from repo root):
  python3 Lasso/src/composite_figure.py
"""

import csv
import importlib.util
import os
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

PROJECT_ROOT    = os.path.abspath(os.path.join(_HERE, os.pardir))
SHARED_DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "data"))
PLOTS_DIR       = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "results", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

RESULTS_DIR     = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "results"))
HS_PRED_FILE    = os.path.join(SHARED_DATA_DIR, "hs_predictions.py")
RV_PRED_FILE    = os.path.join(SHARED_DATA_DIR, "rv_predictions.py")
LASSO_CSV_FILE  = os.path.join(RESULTS_DIR, "models", "real_vs_predicted.csv")
RAW_RECIPES_FILE = os.path.join(SHARED_DATA_DIR, "raw_recipes.py")
OUTPUT_FILE     = os.path.join(PLOTS_DIR, "composite_figure.png")

# ── Import shared constants only (do NOT call apply_style) ─────────────
from plot_config import SENSORY_ORDER, SENSORY_COLORS, ALIASES  # noqa: E402

# ── Import hybrid model ─────────────────────────────────────────────────
from hybrid_analysis import build_analysis_df, loo_evaluate  # noqa: E402

# ── Global styling: sans-serif, all spines on by default ───────────────
plt.rcParams.update({
    "font.family":                    "sans-serif",
    "font.sans-serif":                ["Arial", "DejaVu Sans"],
    "font.size":                      9,
    "axes.spines.top":                True,
    "axes.spines.right":              True,
    "figure.constrained_layout.use":  False,
})

# ── Box colours for panel E ─────────────────────────────────────────────
BOX_COLORS  = {"HS": "#a8c4d4", "RV": "#90c090", "Lasso": "#f5c07a", "Hybrid": "#c8a4d4"}
METHOD_ORDER = ["HS", "RV", "Lasso", "Hybrid"]

# ── Hybrid taste order (matches hybrid_analysis.py) ────────────────────
_HYBRID_TASTES = ["sweet", "sour", "bitter", "umami", "salty"]


# ── Inline helpers ──────────────────────────────────────────────────────

def _r2(actuals, preds):
    """Inline R² to avoid sklearn import."""
    a, p = np.asarray(actuals, float), np.asarray(preds, float)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _load_attr(filepath, variable_name):
    spec = importlib.util.spec_from_file_location(variable_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, variable_name)


def _normalize(d):
    """Map long/short attribute names to canonical short names via ALIASES."""
    out = {}
    for k, v in d.items():
        ck = ALIASES.get(k)
        if ck in SENSORY_ORDER:
            out[ck] = float(v)
    return out


def _extract(pred_dict, method_key):
    """Return (preds, actuals, labels) arrays from a predictions dict."""
    preds, actuals, labels = [], [], []
    for entry in pred_dict.values():
        if not isinstance(entry, dict):
            continue
        if "Actual" not in entry or method_key not in entry:
            continue
        act = _normalize(entry["Actual"])
        prd = _normalize(entry[method_key])
        for k in SENSORY_ORDER:
            if k in act and k in prd:
                preds.append(prd[k])
                actuals.append(act[k])
                labels.append(k)
    return (
        np.array(preds,   dtype=float),
        np.array(actuals, dtype=float),
        np.array(labels),
    )


# ── Data loading ────────────────────────────────────────────────────────

def _load_lasso_csv(csv_path):
    """Load Lasso results from real_vs_predicted.csv (label, actual, predicted)."""
    preds, actuals, labels = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"].strip()
            if lbl in SENSORY_ORDER:
                actuals.append(float(row["actual"]))
                preds.append(float(row["predicted"]))
                labels.append(lbl)
    return (
        np.array(preds,   dtype=float),
        np.array(actuals, dtype=float),
        np.array(labels),
    )


def _load_hybrid(raw_recipes_path):
    """Compute Hybrid (HS bounds + chemistry features) LOO predictions."""
    print("[*] Computing Hybrid LOO predictions (this may take a moment)...")
    raw_recipes = _load_attr(raw_recipes_path, "raw_recipes")
    df, _ = build_analysis_df(raw_recipes)

    chem_cols = [
        "protein_frac", "sugar_frac", "maillard_potential", "salt_frac",
        "water_frac", "conc_factor", "allium_frac", "fermented_frac",
    ]
    alphas = np.logspace(-3, 1, 30)

    preds, actuals, labels = [], [], []
    for t in _HYBRID_TASTES:
        actual = df[f"{t}_actual"].values
        feat   = [f"{t}_hs_mid", f"{t}_voigt"] + chem_cols
        X      = df[feat].values.astype(float)
        pred   = loo_evaluate(X, actual, alphas)
        preds.extend(pred.tolist())
        actuals.extend(actual.tolist())
        labels.extend([t] * len(actual))

    return (
        np.array(preds,   dtype=float),
        np.array(actuals, dtype=float),
        np.array(labels),
    )


def load_all():
    hs_data = _load_attr(HS_PRED_FILE, "hs_predictions")
    rv_data = _load_attr(RV_PRED_FILE, "rv_predictions")

    hs_p, hs_a, hs_l = _extract(hs_data, "HS prediction")
    rv_p, rv_a, rv_l = _extract(rv_data, "RV prediction")

    if not os.path.exists(LASSO_CSV_FILE):
        raise RuntimeError(
            f"Lasso predictions not found at:\n  {LASSO_CSV_FILE}\n"
            "Run:  python3 Lasso/src/train.py  first."
        )
    lasso_p, lasso_a, lasso_l = _load_lasso_csv(LASSO_CSV_FILE)

    hybrid_p, hybrid_a, hybrid_l = _load_hybrid(RAW_RECIPES_FILE)

    return {
        "HS":     (hs_p,     hs_a,     hs_l),
        "RV":     (rv_p,     rv_a,     rv_l),
        "Lasso":  (lasso_p,  lasso_a,  lasso_l),
        "Hybrid": (hybrid_p, hybrid_a, hybrid_l),
    }


# ── Panel A / B / C / D  — Predicted vs Actual scatter ─────────────────

def _scatter_panel(ax, preds, actuals, labels, method_name, panel_label):
    # Axis range: data-driven, rounded to nearest 10
    all_vals = np.concatenate([preds, actuals])
    raw_max  = np.ceil(np.max(all_vals) / 10) * 10
    ax_max   = raw_max * 1.05
    ax_min   = 0.0

    # Grid behind everything
    ax.set_axisbelow(True)
    ax.grid(True, color="#ccddee", linewidth=0.5, zorder=0)

    # All four spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    # Ideal diagonal
    ax.plot([ax_min, ax_max], [ax_min, ax_max],
            linestyle="--", color="black", linewidth=0.8, zorder=2)

    # Scatter points coloured by sensory attribute
    colors = [SENSORY_COLORS[lbl] for lbl in labels]
    ax.scatter(preds, actuals, c=colors, s=22, alpha=0.8,
               edgecolors="none", zorder=3)

    # Square axes at data range
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect("equal", adjustable="box")
    tick_step = 10
    ticks = np.arange(0, raw_max + tick_step, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(method_name,  fontsize=11, fontweight="bold", pad=4)

    # PCC / R² annotation box (top-left)
    pcc, _ = pearsonr(preds, actuals) if len(preds) >= 2 else (float("nan"), None)
    r2     = _r2(actuals, preds)       if len(preds) >= 2 else float("nan")
    ax.text(
        0.05, 0.95,
        f"PCC = {pcc:.2f}\n$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", linewidth=0.8),
    )

    # Sensory + ideal legend (bottom-right)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markersize=4, color=SENSORY_COLORS[k],
                   label=k.capitalize())
        for k in SENSORY_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], color="black", linestyle="--",
                   linewidth=0.8, label="Ideal")
    )
    ax.legend(handles=handles, loc="lower right",
              fontsize=7, frameon=False)

    # Bold panel label (top-left, outside axes)
    ax.text(-0.12, 1.08, panel_label,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="bottom", ha="left")


# ── Panel E — RMSE grouped boxplot (4 methods) ─────────────────────────

def _boxplot_panel(ax, data_dict, panel_label):
    GROUP_W = 4.5   # spacing between sensory attribute groups (4 boxes each)

    positions, box_data, box_colors = [], [], []
    for i, sens in enumerate(SENSORY_ORDER):
        base = i * GROUP_W
        for j, method in enumerate(METHOD_ORDER):
            preds, actuals, labels = data_dict[method]
            mask = (labels == sens)
            err  = np.abs(preds[mask] - actuals[mask])
            positions.append(base + j)
            box_data.append(err)
            box_colors.append(BOX_COLORS[method])

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black",     linewidth=1.5),
        whiskerprops=dict(color="#444444",  linewidth=1.0),
        capprops=dict(   color="#444444",  linewidth=1.0),
        boxprops=dict(                      linewidth=0.8),
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.95)
        patch.set_edgecolor("#555555")

    # Spines: only left and bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Horizontal dashed grid only
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--",
            color="lightgrey", linewidth=0.8, zorder=0)

    # Y axis
    ax.set_ylim(bottom=0)
    y_top     = ax.get_ylim()[1]
    tick_step = 10
    ax.set_yticks(np.arange(0, y_top + tick_step, tick_step))
    ax.set_ylabel("RMSE", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)

    # Title
    ax.set_title("Taste prediction", fontsize=12,
                 fontweight="bold", pad=6)

    # Level-1 tick labels: HS / RV / Lasso / Hybrid repeated per group
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [METHOD_ORDER[j % 4] for j in range(len(positions))],
        fontsize=7,
    )
    ax.tick_params(axis="x", length=0)

    # Level-2 labels: sensory attribute name centred under each group of 4
    for i, sens in enumerate(SENSORY_ORDER):
        centre = i * GROUP_W + 1.5   # midpoint of positions base+0, +1, +2, +3
        ax.text(
            centre, -0.16, sens.capitalize(),
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=10, fontweight="bold",
        )

    # X limits with padding
    ax.set_xlim(positions[0] - 0.8, positions[-1] + 0.8)

    # Legend with filled square patches
    legend_handles = [
        mpatches.Patch(facecolor=BOX_COLORS[m], edgecolor="#555555",
                       linewidth=0.6, label=m)
        for m in METHOD_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, frameon=True)

    # Bold panel label (top-left, outside axes)
    ax.text(-0.06, 1.05, panel_label,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="bottom", ha="left")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("[*] Loading prediction data...")
    data = load_all()

    print("[*] Building composite figure...")
    fig = plt.figure(figsize=(9.0, 13.0), facecolor="white")
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 1.3],
        hspace=0.55,
        wspace=0.38,
    )

    ax_hs     = fig.add_subplot(gs[0, 0])
    ax_rv     = fig.add_subplot(gs[0, 1])
    ax_lasso  = fig.add_subplot(gs[1, 0])
    ax_hybrid = fig.add_subplot(gs[1, 1])
    ax_box    = fig.add_subplot(gs[2, :])

    for ax, method, label in [
        (ax_hs,     "HS",     "A."),
        (ax_rv,     "RV",     "B."),
        (ax_lasso,  "Lasso",  "C."),
        (ax_hybrid, "Hybrid", "D."),
    ]:
        p, a, l = data[method]
        _scatter_panel(ax, p, a, l, method, label)

    _boxplot_panel(ax_box, data, "E.")

    fig.savefig(
        OUTPUT_FILE, dpi=600,
        bbox_inches="tight", pad_inches=0.12,
        facecolor="white",
    )
    plt.close(fig)
    print(f"[+] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
