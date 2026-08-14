#!/usr/bin/env python3
"""Bitterness analysis — supplementary figure.

    python3 bitterness_study/src/make_bitterness_figure.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "Lasso", "src"))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.metrics import roc_curve                          # noqa: E402
from sklearn.model_selection import LeaveOneOut                # noqa: E402
from sklearn.pipeline import Pipeline                          # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

from data import FIGURES, RESULTS, load                        # noqa: E402

try:
    from plot_config import SENSORY_COLORS, apply_style, style_axes
    apply_style()
    C = dict(SENSORY_COLORS)
except Exception:
    def style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3, color="#CCC")
    C = {"sweet": "#4477AA", "bitter": "#EE6677", "salty": "#228833",
         "umami": "#CCBB44", "sour": "#AA3377"}

C_BIT, C_SWE = C["bitter"], C["sweet"]
GREY, DARK = "#888888", "#444444"

# Best-performing fitted model (lowest leave-one-out MAE among fitted structures).
BEST = "log1p-transform Ridge"

FAM_COLOR = {"baseline": "#777777", "physical": "#CCBB44", "as-published": C_SWE,
             "floor-aware": C_BIT, "linear": "#AA3377", "nonlinear": "#228833"}
FAM_LABEL = {"baseline": "constant baseline", "physical": "unfitted physical",
             "as-published": "published models", "floor-aware": "censored / discrete",
             "linear": "other linear", "nonlinear": "nonlinear"}

LETTER_SIZE = 15.0
TITLE_SIZE = 8.8


def label(ax, letter, title, dx=-0.17, dy=1.045, pad=13):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=LETTER_SIZE,
            fontweight="bold", va="bottom", ha="left")
    ax.set_title(title, loc="left", fontsize=TITLE_SIZE, color="#111111", pad=pad)


# ── A ───────────────────────────────────────────────────────────────────────
def panel_distribution(ax, ds):
    y = ds.y("bitter")
    vals, cnts = np.unique(y, return_counts=True)
    ax.bar(vals, cnts, width=0.72, color=C_BIT, edgecolor="black", linewidth=0.5)
    for v, c in zip(vals, cnts):
        ax.text(v, c + 1.2, str(int(c)), ha="center", fontsize=6.8)
    ax.set_xlabel("Ground-truth bitterness (0–100 scale)", fontsize=8)
    ax.set_ylabel("Recipes", fontsize=8)
    ax.set_xlim(-0.9, 9.4)
    ax.set_ylim(0, cnts.max() * 1.22)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.tick_params(labelsize=7)
    label(ax, "A", "Distribution of ground-truth bitterness scores")
    style_axes(ax)


# ── B ───────────────────────────────────────────────────────────────────────
def panel_predictors(ax_mae, ax_r2, bench):
    b = bench[bench.taste == "bitter"].sort_values("MAE").reset_index(drop=True)
    ypos = np.arange(len(b))
    colors = [FAM_COLOR.get(f, "#228833") for f in b.family]

    ax_mae.barh(ypos, b.MAE, color=colors, edgecolor="black", linewidth=0.35, height=0.74)
    ax_mae.set_yticks(ypos)
    ax_mae.set_yticklabels(b.model, fontsize=6.5)
    ax_mae.invert_yaxis()
    const = float(b.loc[b.model == "Constant (median)", "MAE"].iloc[0])
    ax_mae.axvline(const, color="black", linestyle="--", linewidth=1.0, zorder=5)
    ax_mae.set_xlabel("Leave-one-out MAE", fontsize=8)
    ax_mae.set_xlim(0, float(b.MAE.max()) * 1.05)
    ax_mae.tick_params(axis="x", labelsize=7)
    ax_mae.text(const - 0.03, -0.55, "constant = 1.0", fontsize=6.6,
                ha="right", va="bottom", color=DARK)
    label(ax_mae, "B", "Leave-one-out performance of 21 predictors",
          dx=-0.62, dy=1.012)
    style_axes(ax_mae)

    LO, HI = -0.55, 0.42
    r2 = b.R2_oos.values
    ax_r2.barh(ypos, np.clip(r2, LO * 0.97, HI), color=colors, edgecolor="black",
               linewidth=0.35, height=0.74, alpha=0.9)
    ax_r2.axvline(0, color="black", linewidth=0.9, zorder=3)
    for i, v in enumerate(r2):
        if v < LO:
            ax_r2.text(LO * 0.90, i, f"{v:.2f}", fontsize=5.6, color=DARK,
                       va="center", ha="left")
    ax_r2.set_yticks(ypos)
    ax_r2.set_yticklabels([])
    ax_r2.invert_yaxis()
    ax_r2.set_xlabel("Out-of-sample $R^2$", fontsize=8)
    ax_r2.set_xlim(LO, HI)
    ax_r2.set_xticks([-0.4, 0.0, 0.4])
    ax_r2.tick_params(axis="x", labelsize=7)
    style_axes(ax_r2)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=FAM_COLOR[f], ec="black", lw=0.35)
               for f in FAM_LABEL]
    ax_mae.legend(handles, list(FAM_LABEL.values()), fontsize=6.1, loc="upper right",
                  bbox_to_anchor=(1.005, 0.985), frameon=True, framealpha=0.95,
                  borderpad=0.4, handlelength=1.1, labelspacing=0.34)


# ── C ───────────────────────────────────────────────────────────────────────
def panel_permutation(ax, nulls, perm):
    rows = list(perm)
    for i, row in enumerate(rows):
        null = np.asarray(nulls[row["model"]], float)
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax.plot([null.min(), null.max()], [i, i], color="#D4D4D4", lw=1.1, zorder=1)
        ax.plot([lo, hi], [i, i], color=GREY, lw=4.5, solid_capstyle="butt", zorder=2)
        ax.plot(null.mean(), i, "o", ms=3.4, color=DARK, zorder=3)
        ax.plot(row["R2_observed"], i, "D", ms=5.8, color=C_BIT,
                markeredgecolor="black", markeredgewidth=0.4, zorder=4)
        ax.text(row["R2_observed"] + 0.024, i, f"$P$ = {row['p_perm_R2']:.3f}",
                fontsize=6.4, va="center", color=C_BIT)
    ax.axvline(0, color="black", lw=0.8, linestyle=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-sample $R^2$", fontsize=8)
    ax.set_xlim(-0.30, 0.40)
    ax.tick_params(axis="x", labelsize=7)
    ax.text(0.0, -0.235, "grey, null distribution over 1000 label permutations; "
                         "◆, observed value",
            transform=ax.transAxes, fontsize=6.3, color=DARK, va="top")
    label(ax, "C", "Permutation null distributions", dx=-0.26)
    style_axes(ax)


# ── D ───────────────────────────────────────────────────────────────────────
def panel_pred_actual(ax, ds, preds, lev):
    y = ds.y("bitter")
    p = np.asarray(preds["bitter"][BEST], float)
    hi = y >= 8

    jit = np.random.default_rng(0).normal(0, 0.07, len(y))
    ax.scatter(y[~hi] + jit[~hi], p[~hi], s=24, c=C_BIT, alpha=0.75,
               linewidths=0.3, edgecolors="#333333", zorder=3, label="recipes")
    ax.scatter(y[hi], p[hi], s=70, marker="*", c="#222222", zorder=5,
               label="cocoa-containing recipes")

    lim = [-0.6, 9.2]
    ax.plot(lim, lim, "k--", lw=1.0, zorder=2, label="identity")
    ax.axhline(1.0, color=GREY, linestyle=":", lw=1.1, zorder=1,
               label="constant baseline (1.0)")

    row = next(r for r in lev if r["model"] == BEST)
    ax.annotate(f"excluded: $R^2$ {row['R2_full']:.2f} → {row['R2_no_outliers']:.2f}",
                xy=(7.75, float(p[hi].max()) + 0.30), xytext=(2.55, 6.65),
                fontsize=6.5, color=DARK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=DARK))

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual bitterness", fontsize=8)
    ax.set_ylabel("Predicted bitterness", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, loc="lower right", frameon=True, framealpha=0.95,
              handlelength=1.4, borderpad=0.4, labelspacing=0.35)
    label(ax, "D", "Predicted versus actual bitterness", dx=-0.18)
    style_axes(ax)


# ── E ───────────────────────────────────────────────────────────────────────
def panel_detection(ax, ds, det):
    y = ds.y("bitter")
    X = ds.X_voigt5("bitter")
    lab = (y >= 2).astype(int)
    score = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(C=1.0, max_iter=5000,
                                                class_weight="balanced"))]).fit(X[tr], lab[tr])
        score[te] = m.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(lab, score)
    auc = float(det[det.threshold == "bitter >= 2"].LOO_ROC_AUC.iloc[0])

    ax.plot(fpr, tpr, color=C_BIT, lw=1.8, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, zorder=2)
    ax.fill_between(fpr, tpr, fpr, color=C_BIT, alpha=0.13, zorder=1)
    ax.set_xlabel("False positive rate", fontsize=8)
    ax.set_ylabel("True positive rate", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.text(0.95, 0.08, f"AUC = {auc:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=DARK)
    label(ax, "E", "Classification of bitterness ≥ 2", dx=-0.19)
    style_axes(ax)


def main():
    ds = load()
    bench = pd.read_csv(os.path.join(RESULTS, "e1_model_benchmark.csv"))
    det = pd.read_csv(os.path.join(RESULTS, "e3_detection.csv"))
    with open(os.path.join(RESULTS, "e1_loo_predictions.json")) as f:
        preds = json.load(f)
    with open(os.path.join(RESULTS, "e2_significance.json")) as f:
        sig = json.load(f)
    nulls = dict(np.load(os.path.join(RESULTS, "e2_null_distributions.npz")))

    fig = plt.figure(figsize=(13.8, 8.2))
    ax_a = fig.add_axes([0.055, 0.575, 0.205, 0.312])
    ax_c = fig.add_axes([0.355, 0.575, 0.205, 0.312])
    ax_d = fig.add_axes([0.055, 0.085, 0.205, 0.312])
    ax_e = fig.add_axes([0.355, 0.085, 0.205, 0.312])
    ax_b = fig.add_axes([0.700, 0.085, 0.165, 0.802])
    ax_r = fig.add_axes([0.900, 0.085, 0.072, 0.802])

    panel_distribution(ax_a, ds)
    panel_permutation(ax_c, nulls, sig["permutation_bitter"])
    panel_predictors(ax_b, ax_r, bench)
    panel_pred_actual(ax_d, ds, preds, sig["outlier_leverage"])
    panel_detection(ax_e, ds, det)

    fig.text(0.5, 0.960, "Bitterness analysis", ha="center", va="bottom",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.936,
             "21 predictors evaluated under a nested leave-one-out protocol "
             "($N$ = 70 recipes)",
             ha="center", va="bottom", fontsize=8.8, color="#222222")

    out = os.path.join(FIGURES, "Bitterness_analysis.png")
    fig.savefig(out, dpi=600)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[+] {out}")


if __name__ == "__main__":
    main()
