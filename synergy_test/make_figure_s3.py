#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S3 -- nucleotide-glutamate synergy diagnostics (Supplementary S12).

Composition only: every plotted value is read from the cached artefacts in
synergy_test/results/. No model is refit and no statistic is recomputed here.

Style follows Lasso/src/plot_config.py via apply_style(), and the panel-label /
title / delta-bar conventions of Lasso/src/interaction_plots.py, which produces
the closest analogue (results/interactions/sensory_interactions.pdf).

Run from repo root or from synergy_test/:
    python3 synergy_test/make_figure_s3.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(HERE, "vendor"))

from plot_config import (                                          # noqa: E402
    FONT_SIZE_ANNOTATION, FONT_SIZE_TITLE, SENSORY_COLORS, apply_style,
    save_figure_fixed,
)

RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
# No canonical LaTeX figure directory exists in this repo (manuscript.txt uses
# flat \includegraphics{Figure N.pdf} paths and no Figure*.pdf are checked in).
# results/publication_figures/ is the closest existing home for manuscript
# figure artefacts, so the PDF is mirrored there. See the report.
LATEX_DIR = os.path.join(REPO, "results", "publication_figures")

FIG_W, FIG_H = 7.5, 7.4

# ── Palette: Tol Bright, straight from plot_config. No new colours. ──────────
C_UMAMI = SENSORY_COLORS["umami"]     # #CCBB44 -- co-present recipes / umami data
C_ACCENT = SENSORY_COLORS["bitter"]   # #EE6677 -- fitted line, observed statistic
C_BLUE = SENSORY_COLORS["sweet"]      # #4477AA -- main-effect (control) specs
C_PURPLE = SENSORY_COLORS["sour"]     # #AA3377 -- broad-class sensitivity spec
GREY = "#888888"
LIGHT = "#BBBBBB"
DARK = "#333333"

# Sizes derived from plot_config constants, matching interaction_plots.py.
SZ_LETTER = FONT_SIZE_ANNOTATION + 2.5   # 9.5
SZ_TITLE = FONT_SIZE_ANNOTATION + 0.5    # 7.5
SZ_LABEL = FONT_SIZE_ANNOTATION + 1      # 8
SZ_TICK = FONT_SIZE_ANNOTATION - 1       # 6
SZ_LEG = FONT_SIZE_ANNOTATION - 1        # 6
SZ_DATA = FONT_SIZE_ANNOTATION - 1.5     # 5.5

PHI = r"$\phi_{\mathrm{syn}}$"
F_NUC = r"$f_{\mathrm{nuc}}$"
F_GLU = r"$f_{\mathrm{glu}}$"

# Row/column geometry, in figure fractions. Panel letters are placed at a fixed
# figure x per column so they line up down the figure.
# Symmetric 2x2 grid: both columns are the same width, both rows the same
# height, so the four panel cells are identical rectangles.
ROW1 = (0.565, 0.890)     # (bottom, top)
ROW2 = (0.105, 0.430)
COL_L = (0.150, 0.510)    # (left, right)
COL_R = (0.625, 0.985)
A_BLOCK = COL_L           # panel A occupies the same cell as panel C

# Gap between a column's flush left edge and its panel letters, in points.
LETTER_PAD_PT = 6.0


def check_legend_overlap(fig):
    """Verify no legend box overlaps a drawn data element, in display coords.

    Checks scatter offsets, bar patches, and line vertices of every axes against
    that axes' legend bounding box. Prints one line per axes; raises if anything
    collides, so the figure cannot be shipped with a covered legend.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    problems = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(rend)
        hits = 0
        for coll in ax.collections:                      # scatter / hist steps
            pts = coll.get_offsets()
            if pts is not None and len(pts):
                disp = ax.transData.transform(np.asarray(pts))
                hits += int(np.sum((disp[:, 0] >= lb.x0) & (disp[:, 0] <= lb.x1) &
                                   (disp[:, 1] >= lb.y0) & (disp[:, 1] <= lb.y1)))
        for p in ax.patches:                             # bars / hist bins
            pb = p.get_window_extent(rend)
            if pb.x1 > lb.x0 and pb.x0 < lb.x1 and pb.y1 > lb.y0 and pb.y0 < lb.y1:
                hits += 1
        for ln in ax.lines:                              # fits, rules
            xy = ln.get_xydata()
            if xy is None or not len(xy):
                continue
            d = ax.transData.transform(np.asarray(xy, float))
            seg = np.vstack([d[:-1], d[1:]]) if len(d) > 1 else d
            # sample along each segment so long rules are not missed
            if len(d) > 1:
                t = np.linspace(0, 1, 200)[:, None]
                for a, b in zip(d[:-1], d[1:]):
                    s = a + t * (b - a)
                    if np.any((s[:, 0] >= lb.x0) & (s[:, 0] <= lb.x1) &
                              (s[:, 1] >= lb.y0) & (s[:, 1] <= lb.y1)):
                        hits += 1
                        break
        tag = ax.get_title() or "(inset/marginal)"
        if hits:
            problems.append(f"{tag[:44]}: {hits} element(s) under the legend")
        print(f"    legend check | {tag[:46]:<46s} {'OVERLAP' if hits else 'clear'}")
    if problems:
        raise SystemExit("[!] legend overlaps data:\n  - " + "\n  - ".join(problems))


def check_inside_canvas(fig):
    """Verify no text runs off the canvas."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.get_window_extent(rend).x1, fig.get_window_extent(rend).y1
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        b = t.get_window_extent(rend)
        if b.x0 < -1 or b.x1 > W + 1 or b.y0 < -1 or b.y1 > H + 1:
            bad.append(repr(t.get_text())[:52])
    if bad:
        raise SystemExit("[!] text outside canvas:\n  - " + "\n  - ".join(bad))
    print(f"    canvas check | all {len(fig.findobj(matplotlib.text.Text))} "
          f"text objects inside the canvas")


def _panel_fixed_x0(rend, axes_list):
    """Leftmost extent of a panel's *immovable* furniture: axes box and y ticks.

    The y-axis label is excluded because it can be repositioned; the tick labels
    cannot, so they set how far right a column's flush edge can be pushed.
    """
    xs = []
    for a in axes_list:
        xs.append(a.get_window_extent(rend).x0)
        xs += [t.get_window_extent(rend).x0
               for t in a.get_yticklabels() if t.get_text()]
    return min(xs)


def place_letters(fig, columns):
    """Flush-align each column, then set its panel letters on one x.

    Both panels in a column are made to share a left edge: whichever panel starts
    further right has its y-axis label nudged left onto the column's flush edge
    (panel C has categorical tick labels and no y label, so it usually sets the
    edge). The letters then sit LETTER_PAD_PT left of that shared edge, so A/C
    align with each other, B/D align with each other, and no letter is stranded.
    Each letter's baseline is matched to its own panel title's.

    columns: list of lists of (letter, [axes of the panel], title_axes,
             ylabel_axes or None).
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = LETTER_PAD_PT * fig.dpi / 72.0

    gap = 3.0 * fig.dpi / 72.0          # clearance between a y label and its ticks

    for col in columns:
        # Rightmost edge every panel in the column can share. For a panel with a
        # y label the edge must also leave room for the label itself, otherwise a
        # tall (multi-line) rotated label would extend right into the tick labels.
        limits = []
        for _, axes_list, _, lab_ax in col:
            lim = _panel_fixed_x0(rend, axes_list)
            if lab_ax is not None and lab_ax.yaxis.label.get_text():
                lb = lab_ax.yaxis.label.get_window_extent(rend)
                lim -= (lb.x1 - lb.x0) + gap
            limits.append(lim)
        edge = min(limits)

        # Put every y label in the column flush on that edge.
        for _, _, _, lab_ax in col:
            if lab_ax is None or not lab_ax.yaxis.label.get_text():
                continue
            lb = lab_ax.yaxis.label.get_window_extent(rend)
            cx, cy = (lb.x0 + lb.x1) / 2.0, (lb.y0 + lb.y1) / 2.0
            fx, fy = inv.transform((cx + (edge - lb.x0), cy))
            lab_ax.yaxis.set_label_coords(fx, fy, transform=fig.transFigure)

        fig.canvas.draw()
        for letter, _, title_ax, _ in col:
            y0 = title_ax.title.get_window_extent(rend).y0
            fx, fy = inv.transform((edge - pad, y0))
            fig.text(fx, fy, letter, fontsize=SZ_LETTER, weight="bold",
                     ha="left", va="bottom")


def panel_title(ax, text, pad=7):
    ax.set_title(text, fontsize=SZ_TITLE, weight="bold", pad=pad)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, linestyle="--", alpha=0.3, linewidth=0.6,
            color="#CCCCCC")
    ax.set_axisbelow(True)


# ── Panel A ──────────────────────────────────────────────────────────────────
def panel_a(fig, rec):
    """Class co-presence: scatter with marginal count distributions."""
    left, right = A_BLOCK
    bottom, top = ROW1
    outer = fig.add_gridspec(1, 1, left=left, right=right, bottom=bottom, top=top)
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0],
                                 width_ratios=[4.0, 1.0], height_ratios=[1.0, 4.0],
                                 wspace=0.07, hspace=0.07)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    fn, fg = rec["f_nucleotide"].values, rec["f_glutamate"].values
    co = (fn > 0) & (fg > 0)

    ax.scatter(fn[~co], fg[~co], s=17, c=LIGHT, edgecolors=DARK, linewidths=0.3,
               zorder=3, label=f"one class or neither ($n$ = {int((~co).sum())})")
    ax.scatter(fn[co], fg[co], s=26, c=C_UMAMI, edgecolors=DARK, linewidths=0.4,
               zorder=4, label=f"both present ($n$ = {int(co.sum())})")
    ax.set_xlabel(f"Nucleotide-class mass fraction {F_NUC}", fontsize=SZ_LABEL)
    ax.set_ylabel(f"Glutamate-class mass fraction {F_GLU}", fontsize=SZ_LABEL)
    ax.set_xlim(-0.045, 0.98)
    ax.set_ylim(-0.045, 0.95)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(labelsize=SZ_TICK, length=3)
    grid(ax)
    # Both class fractions are mass fractions of the same recipe, so their sum is
    # bounded by 1 and the upper-right quadrant is structurally empty.
    ax.legend(fontsize=SZ_LEG, loc="upper right", frameon=True, framealpha=1.0,
              edgecolor="#CCCCCC", borderpad=0.4, handletextpad=0.35,
              labelspacing=0.32)

    bins = np.linspace(0, 0.95, 24)
    ax_top.hist(fn, bins=bins, color=GREY, edgecolor="white", linewidth=0.35)
    ax_right.hist(fg, bins=bins, color=GREY, edgecolor="white", linewidth=0.35,
                  orientation="horizontal")

    ax_top.tick_params(labelbottom=False, labelsize=SZ_TICK - 1, length=2)
    ax_top.set_yticks([0, 25, 50])
    # Count label on the right of the top marginal, so it cannot collide with the
    # main panel's y-axis label.
    ax_top.yaxis.set_label_position("right")
    ax_top.set_ylabel("Recipes", fontsize=SZ_TICK, labelpad=3, rotation=270,
                      va="bottom")
    ax_right.tick_params(labelleft=False, labelsize=SZ_TICK - 1, length=2)
    ax_right.set_xticks([0, 25, 50])
    ax_right.set_xlabel("Recipes", fontsize=SZ_TICK, labelpad=2)
    for a in (ax_top, ax_right):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    panel_title(ax_top, "Class co-presence across 70 recipes", pad=7)
    return ax, ax_top, ax_right


# ── Panel B ──────────────────────────────────────────────────────────────────
def panel_b(ax, rec, diag):
    """Umami leave-one-out residual against phi_syn."""
    phi = rec["phi_syn"].values
    res = rec["umami_resid"].values
    nz = phi > 0

    ax.axhline(0, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.scatter(phi[~nz], res[~nz], s=17, c=LIGHT, edgecolors=DARK, linewidths=0.3,
               zorder=3, label=f"{PHI} = 0 ($n$ = {int((~nz).sum())})")
    ax.scatter(phi[nz], res[nz], s=26, c=C_UMAMI, edgecolors=DARK, linewidths=0.4,
               zorder=4, label=f"{PHI} > 0 ($n$ = {int(nz.sum())})")

    row = diag[diag["subset"].str.startswith("all recipes")].iloc[0]
    xs = np.array([-0.002, 0.070])
    ax.plot(xs, row["ols_intercept"] + row["ols_slope"] * xs, color=C_ACCENT,
            lw=1.3, zorder=5, label="OLS fit, 70 recipes")

    ax.set_xlim(-0.004, 0.072)
    ax.set_ylim(-17.5, 28.0)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06])
    ax.set_xlabel(f"Synergy term {PHI} = {F_NUC} $\\times$ {F_GLU}",
                  fontsize=SZ_LABEL)
    ax.set_ylabel("Umami leave-one-out residual\n(actual $-$ predicted, 0–100 scale)",
                  fontsize=SZ_LABEL)
    ax.tick_params(labelsize=SZ_TICK, length=3)
    grid(ax)
    # No recipe combines a large phi_syn with a large positive residual, so the
    # upper-right corner is free.
    ax.legend(fontsize=SZ_LEG, loc="upper right", frameon=True, framealpha=1.0,
              edgecolor="#CCCCCC", borderpad=0.4, handletextpad=0.35,
              labelspacing=0.32)
    panel_title(ax, "Umami residual against the synergy term")


# ── Panel C ──────────────────────────────────────────────────────────────────
# Compact forms: phi is phi_syn and v is v_water throughout this panel, as
# spelled out by the axis labels of panels A/B and by the caption.
SPEC_LABEL = {
    "a. + phi_syn": r"a  $\phi$",
    "b. + phi_syn x conc": r"b  $\phi/(1-v)$",
    "c. + phi_syn x conc^2": r"c  $\phi/(1-v)^{2}$",
    "d. + log1p(phi_syn)": r"d  $\log(1+\phi)$",
    "e. + sqrt(phi_syn)": r"e  $\sqrt{\phi}$",
    "f. + main effects only (control)": r"f  $f_{\mathrm{nuc}},f_{\mathrm{glu}}$",
    "g. + main effects + phi_syn (extra)":
        r"g  $f_{\mathrm{nuc}},f_{\mathrm{glu}},\phi$",
    "h. + phi_syn broad classes (extra)": r"h  $\phi$, broad",
}
SPEC_GROUP = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 1, "g": 1, "h": 2}
GROUP_COLOR = {0: C_UMAMI, 1: C_BLUE, 2: C_PURPLE}
GROUP_LABEL = {0: "product term", 1: "class main effects",
               2: "product, broad classes"}


def panel_c(ax, var):
    """Change in umami leave-one-out MAE relative to baseline, per specification."""
    v = var[var["variant"] != "baseline (10 feat)"].reset_index(drop=True)
    deltas = v["delta_MAE_vs_baseline"].values
    keys = [s.split(".")[0] for s in v["variant"]]
    colors = [GROUP_COLOR[SPEC_GROUP[k]] for k in keys]
    ypos = np.arange(len(v))

    ax.barh(ypos, deltas, color=colors, edgecolor=DARK, linewidth=0.5, height=0.65,
            zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.9, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SPEC_LABEL[s] for s in v["variant"]], fontsize=SZ_TICK + 0.3)
    ax.invert_yaxis()

    span = float(deltas.max())
    ax.set_xlim(-span * 0.52, span * 1.34)
    for y, d in zip(ypos, deltas):
        ax.text(d + span * 0.04, y, f"{d:+.3f}", va="center", ha="left",
                fontsize=SZ_DATA, color=DARK, zorder=5)

    ax.set_xlabel("$\\Delta$ umami leave-one-out MAE vs baseline\n"
                  "(points on the 0–100 scale)", fontsize=SZ_LABEL)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15])
    ax.tick_params(axis="x", labelsize=SZ_TICK, length=3)
    ax.tick_params(axis="y", length=0)
    grid(ax, axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLOR[g], ec=DARK, lw=0.5)
               for g in (0, 1, 2)]
    # Placed above the axes: every row of this panel carries a bar, so there is no
    # in-axes region that is reliably free of data.
    ax.legend(handles, [GROUP_LABEL[g] for g in (0, 1, 2)], fontsize=SZ_LEG,
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              frameon=False, handlelength=1.0, borderpad=0.3, labelspacing=0.3,
              columnspacing=1.0, handletextpad=0.45)
    panel_title(ax, "Effect of each specification on umami MAE", pad=26)


# ── Panel D ──────────────────────────────────────────────────────────────────
def panel_d(ax, null_imp, perm_row, kf):
    """Permutation null for specification (a), plus paired 10-fold differences."""
    # The cached arrays store improvement (baseline - variant). Negate so that
    # "positive = worse" matches panel C's sign convention throughout the figure.
    null_d = -np.asarray(null_imp, float)
    obs_d = -float(perm_row["true_improvement_points"])
    n_perm = int(perm_row["n_perm"])

    counts, _, _ = ax.hist(null_d, bins=40, color=LIGHT, edgecolor="white",
                           linewidth=0.35, zorder=2,
                           label=f"shuffled ($n$ = {n_perm})")
    ax.axvline(0, color=GREY, lw=0.8, ls="--", zorder=3)
    ax.axvline(obs_d, color=C_ACCENT, lw=1.5, zorder=5, label="observed")
    ax.set_xlabel("$\\Delta$ umami leave-one-out MAE vs baseline\n"
                  "(points on the 0–100 scale)", fontsize=SZ_LABEL)
    ax.set_ylabel("Permutations", fontsize=SZ_LABEL)
    ax.set_xlim(-0.42, 0.74)
    ax.set_ylim(0, counts.max() * 1.34)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax.tick_params(labelsize=SZ_TICK, length=3)
    grid(ax, axis="y")
    # Left tail: bin counts below x = -0.15 never exceed ~20 of a 1.34x headroom
    # axis, so the upper-left corner is clear of every bar and of both rules.
    ax.legend(fontsize=SZ_LEG, loc="upper left", frameon=True, framealpha=1.0,
              edgecolor="#CCCCCC", handlelength=1.1, borderpad=0.4,
              labelspacing=0.32, handletextpad=0.45)
    # pad matches panel C so both row-2 titles sit on one baseline.
    panel_title(ax, "Permutation null and 10-fold differences", pad=26)

    # Inset: paired 10-fold differences for specification (a), 5 seeds. Placed
    # over the sparse right tail of the null.
    ins = ax.inset_axes([0.615, 0.475, 0.365, 0.335])
    base = kf[kf["variant"] == "baseline (10 feat)"].iloc[0]
    row = kf[kf["variant"] == "a. + phi_syn"].iloc[0]
    seed_cols = [c for c in kf.columns if c.startswith("seed")]
    per_seed = np.array([row[c] - base[c] for c in seed_cols], float)

    ins.axvline(0, color=GREY, lw=0.7, ls="--", zorder=1)
    ins.scatter(per_seed, np.arange(len(per_seed)), s=15, c=C_UMAMI,
                edgecolors=DARK, linewidths=0.4, zorder=3)
    ins.set_yticks(np.arange(len(per_seed)))
    ins.set_yticklabels([f"s{i}" for i in range(len(per_seed))],
                        fontsize=SZ_TICK - 1.5)
    ins.set_xlim(-0.022, 0.115)
    ins.set_ylim(-0.7, len(per_seed) - 0.3)
    ins.set_xticks([0.00, 0.05, 0.10])
    ins.tick_params(labelsize=SZ_TICK - 1.5, length=2, pad=1.5)
    ins.set_xlabel("paired $\\Delta$ MAE, 10-fold (5 seeds)",
                   fontsize=SZ_TICK - 0.5, labelpad=1.5)
    ins.grid(True, axis="x", linestyle="--", alpha=0.3, linewidth=0.5,
             color="#CCCCCC")
    ins.set_axisbelow(True)
    ins.set_facecolor("white")
    for s in ("top", "right"):
        ins.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ins.spines[s].set_linewidth(0.6)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    apply_style()
    # constrained_layout is on globally in plot_config; this figure uses explicit
    # gridspec placement instead, exactly as interaction_plots.py does.
    plt.rcParams["figure.constrained_layout.use"] = False

    rec = pd.read_csv(os.path.join(RESULTS, "step2_all_recipes.csv"))
    diag = pd.read_csv(os.path.join(RESULTS, "step3_residual_diagnostic.csv"))
    var = pd.read_csv(os.path.join(RESULTS, "step5_umami_variants.csv"))
    perm = pd.read_csv(os.path.join(RESULTS, "step6_permutation_test.csv"))
    kf = pd.read_csv(os.path.join(RESULTS, "step6_kfold_robustness.csv"))
    null_a = np.load(os.path.join(RESULTS, "_null_improvement_phi_syn.npy"))
    perm_a = perm[perm["feature"] == "phi_syn"].iloc[0]

    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=False)

    ax_b = fig.add_axes([COL_R[0], ROW1[0], COL_R[1] - COL_R[0], ROW1[1] - ROW1[0]])
    ax_c = fig.add_axes([COL_L[0], ROW2[0], COL_L[1] - COL_L[0], ROW2[1] - ROW2[0]])
    ax_d = fig.add_axes([COL_R[0], ROW2[0], COL_R[1] - COL_R[0], ROW2[1] - ROW2[0]])

    ax_a, ax_a_top, ax_a_right = panel_a(fig, rec)
    panel_b(ax_b, rec, diag)
    panel_c(ax_c, var)
    panel_d(ax_d, null_a, perm_a, kf)

    place_letters(fig, [
        [("A", [ax_a, ax_a_top, ax_a_right], ax_a_top, ax_a),
         ("C", [ax_c], ax_c, None)],
        [("B", [ax_b], ax_b, ax_b),
         ("D", [ax_d], ax_d, ax_d)],
    ])

    fig.suptitle("Nucleotide–Glutamate Synergy Diagnostics",
                 fontsize=FONT_SIZE_TITLE, weight="bold", y=0.958)

    print("[*] Verifying layout before saving:")
    check_legend_overlap(fig)
    check_inside_canvas(fig)

    os.makedirs(FIGURES, exist_ok=True)
    pdf = os.path.join(FIGURES, "Figure_S3.pdf")
    png = os.path.join(FIGURES, "Figure_S3.png")
    fig.savefig(png, dpi=600, facecolor="white")
    save_figure_fixed(fig, pdf)

    os.makedirs(LATEX_DIR, exist_ok=True)
    shutil.copy2(pdf, os.path.join(LATEX_DIR, "Figure_S3.pdf"))
    print(f"[+] Saved: {png}")
    print(f"[+] Copied: {os.path.join(LATEX_DIR, 'Figure_S3.pdf')}")


if __name__ == "__main__":
    main()
