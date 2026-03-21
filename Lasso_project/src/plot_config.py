#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plotting configuration for publication-quality figures.

All plots in the project should import styling from this module to ensure
consistency across figures. Uses the Tol Bright palette (colorblind-safe).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ────────────────── Shared constants ──────────────────

SENSORY_ORDER = ["sweet", "bitter", "salty", "umami", "sour"]

ALIASES = {
    "sweetness": "sweet", "bitterness": "bitter", "sourness": "sour",
    "umaminess": "umami", "saltiness": "salty",
    "sweet": "sweet", "bitter": "bitter", "sour": "sour",
    "umami": "umami", "salty": "salty",
}

METHOD_ORDER = ["HS", "RV", "Lasso"]

METHOD_DISPLAY = {
    "HS":    "Hashin\u2013Shtrikman (HS)",
    "RV":    "Reuss\u2013Voigt (RV)",
    "Lasso": "Lasso",
}

# ────────────────── Tol Bright palette (colorblind-safe) ──────────────────

SENSORY_COLORS = {
    "sweet":  "#4477AA",
    "bitter": "#EE6677",
    "salty":  "#228833",
    "umami":  "#CCBB44",
    "sour":   "#AA3377",
}

METHOD_COLORS = {
    "HS":    "#4477AA",
    "RV":    "#228833",
    "Lasso": "#EE6677",
}

METHOD_COLORS_LIGHT = {
    "HS":    "#A4C4DD",
    "RV":    "#91C4A0",
    "Lasso": "#F5B3BB",
}

# ────────────────── DPI & figure sizes ──────────────────

SAVE_DPI = 600
DISPLAY_DPI = 150

# Sizes in inches (width, height); journal-standard column widths
FIG_SINGLE_COL     = (3.5, 3.0)
FIG_SINGLE_SQUARE  = (3.5, 3.5)
FIG_ONE_AND_HALF   = (5.5, 4.5)
FIG_DOUBLE_COL     = (7.0, 5.5)
FIG_DOUBLE_COL_WIDE = (7.0, 3.5)
FIG_FULL_PAGE      = (7.0, 9.0)

# ────────────────── Font sizes ──────────────────

FONT_FAMILY         = "serif"
FONT_SIZE_BASE      = 9
FONT_SIZE_LABEL     = 10
FONT_SIZE_TITLE     = 11
FONT_SIZE_LEGEND    = 8
FONT_SIZE_ANNOTATION = 7

# ────────────────── rcParams ──────────────────

RCPARAMS = {
    "font.family":        FONT_FAMILY,
    "font.size":          FONT_SIZE_BASE,
    "axes.titlesize":     FONT_SIZE_TITLE,
    "axes.labelsize":     FONT_SIZE_LABEL,
    "xtick.labelsize":    FONT_SIZE_BASE,
    "ytick.labelsize":    FONT_SIZE_BASE,
    "legend.fontsize":    FONT_SIZE_LEGEND,
    "figure.dpi":         DISPLAY_DPI,
    "savefig.dpi":        SAVE_DPI,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "#333333",
    "axes.labelcolor":    "#333333",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.axisbelow":     True,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "lines.linewidth":    1.5,
    "grid.linestyle":     "--",
    "grid.alpha":         0.3,
    "grid.linewidth":     0.6,
    "grid.color":         "#CCCCCC",
    "mathtext.fontset":   "cm",
    "figure.constrained_layout.use": True,
}


# ────────────────── Helper functions ──────────────────

def apply_style(use_seaborn=False):
    """Apply the publication style globally. Call once at module load."""
    mpl.rcParams.update(RCPARAMS)
    if use_seaborn:
        import seaborn as sns
        sns.set_theme(style="whitegrid", rc=RCPARAMS)


_SIZE_MAP = {
    "single":      FIG_SINGLE_COL,
    "single_sq":   FIG_SINGLE_SQUARE,
    "onehalf":     FIG_ONE_AND_HALF,
    "double":      FIG_DOUBLE_COL,
    "double_wide": FIG_DOUBLE_COL_WIDE,
    "full":        FIG_FULL_PAGE,
}


def setup_figure(size="double", aspect=None):
    """Create a figure with standardized sizing.

    Args:
        size: preset name or (width, height) tuple in inches.
        aspect: optional width/height ratio override.

    Returns:
        (fig, ax) tuple.
    """
    figsize = _SIZE_MAP[size] if isinstance(size, str) else size
    if aspect is not None:
        figsize = (figsize[0], figsize[0] / aspect)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, path, dpi=None, transparent=False):
    """Save figure with consistent settings."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi or SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=transparent,
        facecolor="white" if not transparent else "none",
    )
    plt.close(fig)
    print(f"[+] Saved: {path}")


def style_axes(ax, grid=True):
    """Apply consistent axes styling."""
    ax.tick_params(direction="out", width=0.8, length=4)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6, color="#CCCCCC")
    ax.margins(x=0.05, y=0.08)


def get_sensory_color(key):
    """Return the standard color for a sensory attribute."""
    return SENSORY_COLORS[key]


def get_method_color(method, light=False):
    """Return the standard color for a prediction method."""
    return METHOD_COLORS_LIGHT[method] if light else METHOD_COLORS[method]


def get_sensory_palette():
    """Return list of colors in SENSORY_ORDER."""
    return [SENSORY_COLORS[k] for k in SENSORY_ORDER]


def sensory_legend_handles(marker="o", markersize=5):
    """Create legend handles for the five sensory attributes."""
    return [
        plt.Line2D([0], [0], marker=marker, linestyle="",
                   markersize=markersize, label=k.capitalize(),
                   color=SENSORY_COLORS[k])
        for k in SENSORY_ORDER
    ]
