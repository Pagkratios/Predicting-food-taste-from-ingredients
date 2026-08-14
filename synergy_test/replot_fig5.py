#!/usr/bin/env python3
"""Redraw fig5 from the cached permutation nulls (avoids a 15-min rerun)."""
import os
import sys
import numpy as np
import pandas as pd
import core
from core import RESULTS, FIGURES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(core.HERE, "vendor"))
from plot_config import apply_style, save_figure, style_axes

apply_style()
perm = pd.read_csv(os.path.join(RESULTS, "step6_permutation_test.csv"))
N_PERM = int(perm['n_perm'].iloc[0])

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
for a, (_, r) in zip(axes, perm.iterrows()):
    null_imp = np.load(os.path.join(RESULTS, f"_null_improvement_{r['feature']}.npy"))
    a.hist(null_imp, bins=40, color='#BBBBBB', edgecolor='white')
    a.axvline(r['true_improvement_points'], color='#EE6677', lw=1.4,
              label=f"true = {r['true_improvement_points']:+.3f}\n"
                    f"({r['true_percentile_in_null']:.0f}th pct)")
    a.axvline(0, color='#888', ls='--', lw=0.8)
    a.set_title(r['variant'], fontsize=7)
    a.legend(fontsize=6, frameon=False)
    style_axes(a)
axes[0].set_ylabel(f'permutations (n={N_PERM})')
fig.supxlabel('MAE improvement (points on the 0-100 scale)', fontsize=8)
fig.tight_layout()
save_figure(fig, os.path.join(FIGURES, "fig5_permutation_null.png"))
