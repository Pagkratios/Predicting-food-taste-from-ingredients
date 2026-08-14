#!/usr/bin/env python3
"""
Step 6 of the umami-synergy test: repeated 10-fold CV and a permutation test.

Run from synergy_test/ AFTER run_analysis.py:
    python3 run_robustness.py
"""
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import core
from core import TASTES, CHEM_COLS, RESULTS, FIGURES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(core.HERE, "vendor"))
from plot_config import apply_style, setup_figure, save_figure, style_axes

apply_style()

N_PERM = 1000
SEEDS = [0, 1, 2, 3, 4]
K = 10
UM = 'umami'

df = pd.read_pickle(os.path.join(RESULTS, "_analysis_df.pkl"))
y = df[f'{UM}_actual'].values
BASE = [f'{UM}_hs_mid', f'{UM}_voigt'] + CHEM_COLS

VARIANTS = [
    ('baseline (10 feat)', []),
    ('a. + phi_syn', ['phi_syn']),
    ('b. + phi_syn x conc', ['phi_syn_conc']),
    ('c. + phi_syn x conc^2', ['phi_syn_conc2']),
    ('d. + log1p(phi_syn)', ['phi_syn_log1p']),
    ('e. + sqrt(phi_syn)', ['phi_syn_sqrt']),
    ('f. + main effects only (control)', ['f_nucleotide', 'f_glutamate']),
    ('g. + main effects + phi_syn (extra)', ['f_nucleotide', 'f_glutamate', 'phi_syn']),
    ('h. + phi_syn broad classes (extra)', ['phi_syn_broad']),
]

# ══════════════════════════════════════════════════════════════════════
# 6a. Repeated 10-fold CV, 5 seeds
# ══════════════════════════════════════════════════════════════════════
print("=" * 78)
print(f"STEP 6a: {K}-FOLD CV OVER {len(SEEDS)} SEEDS (umami)")
print("=" * 78)

rows = []
per_seed = {}
for label, extra in VARIANTS:
    X = df[BASE + extra].values.astype(float)
    maes = []
    for s in SEEDS:
        p = core.kfold_evaluate(X, y, seed=s, k=K)
        maes.append(core.metrics(p, y)[0])
    maes = np.array(maes)
    per_seed[label] = maes
    rows.append({'variant': label, 'kfold_MAE_mean': maes.mean(),
                 'kfold_MAE_sd': maes.std(ddof=1),
                 **{f'seed{s}_MAE': m for s, m in zip(SEEDS, maes)}})
    print(f"  {label:<38s} MAE = {maes.mean():.3f} +/- {maes.std(ddof=1):.3f}")

kf = pd.DataFrame(rows)
base_mean = kf.loc[0, 'kfold_MAE_mean']
kf['delta_vs_baseline'] = kf['kfold_MAE_mean'] - base_mean
# paired across seeds
kf['paired_mean_delta'] = [float(np.mean(per_seed[l] - per_seed['baseline (10 feat)']))
                           for l in kf['variant']]
kf['paired_sd_delta'] = [float(np.std(per_seed[l] - per_seed['baseline (10 feat)'], ddof=1))
                         if l != 'baseline (10 feat)' else 0.0 for l in kf['variant']]
kf.to_csv(os.path.join(RESULTS, "step6_kfold_robustness.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════
# 6b. Permutation test
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print(f"STEP 6b: PERMUTATION TEST ({N_PERM} shuffles)")
print("=" * 78)

base_pred, _, _ = core.loo_evaluate(df[BASE].values.astype(float), y)
base_mae = core.metrics(base_pred, y)[0]
print(f"  Baseline LOOCV MAE = {base_mae:.4f}")

rng_master = np.random.default_rng(20260727)
perm_idx = [rng_master.permutation(len(y)) for _ in range(N_PERM)]


def perm_mae(col, idx):
    X = df[BASE].values.astype(float)
    v = df[col].values[idx].reshape(-1, 1)
    p, _, _ = core.loo_evaluate(np.hstack([X, v]), y)
    return core.metrics(p, y)[0]


perm_rows = []
for col, vlabel in [('phi_syn', 'a. + phi_syn'),
                    ('phi_syn_conc2', 'c. + phi_syn x conc^2')]:
    Xtrue = df[BASE + [col]].values.astype(float)
    ptrue, _, _ = core.loo_evaluate(Xtrue, y)
    true_mae = core.metrics(ptrue, y)[0]
    true_imp = base_mae - true_mae          # >0 means the feature HELPS

    print(f"\n  [{vlabel}] true MAE={true_mae:.4f}, "
          f"improvement={true_imp:+.4f} points; running {N_PERM} shuffles...")
    null_maes = np.array(Parallel(n_jobs=-1, verbose=0)(
        delayed(perm_mae)(col, idx) for idx in perm_idx))
    null_imp = base_mae - null_maes

    pct = float((null_imp < true_imp).mean() * 100)
    p_one = float(((null_imp >= true_imp).sum() + 1) / (N_PERM + 1))
    print(f"    null improvement: mean={null_imp.mean():+.4f}, "
          f"sd={null_imp.std(ddof=1):.4f}, "
          f"[p5,p95]=[{np.percentile(null_imp,5):+.4f},{np.percentile(null_imp,95):+.4f}]")
    print(f"    true improvement sits at the {pct:.1f}th percentile of the null; "
          f"one-sided p = {p_one:.4f}")

    perm_rows.append({
        'variant': vlabel, 'feature': col,
        'baseline_MAE': base_mae, 'true_MAE': true_mae,
        'true_improvement_points': true_imp,
        'true_improvement_pct': true_imp / base_mae * 100,
        'null_mean_improvement': float(null_imp.mean()),
        'null_sd_improvement': float(null_imp.std(ddof=1)),
        'null_p05': float(np.percentile(null_imp, 5)),
        'null_p50': float(np.percentile(null_imp, 50)),
        'null_p95': float(np.percentile(null_imp, 95)),
        'true_percentile_in_null': pct,
        'one_sided_p': p_one,
        'n_perm': N_PERM,
    })
    np.save(os.path.join(RESULTS, f"_null_improvement_{col}.npy"), null_imp)

pd.DataFrame(perm_rows).to_csv(
    os.path.join(RESULTS, "step6_permutation_test.csv"), index=False)

# Figure: permutation nulls
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
for a, r in zip(axes, perm_rows):
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
plt.close(fig)

# Figure: k-fold mean +/- sd
fig, ax = setup_figure(size="onehalf")
labels = kf['variant'].tolist()
ax.errorbar(range(len(labels)), kf['kfold_MAE_mean'], yerr=kf['kfold_MAE_sd'],
            fmt='o', ms=4, capsize=3, color='#4477AA', ecolor='#888', lw=1)
ax.axhline(base_mean, color='#EE6677', ls='--', lw=1.0,
           label=f'baseline mean = {base_mean:.3f}')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.replace(' (extra)', '') for l in labels], rotation=35,
                   ha='right', fontsize=6)
ax.set_ylabel(f'umami {K}-fold CV MAE (mean $\\pm$ SD, {len(SEEDS)} seeds)')
ax.legend(fontsize=6, frameon=False)
style_axes(ax)
fig.tight_layout()
save_figure(fig, os.path.join(FIGURES, "fig6_kfold_robustness.png"))
plt.close(fig)

print("\n[+] Step 6 complete.")
