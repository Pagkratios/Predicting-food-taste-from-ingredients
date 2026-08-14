#!/usr/bin/env python3
"""
Steps 0-5 of the umami-synergy test.

Run from synergy_test/:
    python3 run_analysis.py

Writes CSVs to results/ and PNGs to figures/. Touches nothing outside
synergy_test/.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

import core
from core import TASTES, CHEM_COLS, RESULTS, FIGURES
from ingredient_classes import (NUCLEOTIDE, GLUTAMATE, GLUTAMATE_BROAD,
                                AMBIGUOUS_NOTES, validate)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(core.HERE, "vendor"))
from plot_config import apply_style, setup_figure, save_figure, style_axes, get_sensory_color

apply_style()
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

UM = 'umami'
BASE_FEATS = lambda t: [f'{t}_hs_mid', f'{t}_voigt'] + CHEM_COLS


def sec(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ══════════════════════════════════════════════════════════════════════
# STEP 0/1: load, build, validate classes
# ══════════════════════════════════════════════════════════════════════
sec("STEP 0-1: DATASET AND INGREDIENT CLASSES")

raw = core.load_recipes()
df, all_names = core.build_analysis_df(raw, NUCLEOTIDE, GLUTAMATE, GLUTAMATE_BROAD)
validate(all_names)

n_instances = sum(len(r['ingredients']) for r in raw)
print(f"Recipes retained: {len(df)} / {len(raw)}")
print(f"Unique ingredient names: {len(all_names)}")
print(f"Total ingredient instances across recipes: {n_instances}")
print(f"NUCLEOTIDE class: {len(NUCLEOTIDE)} ingredients")
print(f"GLUTAMATE class (primary): {len(GLUTAMATE)} ingredients")
print(f"GLUTAMATE class (broad):   {len(GLUTAMATE_BROAD)} ingredients")

# full ingredient inventory with per-recipe mass fractions
v_cols = [c for c in df.columns if c.startswith('v_')]
inv = []
for c in v_cols:
    name = c[2:]
    vals = df[c].values
    nz = vals[vals > 0]
    inv.append({
        'ingredient': name,
        'n_recipes_used': int((vals > 0).sum()),
        'mean_mass_frac_when_used': float(nz.mean()) if len(nz) else 0.0,
        'min_mass_frac_when_used': float(nz.min()) if len(nz) else 0.0,
        'max_mass_frac_when_used': float(nz.max()) if len(nz) else 0.0,
        'nucleotide_class': int(name in NUCLEOTIDE),
        'glutamate_class_primary': int(name in GLUTAMATE),
        'glutamate_class_broad': int(name in GLUTAMATE_BROAD),
    })
pd.DataFrame(inv).to_csv(os.path.join(RESULTS, "ingredient_inventory.csv"), index=False)

# full recipe x ingredient mass fraction matrix
mat = df[['RP', 'recipe_name'] + v_cols].copy()
mat.columns = ['RP', 'recipe_name'] + [c[2:] for c in v_cols]
mat.to_csv(os.path.join(RESULTS, "recipe_ingredient_mass_fractions.csv"), index=False)
print(f"Wrote ingredient_inventory.csv ({len(inv)} rows) and "
      f"recipe_ingredient_mass_fractions.csv ({len(mat)}x{len(v_cols)})")

pd.DataFrame([{'ingredient': n, 'class': 'NUCLEOTIDE'} for n in NUCLEOTIDE] +
             [{'ingredient': n, 'class': 'GLUTAMATE_PRIMARY'} for n in GLUTAMATE] +
             [{'ingredient': n, 'class': 'GLUTAMATE_BROAD_ONLY'}
              for n in GLUTAMATE_BROAD if n not in GLUTAMATE]
             ).to_csv(os.path.join(RESULTS, "ingredient_classes.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════
# STEP 2: coverage / identifiability
# ══════════════════════════════════════════════════════════════════════
sec("STEP 2: CO-PRESENCE AND COVERAGE")

df['has_nuc'] = df['f_nucleotide'] > 0
df['has_glu'] = df['f_glutamate'] > 0
df['copresent'] = df['has_nuc'] & df['has_glu']

n = len(df)
n_nuc = int(df['has_nuc'].sum())
n_glu = int(df['has_glu'].sum())
n_co = int(df['copresent'].sum())
print(f"Recipes with >=1 NUCLEOTIDE ingredient: {n_nuc}/{n} ({n_nuc/n*100:.0f}%)")
print(f"Recipes with >=1 GLUTAMATE ingredient:  {n_glu}/{n} ({n_glu/n*100:.0f}%)")
print(f"Recipes with BOTH (phi_syn > 0):        {n_co}/{n} ({n_co/n*100:.0f}%)")
print(f"Recipes with phi_syn == 0:              {n-n_co}/{n}")


def describe(v, label):
    d = {'variable': label, 'n': len(v), 'n_nonzero': int((v > 0).sum()),
         'mean': v.mean(), 'sd': v.std(ddof=1), 'min': v.min(),
         'p25': np.percentile(v, 25), 'median': np.median(v),
         'p75': np.percentile(v, 75), 'max': v.max()}
    return d


dist_rows = [describe(df['f_nucleotide'].values, 'f_nucleotide'),
             describe(df['f_glutamate'].values, 'f_glutamate'),
             describe(df['phi_syn'].values, 'phi_syn')]
dist = pd.DataFrame(dist_rows)
dist.to_csv(os.path.join(RESULTS, "step2_class_fraction_distributions.csv"), index=False)
print("\n" + dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# baseline umami LOOCV predictions & residuals (needed for the table below)
y_um = df[f'{UM}_actual'].values
X_um_base = df[BASE_FEATS(UM)].values.astype(float)
pred_base, coef_base, alpha_base = core.loo_evaluate(X_um_base, y_um)
df['umami_pred_baseline'] = pred_base
df['umami_resid'] = y_um - pred_base

co = df[df['copresent']].copy()
co_tbl = co[['RP', 'recipe_name', 'f_nucleotide', 'f_glutamate', 'phi_syn',
             'umami_actual', 'umami_pred_baseline', 'umami_resid']].sort_values(
    'phi_syn', ascending=False)
co_tbl.to_csv(os.path.join(RESULTS, "step2_copresent_recipes.csv"), index=False)
print(f"\nCo-present recipes (n={n_co}), sorted by phi_syn:")
print(co_tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# full per-recipe table too
df[['RP', 'recipe_name', 'f_nucleotide', 'f_glutamate', 'phi_syn', 'conc_factor',
    'umami_actual', 'umami_pred_baseline', 'umami_resid']].to_csv(
    os.path.join(RESULTS, "step2_all_recipes.csv"), index=False)

# Figure: distributions + co-presence scatter
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
axes[0].hist(df['f_nucleotide'], bins=20, color="#4477AA", edgecolor='white')
axes[0].set_xlabel(r'$f_{\mathrm{nucleotide}}$'); axes[0].set_ylabel('recipes')
axes[1].hist(df['f_glutamate'], bins=20, color="#228833", edgecolor='white')
axes[1].set_xlabel(r'$f_{\mathrm{glutamate}}$')
axes[2].scatter(df['f_nucleotide'], df['f_glutamate'], s=18,
                c=np.where(df['copresent'], "#EE6677", "#BBBBBB"),
                edgecolors='k', linewidths=0.3)
axes[2].set_xlabel(r'$f_{\mathrm{nucleotide}}$')
axes[2].set_ylabel(r'$f_{\mathrm{glutamate}}$')
axes[2].set_title(f'co-present n={n_co}/{n}', fontsize=8)
for a in axes:
    style_axes(a)
fig.tight_layout()
save_figure(fig, os.path.join(FIGURES, "fig1_class_coverage.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: residual diagnostic
# ══════════════════════════════════════════════════════════════════════
sec("STEP 3: RESIDUAL DIAGNOSTIC (umami residual ~ phi_syn)")

resid = df['umami_resid'].values
phi = df['phi_syn'].values

diag_rows = []
for label, mask in [('all recipes (n=%d)' % n, np.ones(n, bool)),
                    ('co-present only (n=%d)' % n_co, df['copresent'].values)]:
    r_, p_ = resid[mask], phi[mask]
    ls = stats.linregress(p_, r_)
    if len(np.unique(p_)) > 1:
        sp_rho, sp_p = stats.spearmanr(p_, r_)
    else:
        sp_rho, sp_p = np.nan, np.nan
    diag_rows.append({
        'subset': label, 'n': int(mask.sum()),
        'ols_slope': ls.slope, 'ols_intercept': ls.intercept,
        'ols_stderr': ls.stderr, 'ols_p': ls.pvalue, 'ols_r2': ls.rvalue ** 2,
        'pearson_r': ls.rvalue,
        'spearman_rho': sp_rho, 'spearman_p': sp_p,
    })
    print(f"\n{label}:")
    print(f"  OLS  slope={ls.slope:+.3f} (SE {ls.stderr:.3f})  p={ls.pvalue:.4f}  "
          f"R2={ls.rvalue**2:.4f}")
    print(f"  Spearman rho={sp_rho:+.3f}  p={sp_p:.4f}")

diag = pd.DataFrame(diag_rows)
diag.to_csv(os.path.join(RESULTS, "step3_residual_diagnostic.csv"), index=False)

# also the same diagnostic for the two main effects, for context
main_rows = []
for var in ['f_nucleotide', 'f_glutamate', 'phi_syn_conc', 'phi_syn_conc2',
            'phi_syn_log1p', 'phi_syn_sqrt', 'phi_syn_broad']:
    x = df[var].values
    ls = stats.linregress(x, resid)
    rho, rp = stats.spearmanr(x, resid)
    main_rows.append({'variable': var, 'ols_slope': ls.slope, 'ols_p': ls.pvalue,
                      'ols_r2': ls.rvalue ** 2, 'spearman_rho': rho,
                      'spearman_p': rp})
pd.DataFrame(main_rows).to_csv(
    os.path.join(RESULTS, "step3_residual_diagnostic_other_vars.csv"), index=False)
print("\nSame diagnostic for related variables (all 70 recipes):")
print(pd.DataFrame(main_rows).to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}"))

# Figure: residual vs phi_syn
fig, ax = setup_figure(size="single")
ax.axhline(0, color='#888', lw=0.8, ls='--')
zero = ~df['copresent'].values
ax.scatter(phi[zero], resid[zero], s=20, c='#BBBBBB', edgecolors='k',
           linewidths=0.3, label=f'$\\phi_{{syn}}=0$ (n={int(zero.sum())})')
ax.scatter(phi[~zero], resid[~zero], s=26, c=get_sensory_color('umami'),
           edgecolors='k', linewidths=0.3, label=f'co-present (n={n_co})')
ls = stats.linregress(phi, resid)
xs = np.linspace(0, phi.max() * 1.05, 50)
ax.plot(xs, ls.intercept + ls.slope * xs, color='#EE6677', lw=1.2,
        label=f'OLS: slope={ls.slope:+.1f}, p={ls.pvalue:.3f}, $R^2$={ls.rvalue**2:.3f}')
ax.set_xlabel(r'$\phi_{\mathrm{syn}} = f_{\mathrm{nucleotide}}\times f_{\mathrm{glutamate}}$')
ax.set_ylabel('umami LOOCV residual\n(actual $-$ predicted)')
ax.legend(fontsize=6, frameon=False, loc='upper right')
style_axes(ax)
fig.tight_layout()
save_figure(fig, os.path.join(FIGURES, "fig2_residual_vs_phisyn.png"))
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# STEP 4/5: variants
# ══════════════════════════════════════════════════════════════════════
sec("STEP 4-5: MODEL VARIANTS")

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

rows = []
store = {}
for label, extra in VARIANTS:
    feats = BASE_FEATS(UM) + extra
    X = df[feats].values.astype(float)
    pred, coefs, falphas = core.loo_evaluate(X, y_um)
    m = core.metrics_dict(pred, y_um)
    full = core.full_fit(X, y_um)
    cmap = dict(zip(feats, full.coef_))

    extra_coef = {e: cmap[e] for e in extra}
    nonzero_rate = {}
    for e in extra:
        j = feats.index(e)
        nonzero_rate[e] = float(np.mean(np.abs(coefs[:, j]) > 1e-10))

    rows.append({
        'variant': label,
        'n_features': len(feats),
        'MAE': m['MAE'], 'RMSE': m['RMSE'], 'PCC': m['PCC'], 'Bias': m['Bias'],
        'alpha_full_fit': full.alpha_,
        'alpha_loo_median': float(np.median(falphas)),
        'alpha_loo_min': float(falphas.min()),
        'alpha_loo_max': float(falphas.max()),
        'new_feature_coefs_full_fit': json.dumps(
            {k: round(float(v), 4) for k, v in extra_coef.items()}),
        'new_feature_nonzero_fold_rate': json.dumps(
            {k: round(v, 3) for k, v in nonzero_rate.items()}),
    })
    store[label] = {'pred': pred, 'coefs': coefs, 'feats': feats,
                    'full_coef': cmap, 'alpha': full.alpha_}
    print(f"{label:<38s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
          f"PCC={m['PCC']:.3f} Bias={m['Bias']:+.3f} alpha={full.alpha_:.4f}")
    if extra:
        print(f"{'':38s}   coef(full fit)={ {k: round(float(v),4) for k,v in extra_coef.items()} }  "
              f"nonzero in LOO folds={ {k: f'{v*100:.0f}%' for k,v in nonzero_rate.items()} }")

var_tbl = pd.DataFrame(rows)
b_mae = var_tbl.loc[0, 'MAE']
var_tbl['delta_MAE_vs_baseline'] = var_tbl['MAE'] - b_mae
var_tbl['pct_MAE_change'] = (var_tbl['MAE'] - b_mae) / b_mae * 100
var_tbl.to_csv(os.path.join(RESULTS, "step5_umami_variants.csv"), index=False)

# coefficient comparison baseline vs variant a
comp = []
fa = store['a. + phi_syn']
for f in BASE_FEATS(UM):
    c0 = store['baseline (10 feat)']['full_coef'][f]
    c1 = fa['full_coef'][f]
    comp.append({'feature': f, 'coef_baseline': c0, 'coef_with_phi_syn': c1,
                 'abs_change': abs(c1 - c0),
                 'sign_flip': int(np.sign(c0) != np.sign(c1) and
                                  (abs(c0) > 1e-10 or abs(c1) > 1e-10))})
comp.append({'feature': 'phi_syn', 'coef_baseline': np.nan,
             'coef_with_phi_syn': fa['full_coef']['phi_syn'],
             'abs_change': np.nan, 'sign_flip': 0})
comp_df = pd.DataFrame(comp)
comp_df.to_csv(os.path.join(RESULTS, "step4_umami_coefficient_comparison.csv"), index=False)
print("\nUmami coefficients (full-data fit), baseline vs + phi_syn:")
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# ── other four dimensions, with and without phi_syn ────────────────────
print("\nAll five dimensions, baseline vs + phi_syn:")
dim_rows = []
for t in TASTES:
    y = df[f'{t}_actual'].values
    for label, extra in [('baseline', []), ('+phi_syn', ['phi_syn'])]:
        feats = BASE_FEATS(t) + extra
        X = df[feats].values.astype(float)
        pred, coefs, falphas = core.loo_evaluate(X, y)
        m = core.metrics_dict(pred, y)
        full = core.full_fit(X, y)
        nz = (float(np.mean(np.abs(coefs[:, feats.index('phi_syn')]) > 1e-10))
              if extra else np.nan)
        dim_rows.append({
            'taste': t, 'model': label, 'n_features': len(feats),
            'MAE': m['MAE'], 'RMSE': m['RMSE'], 'PCC': m['PCC'], 'Bias': m['Bias'],
            'alpha_full_fit': full.alpha_,
            'phi_syn_coef_full_fit': (float(full.coef_[feats.index('phi_syn')])
                                      if extra else np.nan),
            'phi_syn_nonzero_fold_rate': nz,
        })
        print(f"  {t:<7s} {label:<10s} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
              f"PCC={m['PCC']:.3f} Bias={m['Bias']:+.3f}")

dim_tbl = pd.DataFrame(dim_rows)
dim_tbl.to_csv(os.path.join(RESULTS, "step4_all_dimensions.csv"), index=False)

# Figure: variant MAE bars (left: true zero-based scale; right: zoomed)
labels = var_tbl['variant'].tolist()
short = [l.replace(' (extra)', '') for l in labels]
maes = var_tbl['MAE'].values
cols = ['#666666'] + ['#4477AA'] * 5 + ['#CCBB44', '#CCBB44', '#AA3377']

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
for a, zoom in zip(axs, [False, True]):
    a.bar(range(len(labels)), maes, color=cols, edgecolor='k', linewidth=0.4)
    a.axhline(b_mae, color='#EE6677', ls='--', lw=1.0,
              label=f'baseline MAE={b_mae:.3f}')
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(short, rotation=40, ha='right', fontsize=5.5)
    a.set_ylabel('umami LOOCV MAE')
    if zoom:
        a.set_ylim(5.6, 6.0)
        a.set_title('zoomed (note truncated axis)', fontsize=7)
    else:
        a.set_ylim(0, max(maes) * 1.15)
        a.set_title('true scale', fontsize=7)
    a.legend(fontsize=6, frameon=False, loc='lower right')
    style_axes(a)
fig.tight_layout()
save_figure(fig, os.path.join(FIGURES, "fig3_variant_mae.png"))
plt.close(fig)

# Figure: predicted vs actual, baseline vs +phi_syn
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharex=True, sharey=True)
for a, key in zip(axes, ['baseline (10 feat)', 'a. + phi_syn']):
    p = store[key]['pred']
    m = core.metrics_dict(p, y_um)
    a.scatter(y_um, p, s=20, c=get_sensory_color('umami'), edgecolors='k',
              linewidths=0.3)
    lim = [0, max(y_um.max(), p.max()) * 1.08]
    a.plot(lim, lim, ls='--', c='#888', lw=0.8)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel('actual umami')
    a.set_title(f"{key}\nMAE={m['MAE']:.2f}  PCC={m['PCC']:.3f}", fontsize=7)
    style_axes(a)
axes[0].set_ylabel('LOOCV predicted umami')
fig.tight_layout()
save_figure(fig, os.path.join(FIGURES, "fig4_umami_pred_vs_actual.png"))
plt.close(fig)

# persist artefacts needed by run_robustness.py
np.save(os.path.join(RESULTS, "_umami_baseline_pred.npy"), pred_base)
df.to_pickle(os.path.join(RESULTS, "_analysis_df.pkl"))

print("\n[+] Steps 0-5 complete. CSVs in results/, PNGs in figures/.")
