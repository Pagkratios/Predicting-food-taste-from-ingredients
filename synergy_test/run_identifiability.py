#!/usr/bin/env python3
"""
Supplementary diagnostic: is phi_syn identifiable at all, given n=70 and
the 10 features already in the hybrid model?

Run from synergy_test/ AFTER run_analysis.py:
    python3 run_identifiability.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

import core
from core import CHEM_COLS, RESULTS

UM = 'umami'
df = pd.read_pickle(os.path.join(RESULTS, "_analysis_df.pkl"))
y = df[f'{UM}_actual'].values
BASE = [f'{UM}_hs_mid', f'{UM}_voigt'] + CHEM_COLS
SYN = ['phi_syn', 'phi_syn_conc', 'phi_syn_conc2', 'phi_syn_log1p',
       'phi_syn_sqrt', 'phi_syn_broad', 'f_nucleotide', 'f_glutamate']

print("=" * 78)
print("SUPPLEMENTARY: IDENTIFIABILITY OF phi_syn")
print("=" * 78)

# ── 1. Correlation of each synergy variable with each existing feature ──
rows = []
for s in SYN:
    for b in BASE:
        r = np.corrcoef(df[s], df[b])[0, 1]
        rows.append({'synergy_var': s, 'existing_feature': b, 'pearson_r': r})
cor = pd.DataFrame(rows)
cor.to_csv(os.path.join(RESULTS, "sup_collinearity.csv"), index=False)
piv = cor.pivot(index='synergy_var', columns='existing_feature', values='pearson_r')
print("\nPearson r between synergy variables and the 10 existing features:")
print(piv.to_string(float_format=lambda x: f"{x:+.2f}"))

# ── 2. How much of phi_syn is already explained by the 10 features? ─────
print("\nR^2 of each synergy variable regressed on the 10 existing features")
print("(high R^2 => the 'new' feature is largely already in the model):")
r2rows = []
Xb = df[BASE].values.astype(float)
for s in SYN:
    lr = LinearRegression().fit(Xb, df[s].values)
    r2 = lr.score(Xb, df[s].values)
    vif = 1.0 / max(1e-12, 1.0 - r2)
    r2rows.append({'synergy_var': s, 'R2_on_existing_features': r2, 'VIF': vif})
    print(f"  {s:<18s} R^2 = {r2:.3f}   VIF = {vif:.1f}")
pd.DataFrame(r2rows).to_csv(os.path.join(RESULTS, "sup_redundancy_r2.csv"), index=False)

# ── 3. Oracle ceiling: best achievable in-sample gain from phi_syn ──────
print("\nOracle ceiling (in-sample OLS, no CV, no regularisation):")
orows = []
lr0 = LinearRegression().fit(Xb, y)
mae0 = np.mean(np.abs(lr0.predict(Xb) - y))
r20 = lr0.score(Xb, y)
print(f"  10 features           in-sample MAE={mae0:.3f}  R2={r20:.3f}")
orows.append({'model': 'baseline 10 feat', 'insample_MAE': mae0, 'insample_R2': r20,
              'delta_MAE': 0.0, 'partial_F': np.nan, 'partial_p': np.nan})
n = len(y)
for s in SYN:
    X1 = np.hstack([Xb, df[s].values.reshape(-1, 1)])
    lr1 = LinearRegression().fit(X1, y)
    mae1 = np.mean(np.abs(lr1.predict(X1) - y))
    r21 = lr1.score(X1, y)
    # partial F test for the single added term
    rss0 = np.sum((lr0.predict(Xb) - y) ** 2)
    rss1 = np.sum((lr1.predict(X1) - y) ** 2)
    dfres = n - X1.shape[1] - 1
    F = (rss0 - rss1) / (rss1 / dfres)
    pF = 1 - stats.f.cdf(F, 1, dfres)
    print(f"  + {s:<18s} in-sample MAE={mae1:.3f}  R2={r21:.3f}  "
          f"dMAE={mae1-mae0:+.3f}  partial F={F:.2f} p={pF:.3f}")
    orows.append({'model': f'+ {s}', 'insample_MAE': mae1, 'insample_R2': r21,
                  'delta_MAE': mae1 - mae0, 'partial_F': F, 'partial_p': pF})
pd.DataFrame(orows).to_csv(os.path.join(RESULTS, "sup_oracle_ceiling.csv"), index=False)

# ── 4. Effective sample size for the interaction ────────────────────────
nz = df['phi_syn'].values > 0
print(f"\nEffective sample for the interaction: {int(nz.sum())} recipes with "
      f"phi_syn>0 out of {n}.")
print(f"  phi_syn among those: min={df['phi_syn'][nz].min():.4f}, "
      f"median={np.median(df['phi_syn'][nz]):.4f}, max={df['phi_syn'][nz].max():.4f}")
print(f"  The top 3 recipes carry {df['phi_syn'].nlargest(3).sum()/df['phi_syn'].sum()*100:.0f}% "
      f"of total phi_syn mass.")

# minimum detectable effect: what coefficient on phi_syn would be needed for
# a 2-sigma OLS t-statistic, and what umami shift that implies at max phi_syn
X1 = np.hstack([Xb, df['phi_syn'].values.reshape(-1, 1)])
X1c = np.hstack([np.ones((n, 1)), X1])
beta, *_ = np.linalg.lstsq(X1c, y, rcond=None)
resid = y - X1c @ beta
s2 = resid @ resid / (n - X1c.shape[1])
cov = s2 * np.linalg.pinv(X1c.T @ X1c)
se = np.sqrt(cov[-1, -1])
mde = 1.96 * se
print(f"\n  OLS SE on the phi_syn coefficient = {se:.1f}")
print(f"  Minimum detectable coefficient (2 SE) = {mde:.1f}")
print(f"  At the largest observed phi_syn ({df['phi_syn'].max():.4f}) that is a "
      f"{mde*df['phi_syn'].max():.1f}-point umami shift on the 0-100 scale.")
print(f"  Fitted coefficient (OLS) = {beta[-1]:.1f} "
      f"-> {beta[-1]*df['phi_syn'].max():.1f}-point shift at max phi_syn.")

pd.DataFrame([{
    'n_recipes': n,
    'n_phi_syn_nonzero': int(nz.sum()),
    'phi_syn_max': float(df['phi_syn'].max()),
    'top3_share_of_phi_syn_mass': float(df['phi_syn'].nlargest(3).sum() / df['phi_syn'].sum()),
    'ols_coef_phi_syn': float(beta[-1]),
    'ols_se_phi_syn': float(se),
    'min_detectable_coef_2se': float(mde),
    'min_detectable_shift_at_max_phi_points': float(mde * df['phi_syn'].max()),
}]).to_csv(os.path.join(RESULTS, "sup_power.csv"), index=False)

print("\n[+] Supplementary diagnostic complete.")
