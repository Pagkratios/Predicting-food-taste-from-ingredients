#!/usr/bin/env python3
"""
Standalone core for the umami-synergy test.

Everything below the "VERBATIM" banner is copied unchanged from
Hybrid/src/hybrid_analysis.py (see vendor/hybrid_analysis_ORIGINAL.py for
the copy of the original file).  Deviations are listed in DEVIATIONS and
reproduced in REPORT.md.

Nothing in the parent repo is imported, read at run time, or written to,
except vendor/compute_bounds.py (a verbatim copy) for `load_attr_from_py`
and `normalize_weights`.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

from compute_bounds import load_attr_from_py, normalize_weights  # noqa: E402

RAW_PATH = os.path.join(HERE, "data", "raw_recipes.py")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

DEVIATIONS = """
1. RAW_PATH points at synergy_test/data/raw_recipes.py, a byte-identical
   copy (md5 8c3f1589a2f6b0363634383f2669837a) of data/raw_recipes.py,
   so this folder is self-contained. No content change.
2. `metrics()` additionally returns RMSE. The original returned
   (MAE, PCC, bias); RMSE was requested. MAE/PCC/bias formulas untouched.
3. `loo_evaluate()` additionally returns the per-fold Lasso coefficient
   matrix and the per-fold selected alpha, so fold-wise coefficient
   non-zero rates can be reported. The fitting itself -- StandardScaler
   fit on train fold only, LassoCV(alphas=np.logspace(-3,1,30), cv=5,
   max_iter=20000) -- is unchanged.
4. `chemistry_features()` gained an optional `extra` argument used to
   append synergy features. When `extra` is empty the returned dict is
   identical to the original.
5. Nothing else. Regex patterns, hs_bounds(), build_analysis_df(),
   the ALPHAS grid, and the feature list [t_hs_mid, t_voigt] + 8 chem
   are byte-identical to the original.
"""

# ══════════════════════════════════════════════════════════════════════
# VERBATIM from Hybrid/src/hybrid_analysis.py
# ══════════════════════════════════════════════════════════════════════

TASTES = ['sweet', 'sour', 'bitter', 'umami', 'salty']
ALPHAS = np.logspace(-3, 1, 30)

PROTEIN_PAT = r'(?i)chicken|pork|beef|cod|ham|egg|milk|cheese|yoghurt|whey'
SUGAR_PAT   = r'(?i)sugar|honey|syrup|jam|raisin'
SALT_PAT    = r'(?i)^salt$'
WATER_PAT   = r'(?i)^water'
ALLIUM_PAT  = r'(?i)onion|garlic|leek|spring onion'
FERMENT_PAT = r'(?i)soy sauce|cheese|vinegar|mustard|sauerkraut|worcestershire|tomato puree'

CHEM_COLS = ['protein_frac', 'sugar_frac', 'maillard_potential', 'salt_frac',
             'water_frac', 'conc_factor', 'allium_frac', 'fermented_frac']


def frac_match(ingredients, weights, pattern):
    total = 0.0
    for ing, w in zip(ingredients, weights):
        if re.search(pattern, ing['name']):
            total += w
    return total


def hs_bounds(T_vals, v_vals):
    T = np.maximum(np.array(T_vals, dtype=float), 0.01)
    v = np.array(v_vals, dtype=float)
    def A(T0):
        return 1.0 / np.sum(v / (T + 2*T0)) - 2*T0
    return A(np.min(T)), A(np.max(T))


# ══════════════════════════════════════════════════════════════════════
# Adapted (deviations 2-4)
# ══════════════════════════════════════════════════════════════════════

def frac_exact(ingredients, weights, name_set):
    """Mass fraction of ingredients whose name is exactly in `name_set`."""
    total = 0.0
    for ing, w in zip(ingredients, weights):
        if ing['name'] in name_set:
            total += w
    return total


def chemistry_features(recipe, nucleotide=(), glutamate=(), glutamate_broad=()):
    ings = recipe['ingredients']
    ws = normalize_weights(ings)
    w_list = ws.tolist()

    prot = frac_match(ings, w_list, PROTEIN_PAT)
    sugar = frac_match(ings, w_list, SUGAR_PAT)
    salt = frac_match(ings, w_list, SALT_PAT)
    water = frac_match(ings, w_list, WATER_PAT)
    allium = frac_match(ings, w_list, ALLIUM_PAT)
    ferment = frac_match(ings, w_list, FERMENT_PAT)
    maillard = prot * sugar
    conc = 1.0 / (1.0 - min(water, 0.9))

    out = {
        'protein_frac': prot, 'sugar_frac': sugar,
        'maillard_potential': maillard, 'salt_frac': salt,
        'water_frac': water, 'conc_factor': conc,
        'allium_frac': allium, 'fermented_frac': ferment,
    }

    # ── synergy features (additive to the dict; baseline never uses them)
    f_nuc = frac_exact(ings, w_list, set(nucleotide))
    f_glu = frac_exact(ings, w_list, set(glutamate))
    f_glu_b = frac_exact(ings, w_list, set(glutamate_broad))
    phi = f_nuc * f_glu
    out.update({
        'f_nucleotide': f_nuc,
        'f_glutamate': f_glu,
        'f_glutamate_broad': f_glu_b,
        'phi_syn': phi,
        'phi_syn_conc': phi * conc,
        'phi_syn_conc2': phi * conc ** 2,
        'phi_syn_log1p': np.log1p(phi),
        'phi_syn_sqrt': np.sqrt(phi),
        'phi_syn_broad': f_nuc * f_glu_b,
    })
    return out


def build_analysis_df(raw_recipes, nucleotide=(), glutamate=(), glutamate_broad=()):
    rows = []
    all_ing_names = set()

    for r in raw_recipes:
        for ing in r['ingredients']:
            all_ing_names.add(ing['name'])
    all_ing_names = sorted(all_ing_names)
    ing_to_idx = {n: i for i, n in enumerate(all_ing_names)}

    for r in raw_recipes:
        ings = r['ingredients']
        ws = normalize_weights(ings)
        gt = r['food_sensory_scores']
        row = {'RP': r.get('recipe_id', r['recipe_name']),
               'recipe_name': r['recipe_name']}

        skip = False
        for t in TASTES:
            T_arr = np.array([ing['sensory_scores'].get(t, 0) for ing in ings], dtype=float)
            voigt = float(np.dot(ws, T_arr))
            lo, up = hs_bounds(T_arr.tolist(), ws.tolist())
            row[f'{t}_voigt'] = voigt
            row[f'{t}_hs_lo'] = lo
            row[f'{t}_hs_up'] = up
            row[f'{t}_hs_mid'] = (lo + up) / 2.0
            val = gt.get(t)
            if val is None or np.isnan(float(val)):
                skip = True; break
            row[f'{t}_actual'] = float(val)

        if skip:
            continue

        row.update(chemistry_features(r, nucleotide, glutamate, glutamate_broad))

        ing_vec = np.zeros(len(all_ing_names))
        for ing, w in zip(ings, ws):
            ing_vec[ing_to_idx[ing['name']]] += w
        for j, name in enumerate(all_ing_names):
            row[f'v_{name}'] = ing_vec[j]

        rows.append(row)

    return pd.DataFrame(rows), all_ing_names


def loo_evaluate(X, y, alphas=ALPHAS):
    """Original LOOCV loop; additionally records per-fold coefs and alphas."""
    n = len(y)
    preds = np.zeros(n)
    coefs = np.zeros((n, X.shape[1]))
    fold_alphas = np.zeros(n)
    loo = LeaveOneOut()
    for tri, tei in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tri])
        Xte = sc.transform(X[tei])
        m = LassoCV(alphas=alphas, cv=5, max_iter=20000)
        m.fit(Xtr, y[tri])
        preds[tei] = m.predict(Xte)
        coefs[tei[0]] = m.coef_
        fold_alphas[tei[0]] = m.alpha_
    return preds, coefs, fold_alphas


def kfold_evaluate(X, y, seed, k=10, alphas=ALPHAS):
    """Same fitting recipe as loo_evaluate, but k-fold with a seed."""
    n = len(y)
    preds = np.zeros(n)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tri, tei in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tri])
        Xte = sc.transform(X[tei])
        m = LassoCV(alphas=alphas, cv=5, max_iter=20000)
        m.fit(Xtr, y[tri])
        preds[tei] = m.predict(Xte)
    return preds


def metrics(pred, actual):
    """Original returned (mae, pcc, bias); RMSE appended (deviation 2)."""
    mae = np.mean(np.abs(pred - actual))
    pcc = np.corrcoef(pred, actual)[0, 1] if np.std(pred) > 0 else 0.0
    bias = np.mean(pred - actual)
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    return mae, pcc, bias, rmse


def metrics_dict(pred, actual):
    mae, pcc, bias, rmse = metrics(pred, actual)
    return {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'Bias': bias}


def full_fit(X, y, alphas=ALPHAS):
    """Full-data fit as used for Table S3 in the original."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = LassoCV(alphas=alphas, cv=5, max_iter=20000)
    m.fit(Xs, y)
    return m


def load_recipes():
    return load_attr_from_py(RAW_PATH, "raw_recipes")
