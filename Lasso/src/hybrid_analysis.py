#!/usr/bin/env python3
"""
Hybrid model, bound-coverage analysis, and per-ingredient Lasso baseline.
Produces Tables 1 and 2 from the paper.

Run from repo root:
    python3 Hybrid/src/hybrid_analysis.py
"""
import importlib.util, os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "HS_RV", "src"))

from compute_bounds import load_attr_from_py, build_dataset_records, normalize_weights

RAW_PATH = os.path.join(REPO, "data", "raw_recipes.py")
OUT_DIR = os.path.join(REPO, "results", "hybrid")
COV_DIR = os.path.join(REPO, "results", "coverage")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(COV_DIR, exist_ok=True)

TASTES = ['sweet', 'sour', 'bitter', 'umami', 'salty']
ALPHAS = np.logspace(-3, 1, 30)

# ─── Chemistry-proxy feature extraction ───────────────────────────────

PROTEIN_PAT = r'(?i)chicken|pork|beef|cod|ham|egg|milk|cheese|yoghurt|whey'
SUGAR_PAT   = r'(?i)sugar|honey|syrup|jam|raisin'
SALT_PAT    = r'(?i)^salt$'
WATER_PAT   = r'(?i)^water'
ALLIUM_PAT  = r'(?i)onion|garlic|leek|spring onion'
FERMENT_PAT = r'(?i)soy sauce|cheese|vinegar|mustard|sauerkraut|worcestershire|tomato puree'

import re

def frac_match(ingredients, weights, pattern):
    total = 0.0
    for ing, w in zip(ingredients, weights):
        if re.search(pattern, ing['name']):
            total += w
    return total

def chemistry_features(recipe):
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
    conc = 1.0 / (1.0 - water) if water < 0.9 else 1.0

    return {
        'protein_frac': prot, 'sugar_frac': sugar,
        'maillard_potential': maillard, 'salt_frac': salt,
        'water_frac': water, 'conc_factor': conc,
        'allium_frac': allium, 'fermented_frac': ferment,
    }


# ─── HS bounds (using paper's auxiliary function form) ────────────────

def hs_bounds(T_vals, v_vals):
    T = np.maximum(np.array(T_vals, dtype=float), 0.01)
    v = np.array(v_vals, dtype=float)
    def A(T0):
        return 1.0 / np.sum(v / (T + 2*T0)) - 2*T0
    return A(np.min(T)), A(np.max(T))


# ─── Build recipe-level dataset ──────────────────────────────────────

def build_analysis_df(raw_recipes):
    rows = []
    all_ing_names = set()

    # First pass: collect all ingredient names for per-ingredient features
    for r in raw_recipes:
        for ing in r['ingredients']:
            all_ing_names.add(ing['name'])
    all_ing_names = sorted(all_ing_names)
    ing_to_idx = {n: i for i, n in enumerate(all_ing_names)}

    for r in raw_recipes:
        ings = r['ingredients']
        ws = normalize_weights(ings)
        gt = r['food_sensory_scores']
        row = {'RP': r.get('recipe_id', r['recipe_name'])}

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

        chem = chemistry_features(r)
        row.update(chem)

        # Per-ingredient mass fractions (for 111-feature Lasso)
        ing_vec = np.zeros(len(all_ing_names))
        for ing, w in zip(ings, ws):
            ing_vec[ing_to_idx[ing['name']]] += w
        for j, name in enumerate(all_ing_names):
            row[f'v_{name}'] = ing_vec[j]

        rows.append(row)

    return pd.DataFrame(rows), all_ing_names


# ─── LOO evaluation ──────────────────────────────────────────────────

def loo_evaluate(X, y, alphas=ALPHAS):
    n = len(y)
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for tri, tei in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tri])
        Xte = sc.transform(X[tei])
        m = LassoCV(alphas=alphas, cv=5, max_iter=20000)
        m.fit(Xtr, y[tri])
        preds[tei] = m.predict(Xte)
    return preds

def metrics(pred, actual):
    mae = np.mean(np.abs(pred - actual))
    pcc = np.corrcoef(pred, actual)[0, 1] if np.std(pred) > 0 else 0.0
    bias = np.mean(pred - actual)
    return mae, pcc, bias


# ─── Main analysis ───────────────────────────────────────────────────

def main():
    print("[*] Loading data...")
    raw_recipes = load_attr_from_py(RAW_PATH, "raw_recipes")
    df, all_ing_names = build_analysis_df(raw_recipes)
    print(f"    Recipes: {len(df)}, Ingredients: {len(all_ing_names)}")

    chem_cols = ['protein_frac','sugar_frac','maillard_potential','salt_frac',
                 'water_frac','conc_factor','allium_frac','fermented_frac']
    ing_v_cols = [c for c in df.columns if c.startswith('v_')]

    # ═══════════════════════════════════════════════════════════════════
    # TABLE 1: Bound coverage + HS vs Hybrid performance
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TABLE 1: BOUND COVERAGE AND MODEL PERFORMANCE")
    print("=" * 80)

    table1_rows = []
    hybrid_preds_all = {}

    for t in TASTES:
        actual = df[f'{t}_actual'].values
        hs_mid = df[f'{t}_hs_mid'].values
        hs_lo = df[f'{t}_hs_lo'].values
        hs_up = df[f'{t}_hs_up'].values
        voigt = df[f'{t}_voigt'].values

        # Bound coverage
        below = np.sum(actual < hs_lo)
        within = np.sum((actual >= hs_lo) & (actual <= hs_up))
        above = np.sum(actual > hs_up)
        n = len(actual)

        # HS midpoint metrics
        m_hs = metrics(hs_mid, actual)

        # Hybrid model: HS_mid + Voigt + 8 chemistry features
        feat_hybrid = [f'{t}_hs_mid', f'{t}_voigt'] + chem_cols
        X_hyb = df[feat_hybrid].values.astype(float)
        p_hyb = loo_evaluate(X_hyb, actual)
        m_hyb = metrics(p_hyb, actual)
        hybrid_preds_all[t] = p_hyb

        table1_rows.append({
            'Taste': t.capitalize() if t != 'salty' else 'Salt',
            'GT Mean': np.mean(actual),
            'GT SD': np.std(actual),
            'Below': f"{below/n*100:.0f}%",
            'Within': f"{within/n*100:.0f}%",
            'Above': f"{above/n*100:.0f}%",
            'HS MAE': m_hs[0],
            'HS PCC': m_hs[1],
            'HS Bias': m_hs[2],
            'Hyb MAE': m_hyb[0],
            'Hyb PCC': m_hyb[1],
            'Hyb Bias': m_hyb[2],
        })

        print(f"\n  {t.upper()}")
        print(f"    Ground truth: mean={np.mean(actual):.1f}, sd={np.std(actual):.1f}")
        print(f"    Coverage: below={below}({below/n*100:.0f}%), within={within}({within/n*100:.0f}%), above={above}({above/n*100:.0f}%)")
        print(f"    HS midpoint: MAE={m_hs[0]:.1f}, PCC={m_hs[1]:.3f}, Bias={m_hs[2]:.1f}")
        print(f"    Hybrid:      MAE={m_hyb[0]:.1f}, PCC={m_hyb[1]:.3f}, Bias={m_hyb[2]:.1f}")

    # Overall (excluding bitter)
    meaningful = [t for t in TASTES if t != 'bitter']
    hs_maes = [r['HS MAE'] for r in table1_rows if r['Taste'] != 'Bitter']
    hs_pccs = [r['HS PCC'] for r in table1_rows if r['Taste'] != 'Bitter']
    hyb_maes = [r['Hyb MAE'] for r in table1_rows if r['Taste'] != 'Bitter']
    hyb_pccs = [r['Hyb PCC'] for r in table1_rows if r['Taste'] != 'Bitter']
    hyb_biases = [r['Hyb Bias'] for r in table1_rows if r['Taste'] != 'Bitter']

    print(f"\n  AVG (4D, excl. bitter):")
    print(f"    HS:     MAE={np.mean(hs_maes):.1f}, PCC={np.mean(hs_pccs):.2f}")
    print(f"    Hybrid: MAE={np.mean(hyb_maes):.1f}, PCC={np.mean(hyb_pccs):.2f}, Bias={np.mean(hyb_biases):.1f}")

    # Save Table 1
    pd.DataFrame(table1_rows).to_csv(os.path.join(COV_DIR, "table1_coverage.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # TABLE 2: Full model comparison
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 2: MODEL COMPARISON (avg over sweet, sour, umami, salty)")
    print("=" * 80)

    model_results = {}

    for model_name, get_X in [
        ('HS midpoint', None),
        ('Voigt', None),
        ('Lasso 5D', lambda t: df[[f'{tt}_voigt' for tt in TASTES]].values),
        ('Hybrid HS+chem', lambda t: df[[f'{t}_hs_mid', f'{t}_voigt'] + chem_cols].values),
        ('Lasso per-ingr', lambda t: df[ing_v_cols].values),
    ]:
        maes, pccs = [], []
        for t in TASTES:
            actual = df[f'{t}_actual'].values

            if model_name == 'HS midpoint':
                pred = df[f'{t}_hs_mid'].values
            elif model_name == 'Voigt':
                pred = df[f'{t}_voigt'].values
            elif model_name == 'Hybrid HS+chem':
                pred = hybrid_preds_all[t]  # reuse from Table 1
            else:
                X = get_X(t)
                pred = loo_evaluate(X, actual)

            m = metrics(pred, actual)
            if t != 'bitter':
                maes.append(m[0])
                pccs.append(m[1])

        avg_mae = np.mean(maes)
        avg_pcc = np.mean(pccs)
        model_results[model_name] = (avg_mae, avg_pcc)
        n_feat = {'HS midpoint': 0, 'Voigt': 0, 'Lasso 5D': 5,
                  'Hybrid HS+chem': 10, 'Lasso per-ingr': len(ing_v_cols)}[model_name]
        print(f"  {model_name:<22s}  feat={n_feat:<4d}  MAE={avg_mae:.1f}  PCC={avg_pcc:.2f}")

    # Save Table 2
    t2_rows = [{'Model': k, 'Avg MAE': v[0], 'Avg PCC': v[1]} for k, v in model_results.items()]
    pd.DataFrame(t2_rows).to_csv(os.path.join(OUT_DIR, "table2_models.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # Hybrid model coefficients (Table S3)
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE S3: HYBRID MODEL LASSO COEFFICIENTS")
    print("=" * 80)

    coef_rows = []
    for t in TASTES:
        actual = df[f'{t}_actual'].values
        feat_names = [f'{t}_hs_mid', f'{t}_voigt'] + chem_cols
        X = df[feat_names].values.astype(float)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = LassoCV(alphas=ALPHAS, cv=5, max_iter=20000)
        m.fit(Xs, actual)

        print(f"\n  {t.upper()} (intercept={m.intercept_:.1f}, alpha={m.alpha_:.4f}):")
        for fn, c in sorted(zip(feat_names, m.coef_), key=lambda x: abs(x[1]), reverse=True):
            if abs(c) > 0.01:
                print(f"    {fn:<30s} {c:>+8.2f}")
                coef_rows.append({'taste': t, 'feature': fn, 'coefficient': round(c, 2)})

    pd.DataFrame(coef_rows).to_csv(os.path.join(OUT_DIR, "table_s3_coefficients.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════════
    # Bitterness floor-effect analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("BITTERNESS FLOOR EFFECT")
    print("=" * 80)
    bitter_actual = df['bitter_actual'].values
    print(f"  Mean={np.mean(bitter_actual):.1f}, SD={np.std(bitter_actual):.1f}")
    print(f"  ≤3: {np.sum(bitter_actual<=3)}/{len(bitter_actual)} ({np.sum(bitter_actual<=3)/len(bitter_actual)*100:.0f}%)")
    print(f"  ≤5: {np.sum(bitter_actual<=5)}/{len(bitter_actual)} ({np.sum(bitter_actual<=5)/len(bitter_actual)*100:.0f}%)")
    const_pred = np.full_like(bitter_actual, 2.0)
    print(f"  Constant=2.0 MAE: {np.mean(np.abs(bitter_actual - const_pred)):.2f}")
    print(f"  HS midpoint MAE:  {np.mean(np.abs(bitter_actual - df['bitter_hs_mid'].values)):.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # HS midpoint vs Voigt correlation
    # ═══════════════════════════════════════════════════════════════════
    all_hs = np.concatenate([df[f'{t}_hs_mid'].values for t in TASTES])
    all_vgt = np.concatenate([df[f'{t}_voigt'].values for t in TASTES])
    print(f"\n  HS midpoint vs Voigt correlation: r={np.corrcoef(all_hs, all_vgt)[0,1]:.3f}")

    print("\n[+] All analysis complete. Results saved to results/hybrid/ and results/coverage/")


if __name__ == "__main__":
    main()
