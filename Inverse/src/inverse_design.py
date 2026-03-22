#!/usr/bin/env python3
"""
Inverse design via Differential Evolution — three case studies from the paper.

Run from repo root:
    python3 Inverse/src/inverse_design.py
"""
import importlib.util, os, sys, json, warnings
import numpy as np
from scipy.optimize import differential_evolution
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "HS_RV", "src"))
from compute_bounds import load_attr_from_py, normalize_weights

RAW_PATH = os.path.join(REPO, "data", "raw_recipes.py")
OUT_DIR = os.path.join(REPO, "results", "inverse")
os.makedirs(OUT_DIR, exist_ok=True)

TASTES = ['sweet', 'sour', 'bitter', 'umami', 'salty']

def hs_mid(T_vals, v_vals):
    T = np.maximum(np.array(T_vals, dtype=float), 0.01)
    v = np.array(v_vals, dtype=float)
    def A(T0): return 1.0/np.sum(v/(T+2*T0)) - 2*T0
    return (A(np.min(T)) + A(np.max(T))) / 2

def run_case(raw_recipes, rp_id, target_changes, weights, bounds_dict, case_name):
    recipe = [r for r in raw_recipes if r.get('recipe_id') == rp_id][0]
    ings = recipe['ingredients']
    n = len(ings)
    ws = normalize_weights(ings)
    names = [ing['name'] for ing in ings]

    # Taste score matrix (n_ing x 5)
    T_mat = np.array([[ing['sensory_scores'].get(t, 0) for t in TASTES] for ing in ings], dtype=float)
    v_orig = ws.copy()

    # Original HS profile
    orig = np.array([hs_mid(T_mat[:, j], v_orig) for j in range(5)])
    target = orig.copy()
    for j, delta in target_changes.items():
        target[j] = orig[j] + delta
    W = np.array(weights)

    # Bounds: default ±50%, override with bounds_dict
    bds = []
    for i in range(n):
        nm = names[i]
        lo = max(0.001, v_orig[i] * 0.5)
        hi = min(1.0, v_orig[i] * 1.5)
        for pat, (cl, ch) in bounds_dict.items():
            if pat.lower() in nm.lower():
                lo, hi = max(0.001, cl), ch
        bds.append((lo, hi))

    def objective(v):
        vn = v / v.sum()
        p = np.array([hs_mid(T_mat[:, j], vn) for j in range(5)])
        return np.sum(W * np.abs(target - p) / np.maximum(np.abs(p), 0.01))

    result = differential_evolution(objective, bds, maxiter=1000, popsize=20,
                                     seed=42, tol=1e-8, mutation=(0.5, 1.0),
                                     recombination=0.8, strategy='best1bin')
    v_opt = result.x / result.x.sum()
    opt_profile = np.array([hs_mid(T_mat[:, j], v_opt) for j in range(5)])

    print(f"\n{'=' * 65}")
    print(f"CASE: {case_name}")
    print(f"Recipe: {recipe['recipe_name']} ({rp_id}, {n} ingredients)")
    print(f"{'=' * 65}")

    print(f"\n{'Ingredient':<35} {'Orig':>7} {'Opt':>7} {'Chg':>7}")
    print("-" * 58)
    for i in range(n):
        ch = v_opt[i] - v_orig[i]
        if abs(ch) > 0.003 or v_orig[i] > 0.03:
            print(f"{names[i][:34]:<35} {v_orig[i]:>7.3f} {v_opt[i]:>7.3f} {ch:>+7.3f}")

    print(f"\n{'Taste':<10} {'Orig':>7} {'Target':>7} {'Opt':>7}")
    print("-" * 36)
    for j, t in enumerate(TASTES):
        print(f"{t:<10} {orig[j]:>7.2f} {target[j]:>7.2f} {opt_profile[j]:>7.2f}")

    print(f"\nFractions sum: orig={v_orig.sum():.3f}, opt={v_opt.sum():.3f}")

    return {
        'case': case_name, 'recipe_id': rp_id,
        'ingredients': [{'name': names[i], 'orig': float(v_orig[i]),
                         'opt': float(v_opt[i])} for i in range(n)],
        'profile_orig': {t: float(orig[j]) for j, t in enumerate(TASTES)},
        'profile_target': {t: float(target[j]) for j, t in enumerate(TASTES)},
        'profile_opt': {t: float(opt_profile[j]) for j, t in enumerate(TASTES)},
    }


def main():
    raw_recipes = load_attr_from_py(RAW_PATH, "raw_recipes")
    results = []

    # Case 1: Pea Soup — salt reduction
    results.append(run_case(raw_recipes, 'RP14',
        target_changes={4: -0.8},  # reduce salt
        weights=[1, 1, 1, 2, 3],
        bounds_dict={'Salt': (0.001, 0.005), 'Water': (0.2, 0.5)},
        case_name='Pea soup — salt reduction'))

    # Case 2: Chocolate spread — sugar reduction (penalty via tight bounds)
    results.append(run_case(raw_recipes, 'RP55',
        target_changes={},
        weights=[3, 1, 1, 1, 1],
        bounds_dict={'Sugar': (0.001, 0.35)},
        case_name='Chocolate spread — sugar reduction'))

    # Case 3: Ketchup — umami boost
    results.append(run_case(raw_recipes, 'RP68',
        target_changes={3: +0.8, 0: -0.5},  # more umami, less sweet
        weights=[2, 1, 1, 3, 1],
        bounds_dict={'Sugar': (0.001, 0.10), 'Tomato': (0.45, 0.80)},
        case_name='Ketchup — umami boost'))

    with open(os.path.join(OUT_DIR, "inverse_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {OUT_DIR}/inverse_results.json")


if __name__ == "__main__":
    main()
