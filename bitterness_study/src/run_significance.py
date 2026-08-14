#!/usr/bin/env python3
"""E2 — is the bitterness signal real, and is it worth anything?

Three questions the reviewer's comment raises, answered separately:

  1. Overfitting  — permutation test. The whole LOO pipeline is re-run on
     shuffled bitterness labels. If the observed out-of-sample R^2 sits inside
     that null distribution, the model is fitting noise.
  2. Precision    — bootstrap CIs on R^2 and MAE over recipes.
  3. Practical    — paired bootstrap of MAE against the constant baseline, plus
     a leave-the-outliers-out refit.

    python3 bitterness_study/src/run_significance.py [--n-perm 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))

from data import RESULTS, load                                    # noqa: E402
from evaluate import (bootstrap_metric_ci, loo_predict, mae_of,   # noqa: E402
                      metrics, paired_bootstrap_dmae, r2_of)
from models import ConstantMean, OrdinalPO, build_models          # noqa: E402

# Models cheap enough to permute 1000x, spanning the families that matter.
PERM_MODELS = ["Lasso 5D", "Ridge", "Ordinal (prop. odds)", "log1p-transform Ridge"]


def _perm_once(factory, X, y, seed):
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)
    p = loo_predict(factory, X, yp)
    return r2_of(p, yp), mae_of(p, yp)


def permutation_test(name, factory, X, y, n_perm, n_jobs=-1):
    obs = loo_predict(factory, X, y)
    r2_obs, mae_obs = r2_of(obs, y), mae_of(obs, y)

    null = Parallel(n_jobs=n_jobs)(
        delayed(_perm_once)(factory, X, y, s) for s in range(n_perm))
    null_r2 = np.array([a for a, _ in null])
    null_mae = np.array([b for _, b in null])

    # +1 corrections keep the p-value from ever being exactly 0.
    p_r2 = (np.sum(null_r2 >= r2_obs) + 1) / (n_perm + 1)
    p_mae = (np.sum(null_mae <= mae_obs) + 1) / (n_perm + 1)
    return {
        "model": name,
        "R2_observed": float(r2_obs),
        "R2_null_mean": float(null_r2.mean()),
        "R2_null_p95": float(np.percentile(null_r2, 95)),
        "p_perm_R2": float(p_r2),
        "MAE_observed": float(mae_obs),
        "MAE_null_mean": float(null_mae.mean()),
        "MAE_null_p05": float(np.percentile(null_mae, 5)),
        "p_perm_MAE": float(p_mae),
        "n_perm": n_perm,
    }, null_r2, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    ds = load()
    zoo = build_models()
    y = ds.y("bitter")
    X = ds.X_voigt5("bitter")
    ref = loo_predict(ConstantMean, X, y)
    out = {}

    # ── 1. permutation tests on bitterness ────────────────────────────────
    print(f"[*] Permutation test, {args.n_perm} shuffles per model "
          f"(full LOO pipeline re-run each time)")
    perm_rows, nulls, obs_preds = [], {}, {}
    for name in PERM_MODELS:
        factory = zoo[name][0]
        row, null_r2, obs = permutation_test(name, factory, X, y, args.n_perm)
        perm_rows.append(row)
        nulls[name] = null_r2.tolist()
        obs_preds[name] = obs
        print(f"    {name:<24s} R2_obs={row['R2_observed']:+.3f}  "
              f"null mean={row['R2_null_mean']:+.3f} (95th pct {row['R2_null_p95']:+.3f})  "
              f"p={row['p_perm_R2']:.4f}   |   MAE p={row['p_perm_MAE']:.4f}")
    out["permutation_bitter"] = perm_rows

    # ── 1b. same test on the other four dimensions (protocol control) ─────
    print(f"\n[*] Protocol control — same permutation test, Ridge, other tastes")
    ctrl = []
    for t in ["sweet", "sour", "umami", "salty"]:
        row, _, _ = permutation_test(f"Ridge/{t}", zoo["Ridge"][0],
                                     ds.X_voigt5(t), ds.y(t), args.n_perm)
        ctrl.append(row)
        print(f"    {t:<8s} R2_obs={row['R2_observed']:+.3f}  p={row['p_perm_R2']:.4f}")
    out["permutation_control"] = ctrl

    # ── 2. bootstrap CIs for the best-R2 model on bitterness ──────────────
    print("\n[*] Bootstrap CIs (10,000 resamples over recipes)")
    boot = []
    for name in PERM_MODELS:
        p = obs_preds[name]
        r2 = bootstrap_metric_ci(p, y, r2_of)
        mae = bootstrap_metric_ci(p, y, mae_of)
        boot.append({"model": name,
                     "R2": r2["point"], "R2_lo": r2["ci_lo"], "R2_hi": r2["ci_hi"],
                     "MAE": mae["point"], "MAE_lo": mae["ci_lo"], "MAE_hi": mae["ci_hi"]})
        print(f"    {name:<24s} R2 = {r2['point']:+.3f} "
              f"[{r2['ci_lo']:+.3f}, {r2['ci_hi']:+.3f}]   "
              f"MAE = {mae['point']:.3f} [{mae['ci_lo']:.3f}, {mae['ci_hi']:.3f}]")
    out["bootstrap_bitter"] = boot

    # ── 3. paired comparison against the constant baselines ───────────────
    print("\n[*] Paired bootstrap of MAE difference vs the constant baselines")
    med = np.full(len(y), np.nan)
    for i in range(len(y)):                       # LOO constant-median
        med[i] = np.median(np.delete(y, i))
    paired = []
    for name in PERM_MODELS:
        for base_name, base in [("constant mean", ref), ("constant median", med)]:
            d = paired_bootstrap_dmae(obs_preds[name], base, y)
            d.update(model=name, baseline=base_name)
            paired.append(d)
            verdict = ("model better" if d["ci_hi"] < 0 else
                       "baseline better" if d["ci_lo"] > 0 else "indistinguishable")
            print(f"    {name:<24s} vs {base_name:<16s} "
                  f"dMAE={d['dMAE']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  "
                  f"p={d['p_boot']:.3f}  -> {verdict}")
    out["paired_vs_constant"] = paired

    # ── 4. how much of the signal is the two outlier recipes? ─────────────
    print("\n[*] Outlier leverage — refit with the bitter=8 recipes removed")
    keep = y < 8
    lev = []
    for name in PERM_MODELS:
        p_sub = loo_predict(zoo[name][0], X[keep], y[keep])
        ref_sub = loo_predict(ConstantMean, X[keep], y[keep])
        m_full = metrics(obs_preds[name], y, ref)
        m_sub = metrics(p_sub, y[keep], ref_sub)
        lev.append({"model": name,
                    "R2_full": m_full["R2_oos"], "R2_no_outliers": m_sub["R2_oos"],
                    "MAE_full": m_full["MAE"], "MAE_no_outliers": m_sub["MAE"],
                    "Skill_MSE_full": m_full["Skill_MSE"],
                    "Skill_MSE_no_outliers": m_sub["Skill_MSE"]})
        print(f"    {name:<24s} R2 {m_full['R2_oos']:+.3f} -> {m_sub['R2_oos']:+.3f}   "
              f"skill {m_full['Skill_MSE']:+.3f} -> {m_sub['Skill_MSE']:+.3f}  "
              f"(n={int(keep.sum())})")
    out["outlier_leverage"] = lev

    with open(os.path.join(RESULTS, "e2_significance.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RESULTS, "e2_null_distributions.npz"),
                        **{k: np.asarray(v) for k, v in nulls.items()})
    for key in ["permutation_bitter", "permutation_control", "bootstrap_bitter",
                "paired_vs_constant", "outlier_leverage"]:
        pd.DataFrame(out[key]).to_csv(os.path.join(RESULTS, f"e2_{key}.csv"), index=False)
    print(f"\n[+] {os.path.join(RESULTS, 'e2_significance.json')}")


if __name__ == "__main__":
    main()
