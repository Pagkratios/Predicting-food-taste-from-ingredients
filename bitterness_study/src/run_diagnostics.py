#!/usr/bin/env python3
"""E3 — why bitterness behaves the way it does, and what the ceiling is.

Characterises the response itself rather than any model: dynamic range, level
structure, where the variance actually lives, what one panel quantisation step is
worth relative to the best model gain, the statistical power available, and
whether a coarser (detection) framing survives where regression does not.

    python3 bitterness_study/src/run_diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from data import RESULTS, TASTES, load                # noqa: E402
from evaluate import loo_predict                      # noqa: E402
from models import ConstantMean, build_models         # noqa: E402


def describe_targets(ds):
    rows = []
    for t in TASTES:
        y = ds.y(t)
        vals, cnts = np.unique(y, return_counts=True)
        p = cnts / cnts.sum()
        mode_share = cnts.max() / cnts.sum()
        rows.append({
            "taste": t, "n": len(y), "mean": y.mean(), "sd": y.std(ddof=1),
            "min": y.min(), "max": y.max(),
            "distinct_levels": len(vals),
            "modal_value": float(vals[cnts.argmax()]), "modal_share": mode_share,
            "frac_le_3": float(np.mean(y <= 3)),
            "sd_as_pct_of_scale": y.std(ddof=1) / 100.0,
            "shannon_entropy_bits": float(-np.sum(p * np.log2(p))),
            "IQR": float(np.percentile(y, 75) - np.percentile(y, 25)),
        })
    return pd.DataFrame(rows)


def variance_concentration(y):
    """How much of the total sum of squares a handful of recipes carry."""
    ybar = y.mean()
    ss = (y - ybar) ** 2
    sst = ss.sum()
    order = np.argsort(-ss)
    out = {}
    for k in [1, 2, 3, 5, 10]:
        out[f"top{k}_recipes_share_of_SST"] = float(ss[order[:k]].sum() / sst)
    out["SST"] = float(sst)
    out["n_above_modal"] = int(np.sum(y > stats.mode(y, keepdims=False).mode))
    return out


def quantisation_context(y, best_rmse, const_rmse, best_mae, const_mae):
    """The panel records integers; is the model's gain bigger than one step?"""
    step = float(np.min(np.diff(np.unique(y))))
    return {
        "panel_quantisation_step": step,
        "sd_of_response": float(y.std(ddof=1)),
        "constant_RMSE": const_rmse, "best_model_RMSE": best_rmse,
        "RMSE_gain": const_rmse - best_rmse,
        "RMSE_gain_in_quantisation_steps": (const_rmse - best_rmse) / step,
        "constant_MAE": const_mae, "best_model_MAE": best_mae,
        "MAE_gain": const_mae - best_mae,
        "MAE_gain_in_quantisation_steps": (const_mae - best_mae) / step,
        "RMSE_gain_as_pct_of_0_100_scale": (const_rmse - best_rmse) / 100.0,
        # Rounding-only noise floor: a perfect continuous model still inherits
        # uniform quantisation error of width `step`.
        "irreducible_RMSE_from_rounding": step / np.sqrt(12),
        "max_R2_if_only_rounding_noise": 1 - (step ** 2 / 12) / y.var(ddof=1),
    }


def power_analysis(r2_values, n=70, p=5, alpha=0.05):
    """Sample size needed to detect a given out-of-sample R^2 at 80% power.

    Uses Cohen's f^2 = R^2/(1-R^2) with the standard noncentral-F approximation.
    """
    from scipy.stats import f as fdist
    from scipy.stats import ncf

    rows = []
    for r2 in r2_values:
        r2 = max(r2, 1e-6)
        f2 = r2 / (1 - r2)

        def power_at(nn):
            df1, df2 = p, nn - p - 1
            if df2 < 1:
                return 0.0
            crit = fdist.ppf(1 - alpha, df1, df2)
            return float(1 - ncf.cdf(crit, df1, df2, f2 * nn))

        need = next((nn for nn in range(p + 2, 5000) if power_at(nn) >= 0.80), None)
        rows.append({"target_R2": r2, "cohen_f2": f2,
                     "power_at_n70": power_at(n), "n_for_80pct_power": need})
    return pd.DataFrame(rows)


def detection_reframe(ds, thresholds=(2, 3)):
    """Does bitterness survive as a *detection* task where regression fails?"""
    y = ds.y("bitter")
    X = ds.X_voigt5("bitter")
    rows = []
    for thr in thresholds:
        lab = (y >= thr).astype(int)
        if lab.sum() < 5 or (1 - lab).sum() < 5:
            continue
        score = np.empty(len(y))
        pred = np.empty(len(y), dtype=int)
        for tr, te in LeaveOneOut().split(X):
            if len(np.unique(lab[tr])) < 2:
                score[te], pred[te] = lab[tr][0], lab[tr][0]
                continue
            m = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=1.0, max_iter=5000,
                                                    class_weight="balanced"))]).fit(X[tr], lab[tr])
            score[te] = m.predict_proba(X[te])[:, 1]
            pred[te] = m.predict(X[te])
        rows.append({
            "threshold": f"bitter >= {thr}",
            "n_positive": int(lab.sum()), "n_negative": int((1 - lab).sum()),
            "prevalence": float(lab.mean()),
            "LOO_ROC_AUC": float(roc_auc_score(lab, score)),
            "LOO_balanced_accuracy": float(balanced_accuracy_score(lab, pred)),
            "majority_class_accuracy": float(max(lab.mean(), 1 - lab.mean())),
        })
    return pd.DataFrame(rows)


def rounded_agreement(ds, model_names):
    """After rounding to the panel's integer grid, how often is each model exactly right?"""
    y = ds.y("bitter")
    X = ds.X_voigt5("bitter")
    zoo = build_models()
    rows = []
    for name in model_names + ["Constant (mean)", "Constant (median)"]:
        p = loo_predict(zoo[name][0], X, y)
        r = np.round(np.clip(p, 0, 100))
        rows.append({"model": name,
                     "exact_hit_rate": float(np.mean(r == y)),
                     "within_1_unit": float(np.mean(np.abs(r - y) <= 1)),
                     "distinct_rounded_predictions": int(len(np.unique(r)))})
    return pd.DataFrame(rows)


def main():
    ds = load()
    y = ds.y("bitter")
    out = {}

    print("=" * 78)
    print("E3.1  RESPONSE DISTRIBUTION, ALL FIVE DIMENSIONS")
    print("=" * 78)
    desc = describe_targets(ds)
    print(desc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    desc.to_csv(os.path.join(RESULTS, "e3_target_description.csv"), index=False)

    print("\n" + "=" * 78)
    print("E3.2  WHERE THE BITTERNESS VARIANCE ACTUALLY LIVES")
    print("=" * 78)
    vc = variance_concentration(y)
    names = [r["recipe_name"] for r in ds.raw_recipes]
    top = np.argsort(-((y - y.mean()) ** 2))[:5]
    for i in top:
        print(f"    bitter={y[i]:.0f}  "
              f"{(y[i]-y.mean())**2/vc['SST']:6.1%} of SST   {names[i]}")
    for k, v in vc.items():
        if k.startswith("top"):
            print(f"    {k:<32s} {v:6.1%}")
    out["variance_concentration"] = vc
    # Same statistic for the other dimensions, for contrast.
    ctrl = {t: variance_concentration(ds.y(t))["top2_recipes_share_of_SST"] for t in TASTES}
    print("\n    top-2-recipe share of SST by dimension: " +
          ", ".join(f"{t}={v:.1%}" for t, v in ctrl.items()))
    out["top2_share_by_taste"] = ctrl

    print("\n" + "=" * 78)
    print("E3.3  IS THE MODEL GAIN LARGER THAN THE PANEL'S OWN RESOLUTION?")
    print("=" * 78)
    bench = pd.read_csv(os.path.join(RESULTS, "e1_model_benchmark.csv"))
    b = bench[bench.taste == "bitter"]
    const = b[b.model == "Constant (mean)"].iloc[0]
    best_rmse_row = b.loc[b.RMSE.idxmin()]
    best_mae_row = b.loc[b.MAE.idxmin()]
    q = quantisation_context(y, float(best_rmse_row.RMSE), float(const.RMSE),
                             float(best_mae_row.MAE), float(const.MAE))
    q["best_RMSE_model"] = str(best_rmse_row.model)
    q["best_MAE_model"] = str(best_mae_row.model)
    for k, v in q.items():
        print(f"    {k:<38s} {v if isinstance(v, str) else round(v, 4)}")
    out["quantisation"] = q

    print("\n" + "=" * 78)
    print("E3.4  STATISTICAL POWER")
    print("=" * 78)
    obs_r2 = float(b.R2_oos.max())
    pw = power_analysis([obs_r2, 0.10, 0.25, 0.50])
    print(pw.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pw.to_csv(os.path.join(RESULTS, "e3_power.csv"), index=False)

    print("\n" + "=" * 78)
    print("E3.5  DETECTION REFRAME (classification instead of regression)")
    print("=" * 78)
    det = detection_reframe(ds)
    print(det.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    det.to_csv(os.path.join(RESULTS, "e3_detection.csv"), index=False)

    print("\n" + "=" * 78)
    print("E3.6  AGREEMENT ON THE PANEL'S OWN INTEGER GRID")
    print("=" * 78)
    ra = rounded_agreement(ds, ["Lasso 5D", "Ordinal (prop. odds)", "log1p-transform Ridge"])
    print(ra.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ra.to_csv(os.path.join(RESULTS, "e3_rounded_agreement.csv"), index=False)

    with open(os.path.join(RESULTS, "e3_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] results written to {RESULTS}")


if __name__ == "__main__":
    main()
