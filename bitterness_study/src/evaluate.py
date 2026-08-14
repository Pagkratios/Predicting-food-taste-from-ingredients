#!/usr/bin/env python3
"""Shared leave-one-out harness and metrics.

Every model in the study is scored through `loo_predict`, so the comparison is
apples-to-apples: identical folds, identical data, all tuning inside the fold.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")


def loo_predict(factory, X, y):
    """Strict leave-one-out. `factory()` must return a fresh unfitted estimator."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    preds = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        preds[te] = factory().fit(X[tr], y[tr]).predict(X[te])
    return preds


def metrics(pred, y, ref_pred=None):
    """Out-of-sample metrics.

    `ref_pred` is the leave-one-out constant-mean baseline. Skill scores are
    reported against it because on a floor-effect dimension R^2 against the
    full-sample mean flatters models that merely learn the mean.
    """
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    err = pred - y
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))

    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2_oos": 1.0 - sse / sst if sst > 0 else np.nan,
        "Bias": float(np.mean(err)),
    }
    if np.std(pred) > 1e-12:
        out["PCC"] = float(np.corrcoef(pred, y)[0, 1])
        out["Spearman"] = float(stats.spearmanr(pred, y).statistic)
    else:
        out["PCC"] = np.nan
        out["Spearman"] = np.nan

    if ref_pred is not None:
        ref_err = np.asarray(ref_pred, float) - y
        ref_mse = float(np.mean(ref_err ** 2))
        ref_mae = float(np.mean(np.abs(ref_err)))
        out["Skill_MSE"] = 1.0 - np.mean(err ** 2) / ref_mse if ref_mse > 0 else np.nan
        out["Skill_MAE"] = 1.0 - out["MAE"] / ref_mae if ref_mae > 0 else np.nan
        out["dMAE_vs_const"] = out["MAE"] - ref_mae
        # Paired test on per-recipe absolute errors, model vs constant baseline.
        d = np.abs(err) - np.abs(ref_err)
        if np.any(d != 0):
            out["Wilcoxon_p_vs_const"] = float(stats.wilcoxon(np.abs(err), np.abs(ref_err)).pvalue)
        else:
            out["Wilcoxon_p_vs_const"] = 1.0
        out["Frac_recipes_improved"] = float(np.mean(d < 0))
    return out


def paired_bootstrap_dmae(pred_a, pred_b, y, n_boot=10000, seed=0):
    """Bootstrap CI for MAE(a) - MAE(b) over recipes, and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    ea = np.abs(np.asarray(pred_a, float) - y)
    eb = np.abs(np.asarray(pred_b, float) - y)
    d = ea - eb
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return {
        "dMAE": float(d.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "p_boot": float(min(1.0, p)),
    }


def bootstrap_metric_ci(pred, y, fn, n_boot=10000, seed=0):
    """Percentile CI for any metric computed on resampled recipes."""
    rng = np.random.default_rng(seed)
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(y[i]) < 1e-12:
            continue
        vals.append(fn(pred[i], y[i]))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return {
        "point": float(fn(pred, y)),
        "ci_lo": float(np.percentile(vals, 2.5)),
        "ci_hi": float(np.percentile(vals, 97.5)),
    }


def r2_of(pred, y):
    sst = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((pred - y) ** 2) / sst if sst > 0 else np.nan


def mae_of(pred, y):
    return float(np.mean(np.abs(pred - y)))
