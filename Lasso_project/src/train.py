#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training + required plots only.

Plots produced:
  1) Predicted vs Actual -- three plots (HS, RV, Lasso)
     - Ideal line, PCC and R^2 at top-left, sensory legend at bottom-right
  2) Single RMSE box plot covering all sensory categories and methods
     - Y-axis = RMSE, X-axis groups = Sweet/Bitter/Salty/Umami/Sour x HS/RV/Lasso
"""

import json
import math
import os
import pickle
import pprint
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    r2_score, mean_squared_error, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr

from lasso import LassoRegressor
from env_config import get_raw_recipes_path

# ---------------- Config & Paths ----------------
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
PROC_DIR        = os.path.join(DATA_DIR, "processed")
SRC2_RESULTS    = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR      = os.path.join(SRC2_RESULTS, "models")
PLOTS_DIR       = os.path.join(SRC2_RESULTS, "plots")
METRICS_DIR     = os.path.join(SRC2_RESULTS, "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RECIPE_FILE = get_raw_recipes_path()
PRED_FILE   = os.path.join(DATA_DIR, "data_predictions.py")
X_FILE      = os.path.join(PROC_DIR, "X_train.npy")
Y_FILE      = os.path.join(PROC_DIR, "Y_train.npy")
MODEL_FILE  = os.path.join(MODELS_DIR, "final_models.pkl")
RESULTS_CSV = os.path.join(MODELS_DIR, "real_vs_predicted.csv")

# Import shared constants from plot_config
from plot_config import (
    SENSORY_ORDER, ALIASES, METHOD_ORDER, METHOD_DISPLAY,
    SENSORY_COLORS, METHOD_COLORS, METHOD_COLORS_LIGHT,
)

# Paths for Hashin-Shtrikman and Reuss-Voigt prediction data
HS_PRED_FILE = os.path.join(DATA_DIR, "hs_predictions.py")
RV_PRED_FILE = os.path.join(DATA_DIR, "rv_predictions.py")

# ---------------- Utils & Loaders ----------------
def load_attr_from_py(filepath, variable_name):
    spec = importlib.util.spec_from_file_location(variable_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, variable_name)

def normalize_weights(ingredients):
    total = sum(ing['weight'] for ing in ingredients)
    if total == 0:
        return [{**ing, 'weight': 0.0} for ing in ingredients]
    return [{**ing, 'weight': ing['weight'] / total} for ing in ingredients]

def compute_x_vector(ingredients):
    v = np.zeros(len(SENSORY_ORDER), dtype=float)
    for ing in ingredients:
        w = float(ing['weight'])
        s = ing['sensory_scores']
        v += w * np.array([float(s[ALIASES.get(k, k)]) for k in SENSORY_ORDER], dtype=float)
    return v

def build_X_from_raw(raw_recipes):
    X, names = [], []
    for r in raw_recipes:
        ings = normalize_weights(r['ingredients'])
        X.append(compute_x_vector(ings))
        names.append(r['recipe_name'])
    return np.vstack(X), names

def build_Y_from_raw(raw_recipes):
    Y = []
    for r in raw_recipes:
        scores = r['food_sensory_scores']
        Y.append([float(scores[ALIASES.get(k, k)]) for k in SENSORY_ORDER])
    return np.vstack(Y)

def standardize_fit(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    Xs = (X - mean) / std_safe
    return Xs, mean, std_safe

def center_targets(y):
    y_mean = y.mean()
    return y - y_mean, y_mean

def normalize_attr_dict(d):
    out = {}
    for k, v in d.items():
        ck = ALIASES.get(k)
        if ck in SENSORY_ORDER:
            out[ck] = float(v)
    return out

def extract_from_pred_data(pred_data, method_key):
    preds, actuals, labels = [], [], []
    for _, entry in pred_data.items():
        if not isinstance(entry, dict) or 'Actual' not in entry or method_key not in entry:
            continue
        act = normalize_attr_dict(entry['Actual'])
        prd = normalize_attr_dict(entry[method_key])
        for k in SENSORY_ORDER:
            if k in act and k in prd:
                preds.append(prd[k]); actuals.append(act[k]); labels.append(k)
    return preds, actuals, labels

# ---------------- Metrics ----------------
def _safe_mape(a, p, eps=1e-8):
    a = np.asarray(a, float); p = np.asarray(p, float)
    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    return np.mean(np.abs((a - p) / denom)) * 100.0


def _smape(a, p, eps=1e-8):
    a = np.asarray(a, float); p = np.asarray(p, float)
    denom = np.maximum(eps, (np.abs(a) + np.abs(p)) / 2.0)
    return np.mean(np.abs(a - p) / denom) * 100.0


def compute_metrics(preds, actuals):
    a = np.asarray(actuals, float); p = np.asarray(preds, float)
    err = p - a
    mae = np.mean(np.abs(err))
    medae = np.median(np.abs(err))
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(a, p) if len(a) >= 2 else np.nan
    evs = explained_variance_score(a, p) if len(a) >= 2 else np.nan
    pr, _ = pearsonr(p, a) if len(a) >= 2 else (np.nan, None)
    sr, _ = spearmanr(p, a) if len(a) >= 2 else (np.nan, None)
    bias = np.mean(err)
    sderr = np.std(err, ddof=1) if len(err) > 1 else np.nan
    mape = _safe_mape(a, p)
    smape = _smape(a, p)
    return {
        "MAE": mae, "Median_AE": medae, "MSE": mse, "RMSE": rmse,
        "R2": r2, "ExplainedVar": evs,
        "Pearson_r": pr, "Spearman_rho": sr,
        "Bias": bias, "Std_Error": sderr,
        "MAPE_%": mape, "sMAPE_%": smape
    }


def evaluate_and_save(method_name, preds, actuals, labels, out_dir):
    df_rows = []
    preds = np.asarray(preds, float)
    actuals = np.asarray(actuals, float)
    labels = np.asarray(labels)

    for key in SENSORY_ORDER:
        mask = (labels == key)
        if not np.any(mask):
            continue
        m = compute_metrics(preds[mask], actuals[mask])
        m["method"] = method_name
        m["target"] = key
        df_rows.append(m)

    m_overall = compute_metrics(preds, actuals)
    m_overall["method"] = method_name
    m_overall["target"] = "overall"
    df_rows.append(m_overall)

    df = pd.DataFrame(df_rows)
    order = ["method", "target", "MAE", "Median_AE", "MSE", "RMSE",
             "R2", "ExplainedVar", "Pearson_r", "Spearman_rho",
             "Bias", "Std_Error", "MAPE_%", "sMAPE_%"]
    df = df[order]
    path = os.path.join(out_dir, f"{method_name.lower()}_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[+] Metrics saved: {path}")
    return df


def save_combined_tables(dfs, out_dir):
    all_df = pd.concat(dfs, ignore_index=True)
    long_path = os.path.join(out_dir, "all_metrics_long.csv")
    all_df.to_csv(long_path, index=False)

    wide = (all_df.set_index(["target", "method"]).sort_index())
    tables = []
    for metric in ["MAE","Median_AE","MSE","RMSE","R2","ExplainedVar",
                   "Pearson_r","Spearman_rho","Bias","Std_Error","MAPE_%","sMAPE_%"]:
        sub = (wide[[metric]].reset_index().pivot(index="target", columns="method", values=metric)
               .reindex(SENSORY_ORDER + ["overall"]))
        sub.insert(0, "Metric", metric)
        tables.append(sub.reset_index(names="target"))
    wide_out = pd.concat(tables, ignore_index=True)
    wide_csv = os.path.join(out_dir, "all_metrics_wide.csv")
    wide_out.to_csv(wide_csv, index=False)

    md_path = os.path.join(out_dir, "all_metrics.md")
    with open(md_path, "w") as f:
        f.write(wide_out.to_markdown(index=False))
    tex_path = os.path.join(out_dir, "all_metrics.tex")
    with open(tex_path, "w") as f:
        f.write(wide_out.to_latex(index=False, float_format="%.3f"))

    print(f"[+] Combined tables saved:\n  - {wide_csv}\n  - {md_path}\n  - {tex_path}")

# ---------------- Alpha tuning & training ----------------
def tune_best_alphas(X_std, Y, alphas):
    best_alphas = {}
    loo = LeaveOneOut()
    for i, key in enumerate(SENSORY_ORDER):
        print(f"\n[INFO] Tuning alpha for '{key}'")
        y = Y[:, i]
        y_centered, _ = center_targets(y)
        avg_errors = []
        for alpha in alphas:
            fold_errors = []
            for tr, te in loo.split(X_std):
                model = LassoRegressor(alpha=alpha, learning_rate=0.01, max_iter=1000, tol=1e-6, verbose=False)
                model.fit(X_std[tr], y_centered[tr])
                y_pred = model.predict(X_std[te]) + y[tr].mean()
                fold_errors.append(mean_squared_error(y[te], y_pred))
            avg_errors.append(np.mean(fold_errors))
            print(f"  alpha={alpha:.5f} -> MSE={avg_errors[-1]:.4f}")
        best = alphas[np.argmin(avg_errors)]
        best_alphas[key] = best
        print(f"[+] Selected alpha for '{key}': {best:.5f}")
    return best_alphas

def train_final_models(X, Y, best_alphas):
    models = {}
    inmem_preds, inmem_actuals, inmem_labels = [], [], []

    for i, key in enumerate(SENSORY_ORDER):
        y = Y[:, i]  # raw targets in original range
        alpha = best_alphas[key]
        print(f"\n[INFO] Training final model for '{key}' with alpha={alpha:.5f}")
        m = LassoRegressor(alpha=alpha, learning_rate=0.01, max_iter=1000, tol=1e-6, verbose=False)
        m.fit(X, y)  # train on raw features and raw targets
        y_pred_raw = m.predict(X)
        inmem_preds.extend(y_pred_raw.tolist())
        inmem_actuals.extend(y.tolist())
        inmem_labels.extend([key] * len(y))
        models[key] = m
        print(f"  -> Coefficients: {m.get_coefficients().round(3)}")
        print(f"  -> Intercept: {m.get_intercept():.3f}")

    return models, np.array(inmem_preds), np.array(inmem_actuals), inmem_labels

# ---------------- Preds export ----------------
def update_data_predictions_from_matrix(recipes, pred_matrix, pred_file_path):
    """
    Writes Lasso predictions back into data_predictions.py for each recipe,
    using a (n_recipes x len(SENSORY_ORDER)) matrix that aligns with recipe order.
    """
    with open(pred_file_path, 'r') as f:
        content = f.read()
        global_dict = {}
        exec(content, global_dict)
        pred_data = global_dict.get('pred_data', {})

    for i, r in enumerate(recipes):
        name = r['recipe_name']
        if name not in pred_data:
            pred_data[name] = {}
        if 'Lasso prediction' not in pred_data[name]:
            pred_data[name]['Lasso prediction'] = {}
        for j, key in enumerate(SENSORY_ORDER):
            pred_data[name]['Lasso prediction'][key] = float(round(pred_matrix[i, j], 2))

    with open(pred_file_path, 'w') as f:
        f.write("pred_data = ")
        f.write(pprint.pformat(pred_data, indent=4, width=120))
        f.write("\n")
    print(f"[+] Lasso predictions updated in: {pred_file_path}")


def predict_recipes_table(
    models,
    X,
    recipe_names,
    raw_recipes,
    out_json,
    round_ndigits=2,
    include_diff=True,
):
    """
    Creates a JSON list with one object per recipe containing pred, actual,
    and optional diff dicts for each sensory attribute.
    """

    def _safe_round(x):
        if x is None:
            return None
        try:
            xf = float(x)
            if math.isnan(xf):
                return None
            return round(xf, round_ndigits)
        except Exception:
            return None

    name_to_idx = {nm: i for i, nm in enumerate(recipe_names)}

    preds_by_target = {}
    for key in SENSORY_ORDER:
        m = models[key]
        preds_by_target[key] = np.asarray(m.predict(X), dtype=float)

    raw_actuals = {}
    for r in raw_recipes:
        nm = r.get("recipe_name")
        scores = r.get("food_sensory_scores", {})
        canon = normalize_attr_dict(scores)
        raw_actuals[nm] = canon

    rows = []
    missing_in_X = []
    missing_actuals = []

    for r in raw_recipes:
        nm = r.get("recipe_name")

        if nm not in name_to_idx:
            missing_in_X.append(nm)
            pred_dict = {k: None for k in SENSORY_ORDER}
        else:
            i = name_to_idx[nm]
            pred_dict = {k: _safe_round(preds_by_target[k][i]) for k in SENSORY_ORDER}

        act_dict_raw = raw_actuals.get(nm, {})
        act_dict = {k: _safe_round(act_dict_raw.get(k, None)) for k in SENSORY_ORDER}
        if not act_dict_raw:
            missing_actuals.append(nm)

        diff_dict = None
        if include_diff:
            diff_dict = {}
            for k in SENSORY_ORDER:
                pv = pred_dict[k]
                av = act_dict[k]
                diff_dict[k] = None if pv is None or av is None else _safe_round(pv - av)

        row = {
            "recipe_name": nm,
            "pred": pred_dict,
            "actual": act_dict,
        }
        if include_diff:
            row["diff"] = diff_dict

        rows.append(row)

    if missing_in_X:
        print(f"[!] {len(missing_in_X)} recipe(s) not found in recipe_names/X, preds set to None.")
    if missing_actuals:
        print(f"[!] {len(missing_actuals)} recipe(s) missing actual food_sensory_scores.")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved per-recipe predictions JSON -> {out_json}")

    return rows


# ════════════════════════════════════════════════════════════════════════
# NEW (v2) publication-quality plotting functions -- unified plot_config
# ════════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual_v2(preds, actuals, labels, method_name, output_file):
    """Predicted vs Actual scatter with unified styling."""
    from plot_config import (
        apply_style, setup_figure, save_figure_fixed, style_axes,
        sensory_legend_handles, SENSORY_COLORS, METHOD_DISPLAY,
        FONT_SIZE_ANNOTATION, FONT_SIZE_LEGEND,
    )
    if not preds or not actuals:
        print(f"[!] No data to plot for {method_name}.")
        return

    apply_style()
    p = np.clip(np.asarray(preds, float), 0, 100)
    a = np.clip(np.asarray(actuals, float), 0, 100)
    r_value, _ = pearsonr(p, a) if len(p) >= 2 else (np.nan, None)
    r_squared = r2_score(a, p) if len(p) >= 2 else np.nan

    fig, ax = setup_figure(size="single_sq")
    style_axes(ax)

    colors = [SENSORY_COLORS[lbl] for lbl in labels]
    ax.scatter(p, a, c=colors, alpha=0.85, s=25, linewidths=0.3, edgecolors="#333333")

    ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8, label='Ideal')

    txt = f"PCC = {r_value:.2f}\n$R^2$ = {r_squared:.2f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=FONT_SIZE_ANNOTATION,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

    handles = sensory_legend_handles(markersize=4)
    handles.append(plt.Line2D([0], [0], color='k', linestyle='--', linewidth=0.8, label='Ideal'))
    ax.legend(handles=handles, loc='lower right', fontsize=FONT_SIZE_LEGEND, frameon=True)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Predicted vs Actual ({method_name})")

    save_figure_fixed(fig, output_file)
    print(f"    PCC={r_value:.3f}, R2={r_squared:.3f}")


def plot_rmse_boxplot_v2(hs, rv, lasso, out_path):
    """Grouped RMSE box plot with unified styling."""
    from plot_config import (
        apply_style, setup_figure, save_figure, style_axes,
        METHOD_COLORS_LIGHT, METHOD_COLORS, METHOD_DISPLAY,
        FONT_SIZE_LABEL, FONT_SIZE_TITLE, FONT_SIZE_LEGEND,
    )
    apply_style()

    (hs_p, hs_a, hs_l) = hs
    (rv_p, rv_a, rv_l) = rv
    (ls_p, ls_a, ls_l) = lasso

    def abs_errors(preds, actuals, labels):
        p = np.asarray(preds, float)
        a = np.asarray(actuals, float)
        labs = np.asarray(labels)
        return {k: np.abs(p[labs == k] - a[labs == k]) for k in SENSORY_ORDER}

    hs_err = abs_errors(hs_p, hs_a, hs_l)
    rv_err = abs_errors(rv_p, rv_a, rv_l)
    ls_err = abs_errors(ls_p, ls_a, ls_l)

    data, tick_labels = [], []
    for sens in SENSORY_ORDER:
        data.extend([
            hs_err.get(sens, np.array([])),
            rv_err.get(sens, np.array([])),
            ls_err.get(sens, np.array([])),
        ])
        tick_labels.extend([
            "HS\n",
            f"RV\n{sens.capitalize()}",
            "Lasso\n",
        ])

    fig, ax = setup_figure(size="double_wide")
    style_axes(ax)

    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    method_colors = [METHOD_COLORS_LIGHT["HS"], METHOD_COLORS_LIGHT["RV"], METHOD_COLORS_LIGHT["Lasso"]]
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(method_colors[i % 3])
        box.set_alpha(0.95)
        box.set_edgecolor('black')
        box.set_linewidth(0.6)
    for median in bp['medians']:
        median.set_linewidth(1.2)
        median.set_color('black')

    ax.set_ylabel("RMSE", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_title("Ingredient to Taste (RMSE vs Method)", fontsize=FONT_SIZE_TITLE, fontweight="bold")

    positions = np.arange(1, len(tick_labels) + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker='s', linestyle='', markersize=8,
                   color=METHOD_COLORS[m], label=METHOD_DISPLAY[m])
        for m in METHOD_ORDER
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=FONT_SIZE_LEGEND)

    save_figure(fig, out_path)


# ---------------- Main ----------------
def main():
    print("[*] Loading processed data and raw assets...")
    X = np.load(X_FILE)
    Y = np.load(Y_FILE)
    raw_recipes = load_attr_from_py(RECIPE_FILE, 'raw_recipes')
    pred_data   = load_attr_from_py(PRED_FILE,   'pred_data')

    # Load HS/RV from separate data files
    hs_pred_data = load_attr_from_py(HS_PRED_FILE, 'hs_predictions')
    rv_pred_data = load_attr_from_py(RV_PRED_FILE, 'rv_predictions')

    print("[*] Standardizing X for alpha tuning...")
    X_std, x_mean, x_scale = standardize_fit(X)
    np.save(os.path.join(PROC_DIR, "x_scaler_mean.npy"), x_mean)
    np.save(os.path.join(PROC_DIR, "x_scaler_scale.npy"), x_scale)

    # Alpha tuning
    print("[*] Tuning alpha via LOO...")
    alphas = np.logspace(-4, 0.5, 30)
    best_alphas = tune_best_alphas(X_std, Y, alphas)

    # Train final models on raw X so predictions match HS/RV scale
    print("[*] Training final models...")
    models, lasso_preds_all, lasso_actuals_all, lasso_labels_all = train_final_models(X, Y, best_alphas)

    # Save models
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(models, f)
    print(f"[+] Models saved to: {MODEL_FILE}")

    # Per-recipe prediction table
    recipe_names = [r['recipe_name'] for r in raw_recipes]
    OUT_JSON = os.path.join(MODELS_DIR, "per_recipe_lasso_predictions.json")
    _ = predict_recipes_table(
        models=models, X=X, recipe_names=recipe_names,
        raw_recipes=raw_recipes, out_json=OUT_JSON,
    )

    # Long/flattened CSV
    results_df = pd.DataFrame({
        'label': lasso_labels_all,
        'actual': lasso_actuals_all,
        'predicted': lasso_preds_all,
    })
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"[+] Long-format predictions saved to: {RESULTS_CSV}")

    # Extract HS & RV predictions from their separate data files
    hs_preds, hs_actuals, hs_labels = extract_from_pred_data(hs_pred_data, "HS prediction")
    rv_preds, rv_actuals, rv_labels = extract_from_pred_data(rv_pred_data, "RV prediction")

    # ── New publication-quality plots (unified style) ──
    plot_predicted_vs_actual_v2(
        hs_preds, hs_actuals, hs_labels, method_name="HS",
        output_file=os.path.join(PLOTS_DIR, "hs_predicted_vs_actual.png"))
    plot_predicted_vs_actual_v2(
        rv_preds, rv_actuals, rv_labels, method_name="RV",
        output_file=os.path.join(PLOTS_DIR, "rv_predicted_vs_actual.png"))
    plot_predicted_vs_actual_v2(
        lasso_preds_all.tolist(), lasso_actuals_all.tolist(), lasso_labels_all,
        method_name="Lasso",
        output_file=os.path.join(PLOTS_DIR, "lasso_predicted_vs_actual.png"))
    plot_rmse_boxplot_v2(
        hs=(hs_preds, hs_actuals, hs_labels),
        rv=(rv_preds, rv_actuals, rv_labels),
        lasso=(lasso_preds_all.tolist(), lasso_actuals_all.tolist(), lasso_labels_all),
        out_path=os.path.join(PLOTS_DIR, "rmse_boxplot_all_methods.png"))

    # Metrics
    dfs = []
    dfs.append(evaluate_and_save("HS", hs_preds, hs_actuals, hs_labels, METRICS_DIR))
    dfs.append(evaluate_and_save("RV", rv_preds, rv_actuals, rv_labels, METRICS_DIR))
    dfs.append(evaluate_and_save("Lasso", lasso_preds_all, lasso_actuals_all, lasso_labels_all, METRICS_DIR))
    save_combined_tables(dfs, METRICS_DIR)

    # Write Lasso predictions back to data_predictions.py
    print("[*] Updating data_predictions.py with Lasso predictions...")
    lasso_preds_matrix = np.column_stack([
        models[k].predict(X) for k in SENSORY_ORDER
    ])
    update_data_predictions_from_matrix(raw_recipes, lasso_preds_matrix, PRED_FILE)

    print("[+] Done.")


if __name__ == "__main__":
    main()
