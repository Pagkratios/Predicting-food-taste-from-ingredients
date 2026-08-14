#!/usr/bin/env python3
"""E1 — model zoo under one identical leave-one-out protocol.

Runs every candidate model on bitterness, and on the other four taste dimensions
as a protocol control: if the same harness extracts real signal for sweet /
sour / umami / salty but not for bitter, the null result is a property of the
bitterness data, not of the method.

    python3 bitterness_study/src/run_benchmark.py [--tastes bitter ...]

Writes to bitterness_study/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import RESULTS, TASTES, load                      # noqa: E402
from evaluate import loo_predict, metrics                   # noqa: E402
from models import FEATURE_BLOCK, ConstantMean, Passthrough, build_models  # noqa: E402

# Ordinal / multinomial fits are only meaningful (and only tractable) when the
# response has few levels. Bitterness has 6; the other dimensions have 26-32.
MAX_LEVELS_FOR_CATEGORICAL = 12
CATEGORICAL_MODELS = {"Ordinal (prop. odds)", "Multinomial levels"}
# Poisson / NegBin / sqrt / log links need a non-negative response.
NONNEG_MODELS = {"Poisson GLM", "NegBinomial GLM", "sqrt-transform Ridge",
                 "log1p-transform Ridge"}


_PUBLISHED = {}

# The exported HS/RV files key recipes as "RP1 - <name>" and spell the tastes out.
_LONG_NAME = {"sweet": "sweetness", "sour": "sourness", "bitter": "bitterness",
              "umami": "umami", "salty": "saltiness"}


def published_hs_rv(ds, taste):
    """The manuscript's own phi-calibrated HS/RV predictions, aligned to recipe order.

    Read straight out of data/hs_predictions.py and data/rv_predictions.py so the
    table can be compared line-for-line with Table 1 of the paper.
    """
    if not _PUBLISHED:
        import importlib.util
        from data import REPO
        for key, fname, var in [("HS", "hs_predictions.py", "hs_predictions"),
                                ("RV", "rv_predictions.py", "rv_predictions")]:
            spec = importlib.util.spec_from_file_location(var, os.path.join(REPO, "data", fname))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _PUBLISHED[key] = getattr(mod, var)

    out = []
    for key in ("HS", "RV"):
        store = _PUBLISHED[key]
        # Index by recipe name alone, tolerating the "RPn - name" prefix.
        by_name = {}
        for k, v in store.items():
            by_name[k] = v
            if " - " in k:
                by_name[k.split(" - ", 1)[1]] = v

        vec, ok = [], True
        for r in ds.raw_recipes:
            entry = by_name.get(r["recipe_name"], {}).get(f"{key} prediction", {})
            val = entry.get(taste, entry.get(_LONG_NAME[taste]))
            if val is None:
                ok = False
                break
            vec.append(float(val))
        if ok:
            out.append((f"{key} as published (phi-calibrated)", np.asarray(vec, float),
                        "manuscript Table 1 predictor"))
        else:
            print(f"    [!] could not align published {key} predictions for '{taste}'")
    return out


def features_for(ds, model_name, taste):
    block = FEATURE_BLOCK[model_name]
    return {"voigt5": ds.X_voigt5, "hybrid": ds.X_hybrid,
            "ingredients": ds.X_ingredients, "chem": ds.X_chem_only}[block](taste)


def run_taste(ds, taste, model_zoo, verbose=True):
    y = ds.y(taste)
    n_levels = len(np.unique(y))

    # The reference every skill score is measured against.
    ref = loo_predict(ConstantMean, ds.X_voigt5(taste), y)

    rows, preds = [], {"Constant (mean) [ref]": ref.tolist()}

    # Unfitted physical predictors, scored through the same metric code.
    physical = [
        ("HS midpoint (uncalibrated)", ds.hs_mid(taste),
         "no fitting; midpoint of the HS bounds"),
        ("Voigt weighted avg (uncalibrated)", ds.voigt(taste),
         "no fitting; mass-weighted ingredient average"),
    ]
    # The manuscript's Table 1 HS/RV rows are phi-calibrated by LOO, so include
    # them verbatim rather than letting the uncalibrated versions stand in.
    for label, vec, note in physical + published_hs_rv(ds, taste):
        m = metrics(vec, y, ref)
        m.update(model=label, family="physical", taste=taste, note=note, seconds=0.0)
        rows.append(m)
        preds[label] = np.asarray(vec, float).tolist()

    for name, (factory, family, note) in model_zoo.items():
        if name in CATEGORICAL_MODELS and n_levels > MAX_LEVELS_FOR_CATEGORICAL:
            if verbose:
                print(f"    {name:<24s} skipped ({n_levels} levels > {MAX_LEVELS_FOR_CATEGORICAL})")
            continue
        if name in NONNEG_MODELS and y.min() < 0:
            continue

        X = features_for(ds, name, taste)
        t0 = time.time()
        try:
            p = loo_predict(factory, X, y)
        except Exception as exc:                     # keep the sweep going
            print(f"    {name:<24s} FAILED: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0

        m = metrics(p, y, ref)
        m.update(model=name, family=family, taste=taste, note=note, seconds=round(dt, 2))
        rows.append(m)
        preds[name] = p.tolist()

        if verbose:
            print(f"    {name:<24s} MAE={m['MAE']:6.3f}  RMSE={m['RMSE']:6.3f}  "
                  f"R2={m['R2_oos']:+6.3f}  skill(MSE)={m['Skill_MSE']:+6.3f}  "
                  f"p={m['Wilcoxon_p_vs_const']:.3f}  [{dt:.1f}s]")

    return rows, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tastes", nargs="*", default=TASTES)
    args = ap.parse_args()

    ds = load()
    zoo = build_models()
    print(f"[*] {ds.n} recipes, {len(ds.ingredient_names)} ingredients, "
          f"{len(zoo)} models + 2 unfitted predictors\n")

    all_rows, all_preds = [], {}
    for taste in args.tastes:
        y = ds.y(taste)
        print(f"\n=== {taste.upper()}  (mean={y.mean():.2f}, sd={y.std(ddof=1):.2f}, "
              f"levels={len(np.unique(y))}) ===")
        rows, preds = run_taste(ds, taste, zoo)
        all_rows.extend(rows)
        all_preds[taste] = preds

    df = pd.DataFrame(all_rows)
    cols = ["taste", "model", "family", "MAE", "RMSE", "R2_oos", "PCC", "Spearman",
            "Bias", "Skill_MSE", "Skill_MAE", "dMAE_vs_const",
            "Wilcoxon_p_vs_const", "Frac_recipes_improved", "seconds", "note"]
    df = df[[c for c in cols if c in df.columns]]
    out_csv = os.path.join(RESULTS, "e1_model_benchmark.csv")
    df.to_csv(out_csv, index=False)

    with open(os.path.join(RESULTS, "e1_loo_predictions.json"), "w") as f:
        json.dump(all_preds, f, indent=1)

    # Readable bitterness table, sorted best-first by MAE.
    if "bitter" in args.tastes:
        b = df[df.taste == "bitter"].sort_values("MAE").copy()
        show = b[["model", "family", "MAE", "RMSE", "R2_oos", "PCC", "Skill_MSE",
                  "Skill_MAE", "Wilcoxon_p_vs_const", "Frac_recipes_improved"]]
        md = os.path.join(RESULTS, "e1_bitter_table.md")
        with open(md, "w") as f:
            f.write("# E1 — every model tried on bitterness (leave-one-out, N=70)\n\n")
            f.write("`Skill_*` is relative to the leave-one-out constant-mean predictor; "
                    "positive means better than predicting the mean.\n"
                    "`Wilcoxon_p_vs_const` tests the paired per-recipe absolute errors "
                    "against that same constant.\n\n")
            f.write(show.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n")
        print(f"\n[+] {md}")
        print(show.to_string(index=False))

    print(f"[+] {out_csv}")


if __name__ == "__main__":
    main()
