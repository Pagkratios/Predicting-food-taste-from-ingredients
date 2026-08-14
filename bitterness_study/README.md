# Bitterness sensitivity study

Self-contained response to the reviewer comment:

> **Questionable value of modeling bitterness.** The ground-truth bitterness scores
> have a mean of 1.6 (on a 0–100 scale), with 96% of recipes scoring ≤3, indicating
> a severe floor effect. The authors themselves acknowledge that the bitterness
> result is "not evidence of model accuracy." Under these circumstances, applying
> the same regression structure to bitterness is statistically inefficient and may
> overfit noise.

**This folder does not modify any existing code or result.** `src/data.py` imports
the chemistry features and HS bounds from `Hybrid/src/hybrid_analysis.py` read-only,
so the study cannot drift from the published numbers, and every output is written
inside `bitterness_study/`.

## Run

```bash
python3 bitterness_study/src/run_benchmark.py       # E1  model zoo, all 5 dimensions  (~5 min)
python3 bitterness_study/src/run_significance.py    # E2  permutation + bootstrap      (~5 min)
python3 bitterness_study/src/run_diagnostics.py     # E3  floor-effect characterisation (~1 min)
python3 bitterness_study/src/make_bitterness_figure.py   # supplementary figure
```

Requires `statsmodels` in addition to the repo's existing dependencies
(`pip install statsmodels`), used only for the negative-binomial GLM.

## What was tested

**E1 — 19 model structures on one identical leave-one-out protocol.** Same folds,
same data, all hyper-parameter tuning inside the fold, for bitterness and for the
other four dimensions as a protocol control.

| Family | Models |
|---|---|
| Constant baselines | training-fold mean, training-fold median |
| Unfitted physical | HS midpoint, Voigt weighted average, and the manuscript's φ-calibrated HS and RV predictors |
| As published | Lasso 5D, Hybrid HS+chem, Lasso per-ingredient (115 features) |
| Alternative shrinkage | Ridge, ElasticNet |
| Floor-appropriate | Poisson GLM, negative-binomial GLM, Tobit (left-censored at 0), proportional-odds ordinal, multinomial-over-levels, √-transform, log1p-transform, clip-at-zero, shrink-to-mean with CV-tuned blend weight |
| Nonlinear | random forest, gradient boosting, k-NN |

**E2 — is the signal real and is it worth anything.** 1000-shuffle permutation test
of the entire LOO pipeline (directly tests "may overfit noise"); bootstrap CIs on
R² and MAE; paired bootstrap of MAE against both constant baselines; refit with the
two extreme recipes removed.

**E3 — properties of the response itself.** Level structure, where the variance
lives, the model gain expressed in units of the panel's own quantisation step,
statistical power, exact-hit rate on the integer grid, and a
detection (classification) reframe.

## Headline results

See `SUMMARY.md` for the full write-up and `RESPONSE_DRAFT.md` for the text
intended for the response letter. In brief:

1. The bitterness signal is **not** noise — permutation p = 0.001–0.002 for every
   model family tested. That part of the reviewer's concern is not supported.
2. But no model, of any structure, beats a **constant prediction of 1.0** on MAE.
3. The best achievable RMSE gain over a constant is 0.18 scale points — under one
   fifth of the panel's own 1-point resolution, and 0.18% of the 0–100 scale.
4. Two chocolate recipes carry 62% of the total bitterness variance; removing them
   halves the apparent R².
5. Floor-appropriate structures (ordinal, log1p, Poisson) *do* beat the published
   Lasso, and the published **hybrid** model is worse than a constant on bitterness
   (R² = −0.39) — the one place the reviewer's overfitting concern lands.
6. The same protocol yields R² = 0.32–0.69, p = 0.001, on the other four dimensions,
   so the null result is a property of the bitterness data, not the method.

## Outputs

```
results/
  e1_model_benchmark.csv        every model x every dimension
  e1_bitter_table.md            readable bitterness ranking
  e1_loo_predictions.json       per-recipe LOO predictions for reuse
  e2_permutation_bitter.csv     permutation test, bitterness
  e2_permutation_control.csv    permutation test, other four dimensions
  e2_bootstrap_bitter.csv       bootstrap CIs
  e2_paired_vs_constant.csv     paired MAE differences vs constants
  e2_outlier_leverage.csv       refit without the two bitter=8 recipes
  e2_null_distributions.npz     raw permutation nulls
  e3_target_description.csv     distribution of all five dimensions
  e3_diagnostics.json           variance concentration, quantisation context
  e3_power.csv                  power analysis
  e3_detection.csv              classification reframe
  e3_rounded_agreement.csv      exact-hit rate on the integer grid
figures/
  Bitterness_analysis.png / Bitterness_analysis.pdf
```
