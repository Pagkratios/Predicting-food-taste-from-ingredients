# Bitterness sensitivity study — findings

All numbers are leave-one-out, N = 70 recipes, produced by the scripts in `src/`.
Nothing in this study modifies the existing pipeline or any published result.

---

## The short version

The reviewer makes two separate claims. **One holds, one does not.**

| Reviewer's claim | Verdict | Evidence |
|---|---|---|
| The regression structure is *statistically inefficient* for bitterness | **Supported** | Floor-appropriate models beat the published Lasso by 7–12% MAE; and **no model of any structure beats a constant prediction of 1.0** |
| The model *may overfit noise* | **Not supported for the Lasso**, supported for the hybrid | Permutation test over the full LOO pipeline: p = 0.001–0.002 for every model family. The hybrid model, however, is genuinely worse than a constant (R² = −0.39) |

The practical conclusion is the same either way, and it is the one the manuscript
already reaches: **bitterness should not be treated as a predictive target in this
corpus.** What this study adds is that the conclusion is now empirical rather than
asserted — we tried 19 model structures and can say exactly how much is on the table.

---

## ⚠ Correction the manuscript should make proactively

§3.1 and §3.2 both state:

> *"a constant prediction of 2.0 achieves MAE = 1.06, only marginally worse than the
> model's 0.92"*

**2.0 is not the right constant, and the sentence understates the authors' own case.**
The MAE-optimal constant is the median, 1.0:

| Constant | MAE | RMSE |
|---|---|---|
| 0.0 | 1.571 | 2.091 |
| **1.0 (median)** | **0.714** | 1.493 |
| 1.57 (mean) | 0.910 | 1.379 |
| 2.0 (as quoted in the manuscript) | 1.057 | 1.444 |
| 3.0 | 1.743 | 1.986 |

The best constant achieves MAE 0.714 — **23% better** than the model's 0.92, not
"marginally worse". A reviewer who checks this will find the authors compared against
a deliberately handicapped baseline, which reads badly even though it was clearly
inadvertent. Fixing it in revision turns a potential follow-up objection into evidence
that the authors did the analysis properly.

Suggested replacement wording:

> *"a constant prediction of 1.0 — the MAE-optimal constant — achieves MAE = 0.71,
> better than any of the 19 model structures we evaluated (best 0.76; the Lasso
> reported here 0.86)."*

---

## E1 — 19 model structures, one identical LOO protocol

Ranked by MAE. `Skill` is relative to the leave-one-out constant-mean predictor.

| Model | Family | MAE | RMSE | R²(oos) | Skill(MSE) |
|---|---|---|---|---|---|
| **Constant = 1.0 (median)** | baseline | **0.714** | 1.493 | −0.172 | −0.138 |
| log1p-transform Ridge | floor-aware | 0.761 | 1.241 | +0.190 | +0.213 |
| √-transform Ridge | floor-aware | 0.770 | 1.249 | +0.179 | +0.203 |
| Ordinal (proportional odds) | floor-aware | 0.802 | **1.221** | **+0.216** | **+0.238** |
| Poisson GLM | floor-aware | 0.811 | 1.239 | +0.193 | +0.216 |
| Multinomial over levels | floor-aware | 0.823 | 1.300 | +0.111 | +0.136 |
| Negative-binomial GLM | floor-aware | 0.825 | 1.259 | +0.167 | +0.191 |
| Ridge | linear | 0.833 | 1.237 | +0.195 | +0.218 |
| Lasso shrunk to mean | floor-aware | 0.835 | 1.280 | +0.138 | +0.162 |
| Tobit (left-censored at 0) | floor-aware | 0.847 | 1.249 | +0.179 | +0.203 |
| Lasso per-ingredient (115 feat.) | as published | 0.854 | 1.444 | −0.096 | −0.065 |
| ElasticNet | linear | 0.854 | 1.254 | +0.174 | +0.197 |
| **Lasso 5D (as published)** | as published | 0.861 | 1.262 | +0.163 | +0.187 |
| Random forest | nonlinear | 0.863 | 1.327 | +0.074 | +0.100 |
| **RV as published (φ-calibrated)** | physical | 0.886 | 1.444 | −0.097 | −0.065 |
| **HS as published (φ-calibrated)** | physical | 0.900 | 1.488 | −0.164 | −0.131 |
| HS midpoint (uncalibrated) | physical | 0.920 | 1.395 | −0.023 | +0.006 |
| Constant = 1.57 (mean) | baseline | 0.923 | 1.399 | −0.029 | 0 |
| k-NN | nonlinear | 0.940 | 1.509 | −0.198 | −0.164 |
| Gradient boosting | nonlinear | 0.950 | 1.418 | −0.057 | −0.027 |
| **Hybrid HS+chem (as published)** | as published | 0.956 | 1.626 | **−0.389** | **−0.350** |
| Voigt weighted average | physical | 1.650 | 2.420 | −2.078 | −1.991 |

The HS and RV rows are the manuscript's own φ-calibrated predictors, read directly
from `data/hs_predictions.py` and `data/rv_predictions.py`; their MAEs (0.900, 0.886)
reproduce `results/metrics/all_metrics_long.csv` exactly, so this table is directly
comparable to Table 1.

Four things to take from this table:

1. **The constant wins on MAE.** Predicting `1.0` for every recipe gives MAE 0.714.
   The best fitted model of any kind gives 0.761. The published Lasso gives 0.861 —
   *worse than doing nothing*.
2. **Floor-appropriate structures do help, modestly.** Ordinal / log1p / Poisson
   improve on the published Lasso by 7–12% MAE and give the best RMSE. So the
   reviewer is right that the Gaussian-response Lasso is the wrong structure — it is
   just that the better structure still lands below a constant.
3. **The hybrid model is actively harmful on bitterness** (R² = −0.39, skill −0.35).
   This is consistent with the PCC of −0.10 already reported for bitterness in
   Table 1, and is the strongest single argument for keeping bitterness out of every
   aggregate — which the manuscript already does.
4. **The published HS and RV predictors are also worse than a constant** on this
   dimension (R² = −0.16 and −0.10). The floor problem is not specific to the
   regression; it defeats the physics-based predictors too.

## E2 — is it noise? No. Is it worth anything? Also no.

**Permutation test** (1000 shuffles of the bitterness labels, entire LOO pipeline
re-run each time — this is the direct test of "may overfit noise"):

| Model | Observed R² | Null mean | Null 95th pct | p |
|---|---|---|---|---|
| Lasso 5D | +0.163 | −0.043 | −0.016 | **0.001** |
| Ridge | +0.195 | −0.039 | −0.010 | **0.001** |
| Ordinal (prop. odds) | +0.216 | −0.077 | +0.001 | **0.002** |
| log1p Ridge | +0.190 | −0.053 | −0.026 | **0.002** |

The models are **not** fitting noise. There is a genuine, reproducible association
between ingredient composition and bitterness.

**But the effect is imprecise and practically nil.** Bootstrap CIs over recipes:

| Model | R² [95% CI] | MAE [95% CI] |
|---|---|---|
| Lasso 5D | +0.163 [−0.410, +0.322] | 0.861 [0.660, 1.095] |
| Ridge | +0.195 [−0.212, +0.321] | 0.833 [0.637, 1.068] |
| Ordinal | +0.216 [−0.203, +0.391] | 0.802 [0.606, 1.036] |
| log1p Ridge | +0.190 [−0.089, +0.332] | 0.761 [0.555, 1.016] |

Every R² interval includes zero. And against the constant baselines:

| Model | vs constant mean | vs constant **median** |
|---|---|---|
| Lasso 5D | ΔMAE −0.062 [−0.164, +0.039] — indistinguishable | +0.147 [−0.011, +0.290] — **indistinguishable** |
| Ridge | −0.091 [−0.179, −0.008] — model better | +0.118 [−0.026, +0.252] — **indistinguishable** |
| Ordinal | −0.121 [−0.220, −0.028] — model better | +0.088 [−0.060, +0.222] — **indistinguishable** |
| log1p Ridge | −0.162 [−0.249, −0.076] — model better | +0.047 [−0.072, +0.155] — **indistinguishable** |

No model is distinguishable from a constant `1.0`.

**The signal is largely two chocolate recipes.** Refitting with the two `bitter = 8`
recipes removed:

| Model | R² full | R² without the two | Skill full | Skill without |
|---|---|---|---|---|
| Lasso 5D | +0.163 | +0.047 | +0.187 | +0.074 |
| Ridge | +0.195 | +0.062 | +0.218 | +0.089 |
| Ordinal | +0.216 | +0.096 | +0.238 | +0.123 |
| log1p Ridge | +0.190 | +0.037 | +0.213 | +0.065 |

Between half and three-quarters of the apparent signal comes from 2 of 70 recipes.
Practically, the model has learned *"contains cocoa"*.

**Protocol control.** The identical harness on the other four dimensions:

| Dimension | R²(oos), Ridge | p (permutation) |
|---|---|---|
| Sweet | +0.687 | 0.001 |
| Umami | +0.411 | 0.001 |
| Sour | +0.388 | 0.001 |
| Salty | +0.320 | 0.001 |
| **Bitter** | **+0.195** | 0.001 |

The null result for bitterness is a property of the bitterness data, not of the
method — the same pipeline extracts 2–3.5× more variance elsewhere.

Out-of-sample R² for the same models across all five dimensions
(`results/e1_control_r2_by_dimension.csv`):

| Model | Sweet | Sour | Umami | Salty | **Bitter** |
|---|---|---|---|---|---|
| Constant (median) | −0.194 | −0.174 | −0.009 | −0.048 | −0.172 |
| Lasso 5D | +0.651 | +0.429 | +0.405 | +0.333 | **+0.163** |
| Hybrid HS+chem | +0.455 | +0.503 | +0.524 | +0.397 | **−0.389** |
| Ridge | +0.687 | +0.388 | +0.411 | +0.320 | **+0.195** |
| Random forest | +0.731 | +0.540 | +0.250 | +0.381 | **+0.074** |

The hybrid model is the best or near-best predictor on sour, umami and salty
(R² = 0.40–0.52) and the *worst* of any fitted model on bitterness (−0.389). It is
not a weak model; bitterness is a dimension on which no model can work.

## E3 — properties of the response itself

**Dynamic range.** Bitterness occupies 6 of the 101 available scale points; 60% of
recipes sit on the single value `1`. Its SD is 1.4% of the scale, against 10.9–17.7%
for the other four dimensions. Shannon entropy 1.74 bits vs 4.13–4.78 bits.

| Dimension | Mean | SD | Distinct levels | Modal share | Entropy (bits) |
|---|---|---|---|---|---|
| Sweet | 17.7 | 17.7 | 29 | 11% | 4.49 |
| Sour | 11.4 | 13.0 | 26 | 19% | 4.13 |
| Umami | 15.0 | 10.9 | 30 | 7% | 4.69 |
| Salty | 29.8 | 14.7 | 32 | 9% | 4.78 |
| **Bitter** | **1.6** | **1.4** | **6** | **60%** | **1.74** |

**Where the variance lives.** Two recipes — *Spread chocolade plain* and *Hot
chocolate from vending machine*, both scoring 8 — carry **62.1%** of the total
bitterness sum of squares (31.0% each). For the other dimensions the top-2 share is
10–29%.

**The gain is smaller than the panel's own resolution.** Scores are recorded as
integers, so one quantisation step is 1.0 scale point.

| | Value |
|---|---|
| Constant RMSE → best model RMSE | 1.399 → 1.221 |
| RMSE gain | **0.178 scale points** = **0.18 of one quantisation step** |
| RMSE gain as % of the 0–100 scale | **0.18%** |
| Best MAE gain over the constant mean | 0.209 (and *negative* against the constant median) |

The entire benefit of modelling bitterness is roughly one-fifth of the smallest
difference the panel is able to record.

**Exact agreement on the panel's integer grid.** Rounding LOO predictions back to
integers:

| Model | Exact hits | Within 1 unit | Distinct predictions |
|---|---|---|---|
| **Constant = 1.0** | **60.0%** | 84.3% | 1 |
| log1p Ridge | 50.0% | 85.7% | 5 |
| Ordinal | 47.1% | 91.4% | 5 |
| Lasso 5D | 42.9% | 88.6% | 6 |
| Constant = 1.57 (mean) | 17.1% | 88.6% | 2 |

A single constant reproduces the panel score exactly for 60% of recipes; no fitted
model reaches that.

**Power.** N = 70 has 92% power to detect the observed effect (R² = 0.216, f² = 0.28,
5 predictors, α = 0.05); 53 recipes would suffice. Under-powering is *not* the
problem — the effect is detectable, it is simply too small to matter.

**What bitterness can support: detection, not regression.**

| Task | n pos / n neg | LOO ROC AUC | LOO balanced acc. | Majority-class acc. |
|---|---|---|---|---|
| bitter ≥ 2 | 23 / 47 | **0.764** | 0.721 | 0.671 |
| bitter ≥ 3 | 11 / 59 | 0.696 | 0.646 | 0.843 |

Reframed as "is this recipe perceptibly bitter at all?", ingredient composition
carries real, usable information (AUC 0.76). This is the one framing in which the
bitterness data support a positive claim.

---

## Recommendation

The manuscript's current handling is already defensible — bitterness is excluded from
all aggregate metrics, its coefficients are not interpreted, and §3.1/§3.2/§4 all flag
the floor effect. The reviewer's objection is best answered not by changing the model
but by **replacing the assertion with the measurement**.

Concretely, three changes, in order of value:

1. **Add a supplementary table** (the E1 table above) and one paragraph reporting
   that 19 model structures — including Tobit, ordinal, Poisson, negative-binomial and
   variance-stabilising transforms specifically chosen for floor-censored responses —
   were tested, and none beats a constant. This converts "not evidence of model
   accuracy" from a caveat into a result.
2. **State the two quantitative facts that settle it**: two chocolate recipes carry
   62% of the bitterness variance, and the best possible RMSE gain over a constant is
   0.18 scale points — under one-fifth of the panel's 1-point resolution.
3. **Optionally add the detection result** (AUC 0.76 for bitter ≥ 2) as the
   constructive counterpart, showing what the bitterness data *can* support. This
   turns a purely defensive answer into a small positive contribution.

Note that bound *coverage* for bitterness (61% within bounds, 26% exceedance) is a
separate matter and remains valid: it requires no regression and is part of the
paper's central exceedance-fingerprint result. It should be retained.
