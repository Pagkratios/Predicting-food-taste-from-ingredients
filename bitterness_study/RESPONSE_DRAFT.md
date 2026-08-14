# Draft response to reviewer — bitterness

Two versions below. **Version A** is the full response; **Version B** is a compressed
form if the journal caps response length. Both are backed by `SUMMARY.md` and
reproducible from `src/`.

Text in _[square brackets]_ needs a decision or a cross-reference filled in.

---

## Version A — full response

> **Comment.** *Questionable value of modeling bitterness: The ground-truth bitterness
> scores have a mean of 1.6 (on a 0–100 scale), with 96% of recipes scoring ≤3,
> indicating a severe floor effect. The authors themselves acknowledge that the
> bitterness result is "not evidence of model accuracy." Under these circumstances,
> applying the same regression structure to bitterness is statistically inefficient
> and may overfit noise.*

**Response.** We agree with the reviewer, and we thank them for pushing us to
substantiate this rather than assert it. In revision we have replaced our qualitative
caveat with a direct empirical test, and we have added a new supplementary analysis
(new Supplementary Table _[S#]_, new Supplementary Figure _[S#]_, new §_[x]_) that
reports it. We summarise the four findings.

**(1) We tested whether a different model structure helps.** We evaluated 19 model
structures for bitterness under a single identical leave-one-out protocol (N = 70,
all hyper-parameters tuned strictly within each training fold). These included
structures chosen specifically because they are appropriate for a floor-censored,
low-count response: a Tobit model treating the zeros as left-censored, a
proportional-odds ordinal model over the six observed levels, a multinomial model
over levels, Poisson and negative-binomial GLMs, square-root and log(1+x)
variance-stabilising transforms, non-negativity clipping, and a shrink-to-the-mean
estimator whose blend weight was tuned by inner cross-validation. We also tested
ridge and elastic-net shrinkage and three nonlinear learners.

The reviewer is correct that the Gaussian-response Lasso is not the efficient choice:
the floor-appropriate structures do improve on it, by 7–12% in MAE
(log(1+x)-transformed ridge, MAE = 0.76; proportional-odds ordinal, MAE = 0.80;
Lasso 5D, MAE = 0.86). However, **none of the 19 structures outperforms a constant
prediction of 1.0**, which achieves MAE = 0.71. In a paired bootstrap over recipes,
no model is distinguishable from that constant (all 95% CIs on ΔMAE contain zero).
Rounded back onto the panel's integer grid, the constant reproduces the recorded
score exactly for 60% of recipes, which no fitted model matches (43–50%).

In the course of this analysis we identified an error in our own previous statement,
which we have corrected: §3.1 and §3.2 of the submitted manuscript compared against a
constant of 2.0 (MAE = 1.06) and described it as "only marginally worse" than the
model. 2.0 is not the MAE-optimal constant; 1.0 is, and it achieves MAE = 0.71, which
is *better* than the model rather than worse. We have corrected both passages. The
correction strengthens rather than weakens the point the reviewer is making.

**(2) The models are not, however, fitting noise.** We ran a permutation test in
which the bitterness labels were shuffled and the entire leave-one-out pipeline
re-fitted, 1000 times. The observed out-of-sample R² lies far outside the
permutation null for every model family (Lasso 5D R² = 0.16, p = 0.001; ordinal
R² = 0.22, p = 0.002; null means −0.04 to −0.08). There is a genuine association
between ingredient composition and bitterness. What is absent is not signal but
*useful* signal, which brings us to the third point.

**(3) The effect is real but far too small to be meaningful, and is carried by two
recipes.** The best attainable RMSE improvement over a constant is 0.178 points on
the 0–100 scale — 0.18% of the scale, and under one-fifth of the 1-point quantisation
step at which the trained panel records scores. Furthermore, two recipes (*Spread
chocolade plain* and *Hot chocolate from vending machine*, both scoring 8) carry
62.1% of the total bitterness sum of squares; for the other four dimensions the
corresponding top-2 share is 10–29%. Removing those two recipes halves the apparent
R² (0.216 → 0.096 for the ordinal model, 0.163 → 0.047 for the Lasso). In effect the
model has learned "this recipe contains cocoa", which is not a finding that
generalises.

We note that this is a property of the bitterness data and not of our protocol: the
identical harness applied to the other four dimensions yields R² = 0.32–0.69
(all p = 0.001), 2–3.5× the bitterness value.

**(4) Where the reviewer's overfitting concern does land.** The hybrid model performs
*worse than a constant* on bitterness (R² = −0.39, MSE skill −0.35), consistent with
the negative PCC of −0.10 already reported for this dimension in Table 1. This is the
clearest possible demonstration of the reviewer's point and is precisely why
bitterness is excluded from every aggregate metric we report. We emphasise that this
is specific to bitterness and not a deficiency of the hybrid model: on sour, umami and
salty the same model is the best or near-best predictor we tested (R² = 0.40–0.52).
The same is true of the physics-based predictors — the φ-calibrated HS and RV
predictors reported in Table 1 also fall below a constant on bitterness
(R² = −0.16 and −0.10). No model of any family, physical or learned, can do useful
work on this dimension.

**Changes made.** Bitterness was already excluded from all aggregate metrics, its
coefficients were not interpreted, and the floor effect was flagged in §3.1, §3.2 and
§4. We have strengthened this in revision:

- Added Supplementary Table _[S#]_ reporting all 19 model structures with LOO MAE,
  RMSE, R² and skill relative to the constant baseline, and Supplementary Figure
  _[S#]_ showing the distribution, the model ranking, the permutation null and the
  predicted-vs-actual relation.
- Added a new paragraph to §_[3.2 / 4]_ reporting the permutation test, the two-recipe
  variance concentration, and the sub-quantisation-step magnitude of the achievable
  gain, replacing the previous single-sentence caveat.
- Amended §_[3.1]_ to state explicitly that bitterness is reported for completeness of
  the bound-coverage analysis only and is **not** modelled as a predictive target.
- Corrected the constant-baseline comparison in §3.1 and §3.2 from a constant of 2.0
  (MAE 1.06, described as "marginally worse") to the MAE-optimal constant of 1.0
  (MAE 0.71, which is better than the model).
- _[Optional — see note below]_ Added the detection result: reframed as a
  classification task ("is this recipe perceptibly bitter, i.e. score ≥ 2?"),
  ingredient composition is genuinely informative (LOO ROC AUC = 0.76, balanced
  accuracy 0.72 against a 0.67 majority-class rate). We report this as the framing in
  which the bitterness data do support a positive claim.

We have retained the bitterness **bound-coverage** result (61% of values within the HS
bounds, 26% exceedance). That result requires no regression, is part of the paper's
central exceedance-fingerprint finding, and is unaffected by the floor effect
concern — indeed the low exceedance rate for bitterness is itself informative about
the limited role of processing chemistry in this dimension.

---

## Version B — compressed

**Response.** We agree, and we have replaced our qualitative caveat with a direct
test. We evaluated 19 model structures for bitterness under one identical
leave-one-out protocol, including several chosen specifically for a floor-censored
response (Tobit with left-censoring at zero, proportional-odds ordinal over the six
observed levels, Poisson and negative-binomial GLMs, √ and log(1+x) transforms,
non-negativity clipping, and CV-tuned shrinkage toward the mean).

The reviewer is right that the Gaussian Lasso is the inefficient choice — the
floor-appropriate models beat it by 7–12% MAE. But **no structure beats a constant
prediction of 1.0** (MAE 0.71 vs 0.76 for the best model and 0.86 for the published
Lasso); in a paired bootstrap none is distinguishable from that constant. The
reviewer's second concern, overfitting, we can now bound precisely: a 1000-shuffle
permutation test rejects the null for every model family (p = 0.001–0.002), so the
association is real — but the best attainable RMSE gain over a constant is 0.178
points on the 0–100 scale, less than one-fifth of the panel's own 1-point recording
resolution, and two chocolate recipes carry 62% of the total bitterness variance
(removing them halves R²). The same protocol on the other four dimensions gives
R² = 0.32–0.69, confirming this is a property of the bitterness data rather than the
method. Consistently, our hybrid model is *worse* than a constant on bitterness
(R² = −0.39), which is why bitterness is excluded from every aggregate metric.

We have added Supplementary Table _[S#]_ and Figure _[S#]_ reporting this analysis,
amended §_[3.1]_ to state that bitterness is retained for completeness of the
bound-coverage analysis and is not modelled as a predictive target, and _[optionally]_
added the one framing the data do support: as a detection task ("score ≥ 2"),
ingredient composition gives LOO ROC AUC = 0.76.

---

## Notes for the authors

- **Fix the constant-baseline sentence before resubmitting, whatever else you do.**
  §3.1 and §3.2 currently compare against a constant of 2.0 and call it "only
  marginally worse" than the model. The MAE-optimal constant is 1.0 and it *beats*
  the model (0.71 vs 0.92). This is checkable in one line and it is better to correct
  it yourselves than to have the reviewer find it.

- **The detection result is optional but recommended.** It costs two sentences and
  converts a purely defensive answer into a small positive contribution. If included,
  it needs a line in Methods describing the logistic model and the LOO protocol.
- **Do not concede the bound-coverage result.** The reviewer's comment is about the
  regression, not about coverage. The 26% exceedance figure for bitterness is one of
  the five numbers in the paper's headline "quantitative fingerprint" claim.
- **Be careful not to overclaim the permutation result.** It shows the association is
  not noise; it does *not* show the model is useful. Points (2) and (3) must be stated
  together or the response invites a follow-up objection.
- Numbers to check against the final run before submission:
  `results/e1_bitter_table.md`, `results/e2_permutation_bitter.csv`,
  `results/e3_diagnostics.json`.
