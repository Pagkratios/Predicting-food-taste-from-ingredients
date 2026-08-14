# Response to Comment 2 — edited

Your original response, kept intact, with the new empirical evidence added.
_[Square brackets]_ mark cross-references to fill in.

---

**2) Questionable value of modeling bitterness:** *The ground-truth bitterness scores
have a mean of 1.6 (on a 0-100 scale), with 96% of recipes scoring ≤3, indicating a
severe floor effect. The authors themselves acknowledge that the bitterness result is
"not evidence of model accuracy." Under these circumstances, applying the same
regression structure to bitterness is statistically inefficient and may overfit noise.*

We agree that the bitterness dimension has limited statistical value in this dataset
because the ground-truth bitterness scores are near the perceptual floor, and that
applying the full regression structure to it is statistically inefficient. We revised
the Results to make explicit that bitterness is not treated as a predictive target. It
is retained in Table 1 for completeness, but its fitted coefficients are not
interpreted, and it is excluded from the aggregate performance metrics (Avg 4D,
Tables 1 and 3), so any inefficiency or overfitting on this near-constant dimension
cannot affect the reported results (Lines 174–181).

To substantiate this rather than assert it, we additionally tested the reviewer's
concern directly. We evaluated **19 model structures** for bitterness under a single
identical leave-one-out protocol (N = 70, all hyperparameters tuned strictly within
each training fold), including several chosen specifically because they are
appropriate for a floor-censored, low-count response: a Tobit model treating the zeros
as left-censored, a proportional-odds ordinal model over the six observed levels,
multinomial regression over levels, Poisson and negative-binomial GLMs, square-root
and log(1+x) variance-stabilising transforms, non-negativity clipping, and a
shrink-to-the-mean estimator with the blend weight tuned by inner cross-validation.
The reviewer is correct that the Gaussian-response Lasso is not the efficient choice —
the floor-appropriate structures improve on it by 7–12% in MAE. However, **none of the
19 structures outperforms a constant prediction of 1.0** (MAE 0.71, versus 0.76 for
the best model and 0.86 for the Lasso), and in a paired bootstrap over recipes no
model is distinguishable from that constant. This is not specific to the regression:
the φ-calibrated HS and RV predictors of Table 1 also fall below a constant on this
dimension (R² = −0.16 and −0.10), as does the hybrid model (R² = −0.39), even though
the hybrid model is our best or near-best predictor on sourness, umami and saltiness
(R² = 0.40–0.52). No model of any family, physical or learned, does useful work on
bitterness.

We can also now bound the "overfit noise" concern precisely. A permutation test in
which the bitterness labels were shuffled and the entire leave-one-out pipeline
refitted 1000 times rejects the null for every model family (p = 0.001–0.002), so the
association between composition and bitterness is genuine rather than fitted noise.
What is absent is not signal but *usable* signal: the best attainable RMSE improvement
over a constant is 0.178 points on the 0–100 scale — less than one fifth of the
1-point resolution at which the trained panel records scores — and two recipes
(*Spread chocolade plain* and *Hot chocolate from vending machine*, both scoring 8)
carry 62.1% of the total bitterness sum of squares, against 10–29% for the other four
dimensions. Removing those two recipes halves the apparent R². In effect the model
learns "this recipe contains cocoa," which does not generalise. Applying the identical
protocol to the other four dimensions yields R² = 0.32–0.69 (all p = 0.001),
confirming that this is a property of the bitterness data rather than of our method.

In revising these passages we also corrected an error in our own previous statement.
The submitted text compared against a constant prediction of 2.0 (MAE = 1.06) and
described it as "only marginally worse" than the model. 2.0 is not the MAE-optimal
constant; 1.0 is, and it achieves MAE = 0.71, which is *better* than the model rather
than worse. Both passages have been corrected, and the correction strengthens the
point the reviewer is making.

These analyses are reported in new Supplementary Table _[S#]_ (all 19 structures with
LOO MAE, RMSE, R² and skill relative to the constant baseline) and new Supplementary
Figure _[S#]_ (score distribution, model ranking, permutation null, and
predicted-versus-actual relation), and summarised in Lines _[###–###]_.

_[OPTIONAL — see note below]_ Finally, we note the one framing in which the bitterness
data do support a positive claim. Reframed as a detection task ("is this recipe
perceptibly bitter, i.e. score ≥ 2?"), ingredient composition is genuinely informative
(leave-one-out ROC AUC = 0.76, balanced accuracy 0.72 against a 0.67 majority-class
rate). We report this in Supplementary _[S#]_ as the appropriate use of this dimension.

---

## Notes

- **The last paragraph is optional.** It costs three sentences and turns a purely
  defensive answer into a small positive contribution. If you include it, add one line
  to Methods describing the logistic model and the LOO protocol. If you'd rather keep
  the response tight, drop it — nothing else depends on it.
- **Do not concede the bound-coverage result.** The reviewer is objecting to the
  regression, not to coverage. The 26% exceedance figure for bitterness needs no
  regression and is one of the five numbers in the paper's headline
  "quantitative fingerprint" claim.
- **Keep the permutation result paired with the magnitude result.** Stated alone, "the
  signal is real (p = 0.001)" invites a follow-up asking why you then excluded it.
  The two must travel together.
- If you need a shorter version for a length-capped response, use Version B in
  `RESPONSE_DRAFT.md`.
