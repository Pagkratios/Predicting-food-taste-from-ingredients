#!/usr/bin/env python3
"""Candidate models for the bitterness sensitivity study.

Every entry exposes the sklearn `fit`/`predict` interface and does ALL of its own
preprocessing and hyper-parameter selection inside `fit`, so it can be dropped
into a leave-one-out loop without leaking the held-out recipe.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import optimize, stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNetCV, LassoCV, LogisticRegression, PoissonRegressor, RidgeCV,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
ALPHAS = np.logspace(-3, 1, 30)      # same grid as Hybrid/src/hybrid_analysis.py
INNER_CV = 5


def _inner(n):
    return KFold(n_splits=min(INNER_CV, max(2, n)), shuffle=True, random_state=SEED)


# ══════════════════════════════════════════════════════════════════════════
# Trivial baselines
# ══════════════════════════════════════════════════════════════════════════
class ConstantMean(BaseEstimator, RegressorMixin):
    """Predict the training-fold mean. The honest floor for any regression."""

    def fit(self, X, y):
        self.c_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.c_)


class ConstantMedian(BaseEstimator, RegressorMixin):
    """Predict the training-fold median — the MAE-optimal constant."""

    def fit(self, X, y):
        self.c_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.c_)


class Passthrough(BaseEstimator, RegressorMixin):
    """Use a supplied unfitted physical prediction (HS midpoint / Voigt).

    The prediction is carried in column 0 of X, so it slots into the same LOO
    harness as everything else without any fitting.
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X, float)[:, 0]


# ══════════════════════════════════════════════════════════════════════════
# Censored / count / ordinal models — the "right structure for a floor" family
# ══════════════════════════════════════════════════════════════════════════
class TobitRegressor(BaseEstimator, RegressorMixin):
    """Type-I Tobit, left-censored at `lower`.

    Treats the recorded zeros as censored observations of a latent continuous
    bitterness that the panel could not resolve below the floor. Predictions are
    the expectation of the *observed* (censored) variable,
    E[y] = Phi(mu/s) * mu + s * phi(mu/s).
    """

    def __init__(self, lower=0.0, l2=1e-3):
        self.lower = lower
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), np.where(X.std(0) == 0, 1.0, X.std(0))
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu_) / self.sd_])
        cens = y <= self.lower + 1e-12

        def nll(theta):
            beta, log_s = theta[:-1], theta[-1]
            s = np.exp(np.clip(log_s, -6, 6))
            m = Z @ beta
            ll = 0.0
            if np.any(~cens):
                ll += np.sum(stats.norm.logpdf(y[~cens], m[~cens], s))
            if np.any(cens):
                ll += np.sum(stats.norm.logcdf((self.lower - m[cens]) / s))
            return -ll + self.l2 * np.sum(beta[1:] ** 2)

        theta0 = np.concatenate([[y.mean()], np.zeros(Z.shape[1] - 1), [np.log(y.std() + 1e-6)]])
        res = optimize.minimize(nll, theta0, method="L-BFGS-B")
        self.beta_, self.s_ = res.x[:-1], float(np.exp(np.clip(res.x[-1], -6, 6)))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu_) / self.sd_])
        m = Z @ self.beta_
        z = (m - self.lower) / self.s_
        return self.lower + stats.norm.cdf(z) * (m - self.lower) + self.s_ * stats.norm.pdf(z)


class OrdinalPO(BaseEstimator, RegressorMixin):
    """Proportional-odds ordinal logistic on the observed bitterness levels.

    Bitterness takes only 6 distinct values, so treating it as ordered categories
    respects the measurement far better than a continuous response. The point
    prediction is the posterior mean, sum_k p_k * level_k.
    """

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), np.where(X.std(0) == 0, 1.0, X.std(0))
        Xs = (X - self.mu_) / self.sd_
        self.levels_ = np.unique(y)
        K = len(self.levels_)
        rank = np.searchsorted(self.levels_, y)
        p = Xs.shape[1]

        def unpack(theta):
            beta = theta[:p]
            # cutpoints as first value + softplus increments, keeps them ordered
            cuts = np.concatenate([[theta[p]], theta[p] + np.cumsum(np.exp(theta[p + 1:]))])
            return beta, cuts

        def nll(theta):
            beta, cuts = unpack(theta)
            eta = Xs @ beta
            cdf = 1.0 / (1.0 + np.exp(-(cuts[None, :] - eta[:, None])))   # (n, K-1)
            cdf = np.hstack([np.zeros((len(Xs), 1)), cdf, np.ones((len(Xs), 1))])
            pr = np.clip(np.diff(cdf, axis=1), 1e-12, 1.0)
            return -np.sum(np.log(pr[np.arange(len(Xs)), rank])) + self.l2 * np.sum(beta ** 2)

        theta0 = np.concatenate([np.zeros(p), [-1.0], np.zeros(max(0, K - 2))])
        res = optimize.minimize(nll, theta0, method="L-BFGS-B")
        self.beta_, self.cuts_ = unpack(res.x)
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        eta = Xs @ self.beta_
        cdf = 1.0 / (1.0 + np.exp(-(self.cuts_[None, :] - eta[:, None])))
        cdf = np.hstack([np.zeros((len(Xs), 1)), cdf, np.ones((len(Xs), 1))])
        return np.clip(np.diff(cdf, axis=1), 1e-12, 1.0)

    def predict(self, X):
        return self.predict_proba(X) @ self.levels_


class MultinomialLevels(BaseEstimator, RegressorMixin):
    """Unordered multinomial logit over the observed levels; predicts the posterior mean."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.levels_ = np.unique(y)
        self.pipe_ = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=5000)),
        ]).fit(X, np.searchsorted(self.levels_, y))
        self.classes_ = self.pipe_["lr"].classes_
        return self

    def predict(self, X):
        P = self.pipe_.predict_proba(X)
        return P @ self.levels_[self.classes_]


class NegBinGLM(BaseEstimator, RegressorMixin):
    """Negative-binomial GLM (log link) — over-dispersed counts near a floor."""

    def __init__(self, alpha_nb=1.0, l2=1e-3):
        self.alpha_nb = alpha_nb
        self.l2 = l2

    def fit(self, X, y):
        import statsmodels.api as sm
        X = np.asarray(X, float)
        self.mu_, self.sd_ = X.mean(0), np.where(X.std(0) == 0, 1.0, X.std(0))
        Z = sm.add_constant((X - self.mu_) / self.sd_, has_constant="add")
        fam = sm.families.NegativeBinomial(alpha=self.alpha_nb)
        try:
            self.res_ = sm.GLM(y, Z, family=fam).fit_regularized(alpha=self.l2, L1_wt=0.0)
        except Exception:
            self.res_ = sm.GLM(y, Z, family=fam).fit()
        return self

    def predict(self, X):
        import statsmodels.api as sm
        X = np.asarray(X, float)
        Z = sm.add_constant((X - self.mu_) / self.sd_, has_constant="add")
        return np.asarray(self.res_.predict(Z), float)


class ShrunkToMean(BaseEstimator, RegressorMixin):
    """Convex blend of a base regressor with the training mean.

    lam is chosen by inner CV, so the model is free to collapse to the constant
    (lam -> 0) if the features carry nothing. Directly tests the reviewer's
    "statistically inefficient" claim: does optimal shrinkage keep the regression?
    """

    def __init__(self, base=None, lams=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0)):
        self.base = base
        self.lams = lams

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        errs = np.zeros(len(self.lams))
        cv = _inner(len(y))
        for tr, te in cv.split(X):
            m = clone(self.base).fit(X[tr], y[tr])
            p = m.predict(X[te])
            c = y[tr].mean()
            for i, lam in enumerate(self.lams):
                errs[i] += np.sum((y[te] - (lam * p + (1 - lam) * c)) ** 2)
        self.lam_ = self.lams[int(np.argmin(errs))]
        self.c_ = float(y.mean())
        self.model_ = clone(self.base).fit(X, y)
        return self

    def predict(self, X):
        return self.lam_ * self.model_.predict(X) + (1 - self.lam_) * self.c_


class Clipped(BaseEstimator, RegressorMixin):
    """Wrap a regressor and clip predictions into [lo, hi] (bitterness cannot be < 0)."""

    def __init__(self, base=None, lo=0.0, hi=100.0):
        self.base = base
        self.lo = lo
        self.hi = hi

    def fit(self, X, y):
        self.model_ = clone(self.base).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model_.predict(X), self.lo, self.hi)


# ══════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════
def _pipe(est):
    return Pipeline([("sc", StandardScaler()), ("m", est)])


def _lasso():
    return _pipe(LassoCV(alphas=ALPHAS, cv=INNER_CV, max_iter=20000, random_state=SEED))


def build_models():
    """name -> (factory, family, note). `family` groups rows in the output table."""
    m = {}

    # --- trivial baselines -------------------------------------------------
    m["Constant (mean)"] = (ConstantMean, "baseline",
                            "training-fold mean; no features")
    m["Constant (median)"] = (ConstantMedian, "baseline",
                              "training-fold median; MAE-optimal constant")

    # --- the manuscript's own structure -----------------------------------
    m["Lasso 5D"] = (_lasso, "as-published", "paper's Lasso on the 5-D Voigt vector")
    m["Hybrid HS+chem"] = (_lasso, "as-published", "paper's hybrid: HS mid + Voigt + 8 chem")
    m["Lasso per-ingredient"] = (_lasso, "as-published", "115 ingredient mass fractions")

    # --- alternative linear regularisation ---------------------------------
    m["Ridge"] = (lambda: _pipe(RidgeCV(alphas=np.logspace(-3, 4, 40))), "linear",
                  "L2 instead of L1")
    m["ElasticNet"] = (lambda: _pipe(ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], alphas=ALPHAS,
                                                  cv=INNER_CV, max_iter=20000,
                                                  random_state=SEED)),
                       "linear", "L1/L2 mix")

    # --- floor-appropriate response models ---------------------------------
    m["Poisson GLM"] = (lambda: _pipe(GridSearchCV(
        PoissonRegressor(max_iter=5000),
        {"alpha": np.logspace(-3, 2, 12)}, cv=INNER_CV,
        scoring="neg_mean_squared_error")), "floor-aware", "log link, count response")
    m["NegBinomial GLM"] = (NegBinGLM, "floor-aware", "over-dispersed count response")
    m["Tobit (censored at 0)"] = (TobitRegressor, "floor-aware",
                                  "zeros treated as left-censored")
    m["Ordinal (prop. odds)"] = (OrdinalPO, "floor-aware",
                                 "6 observed levels as ordered categories")
    m["Multinomial levels"] = (MultinomialLevels, "floor-aware",
                               "levels as unordered classes, posterior mean")
    m["sqrt-transform Ridge"] = (lambda: TransformedTargetRegressor(
        regressor=_pipe(RidgeCV(alphas=np.logspace(-3, 4, 40))),
        func=np.sqrt, inverse_func=np.square), "floor-aware",
        "variance-stabilising transform")
    m["log1p-transform Ridge"] = (lambda: TransformedTargetRegressor(
        regressor=_pipe(RidgeCV(alphas=np.logspace(-3, 4, 40))),
        func=np.log1p, inverse_func=np.expm1), "floor-aware",
        "variance-stabilising transform")
    m["Lasso + clip at 0"] = (lambda: Clipped(base=_lasso(), lo=0.0, hi=100.0),
                              "floor-aware", "respects the hard non-negativity floor")
    m["Lasso shrunk to mean"] = (lambda: ShrunkToMean(base=_lasso()), "floor-aware",
                                 "blend weight to the constant tuned by inner CV")

    # --- nonlinear ----------------------------------------------------------
    m["Random Forest"] = (lambda: RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        "nonlinear", "captures thresholds the linear model cannot")
    m["Gradient Boosting"] = (lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=SEED),
        "nonlinear", "shallow boosted stumps")
    m["k-NN"] = (lambda: _pipe(GridSearchCV(
        KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 10, 15]}, cv=INNER_CV,
        scoring="neg_mean_squared_error")), "nonlinear", "local averaging")

    return m


# Which feature block each model consumes.
FEATURE_BLOCK = {
    "Constant (mean)": "voigt5",
    "Constant (median)": "voigt5",
    "Lasso 5D": "voigt5",
    "Hybrid HS+chem": "hybrid",
    "Lasso per-ingredient": "ingredients",
    "Ridge": "voigt5",
    "ElasticNet": "voigt5",
    "Poisson GLM": "voigt5",
    "NegBinomial GLM": "voigt5",
    "Tobit (censored at 0)": "voigt5",
    "Ordinal (prop. odds)": "voigt5",
    "Multinomial levels": "voigt5",
    "sqrt-transform Ridge": "voigt5",
    "log1p-transform Ridge": "voigt5",
    "Lasso + clip at 0": "voigt5",
    "Lasso shrunk to mean": "voigt5",
    "Random Forest": "voigt5",
    "Gradient Boosting": "voigt5",
    "k-NN": "voigt5",
}
