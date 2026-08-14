#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the defects fixed in this pass.

Each test pins an invariant that was previously violated, so the specific bug
cannot silently return. Run from the repo root:

    python3 -m pytest tests/ -v
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "Lasso", "src"))
sys.path.insert(0, os.path.join(REPO, "HS_RV", "src"))
sys.path.insert(0, os.path.join(REPO, "Hybrid", "src"))


def _load(path, attr):
    spec = importlib.util.spec_from_file_location(attr, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


@pytest.fixture(scope="module")
def raw_recipes():
    return _load(os.path.join(REPO, "data", "raw_recipes.py"), "raw_recipes")


# ── Fix #1: sensory column ordering ──────────────────────────────────────

def test_preprocess_and_plotconfig_agree_on_column_order():
    """preprocess.py must write columns in plot_config.SENSORY_ORDER.

    These were ['sweet','bitter','sour','umami','salty'] and
    ['sweet','bitter','salty','umami','sour'] respectively, which transposed
    salty and sour in every array train.py consumed.
    """
    import preprocess
    from plot_config import SENSORY_ORDER

    assert list(preprocess.SENSORY_KEYS) == list(SENSORY_ORDER)


def test_saved_targets_match_raw_recipes_by_name(raw_recipes):
    """Y_train.npy column j must hold SENSORY_ORDER[j] for every recipe."""
    from plot_config import SENSORY_ORDER

    y_path = os.path.join(REPO, "Lasso", "data", "processed", "Y_train.npy")
    if not os.path.exists(y_path):
        pytest.skip("run Lasso/src/preprocess.py first")

    Y = np.load(y_path)
    expected = np.array([
        [float(r["food_sensory_scores"][k]) for k in SENSORY_ORDER]
        for r in raw_recipes
    ])
    assert Y.shape == expected.shape
    np.testing.assert_allclose(Y, expected)


def test_target_means_are_not_transposed(raw_recipes):
    """Guard the specific symptom: salt and sour means must not be swapped."""
    from plot_config import SENSORY_ORDER

    y_path = os.path.join(REPO, "Lasso", "data", "processed", "Y_train.npy")
    if not os.path.exists(y_path):
        pytest.skip("run Lasso/src/preprocess.py first")

    Y = np.load(y_path)
    means = dict(zip(SENSORY_ORDER, Y.mean(axis=0)))
    # Salt is by far the highest-intensity dimension in this corpus; sour is low.
    assert means["salty"] > means["sour"] + 10.0


# ── Fix #6: conc_factor monotonicity ─────────────────────────────────────

def test_conc_factor_is_monotone_in_water_fraction():
    """conc_factor must never decrease as water fraction rises.

    The old `1/(1-water) if water < 0.9 else 1.0` mapped the most water-rich
    recipe in the corpus (0.97) to the smallest possible value.
    """
    from hybrid_analysis import chemistry_features

    def synth(water):
        return {"ingredients": [
            {"name": "Water", "weight": water, "sensory_scores": {}},
            {"name": "Sugar granulated", "weight": 1.0 - water, "sensory_scores": {}},
        ]}

    fracs = [0.0, 0.25, 0.5, 0.75, 0.85, 0.89, 0.9, 0.95, 0.99]
    concs = [chemistry_features(synth(w))["conc_factor"] for w in fracs]
    assert all(b >= a - 1e-9 for a, b in zip(concs, concs[1:])), concs
    assert concs[-1] == pytest.approx(10.0)


def test_no_duplicate_hybrid_analysis_module():
    """hybrid_analysis must exist only under Hybrid/src."""
    assert not os.path.exists(os.path.join(REPO, "Lasso", "src", "hybrid_analysis.py"))
    assert os.path.exists(os.path.join(REPO, "Hybrid", "src", "hybrid_analysis.py"))


# ── Fix #4: Lasso evaluation must be out-of-sample ───────────────────────

def test_loo_predictions_are_out_of_sample():
    """A held-out row must never influence its own prediction.

    Perturbing y at index i changes an in-sample fit's prediction at i, but must
    leave a leave-one-out prediction at i untouched.
    """
    from train import loo_predict_lasso

    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 5))
    y = X @ np.array([3.0, -2.0, 1.0, 0.0, 0.5]) + rng.normal(scale=0.5, size=25) + 30.0
    alphas = np.logspace(-3, 0, 4)

    base = loo_predict_lasso(X, y, alphas, inner_splits=3)

    y2 = y.copy()
    y2[7] += 100.0
    bumped = loo_predict_lasso(X, y2, alphas, inner_splits=3)

    assert bumped[7] == pytest.approx(base[7], abs=1e-9), (
        "prediction for row 7 moved when only row 7's target changed — the fit "
        "is seeing the held-out row"
    )


def test_composite_figure_does_not_read_insample_csv():
    """Figure 4's Lasso panel must not come from train.py's in-sample CSV."""
    src = open(os.path.join(REPO, "Lasso", "src", "composite_figure.py"),
               encoding="utf-8").read()
    # Ignore prose; only executable references to the in-sample CSV matter.
    code = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "LASSO_CSV_FILE" not in code
    assert "_load_lasso_csv" not in code
    assert "_load_loo_models" in code


# ── Fix #2: inverse-design constraints ───────────────────────────────────

def test_inverse_design_respects_bounds_after_normalisation():
    """Optimised mass fractions must satisfy their bounds and sum to 1."""
    import json

    path = os.path.join(REPO, "results", "inverse", "inverse_results.json")
    if not os.path.exists(path):
        pytest.skip("run Inverse/src/inverse_design.py first")

    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    assert cases, "no inverse-design cases recorded"
    for case in cases:
        opts = np.array([g["opt"] for g in case["ingredients"]])
        los = np.array([g["lo"] for g in case["ingredients"]])
        his = np.array([g["hi"] for g in case["ingredients"]])

        assert opts.sum() == pytest.approx(1.0, abs=1e-6), case["case"]
        assert (opts >= los - 1e-6).all(), f"{case['case']}: below lower bound"
        assert (opts <= his + 1e-6).all(), f"{case['case']}: above upper bound"


def test_renormalising_a_boxed_vector_can_breach_its_bounds():
    """Documents why the constraint is required rather than renormalising."""
    v = np.array([0.30, 0.10, 0.05, 0.10, 0.05, 0.10, 0.05])
    cap = 0.35
    assert v[0] <= cap
    assert (v / v.sum())[0] > cap


# ── Cross-module consistency ─────────────────────────────────────────────

def test_hs_implementations_agree(raw_recipes):
    """compute_bounds.compute_hs and the A(T0) form must match."""
    from compute_bounds import compute_hs

    def hs_aux(T, v, T0):
        return 1.0 / np.sum(v / (T + 2 * T0)) - 2 * T0

    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(2, 7))
        T = rng.uniform(1.0, 95.0, n)
        v = rng.dirichlet(np.ones(n))
        got = compute_hs(T, v)
        assert got["hs_lower"] == pytest.approx(hs_aux(T, v, T.min()), rel=1e-9)
        assert got["hs_upper"] == pytest.approx(hs_aux(T, v, T.max()), rel=1e-9)


def test_bounds_ordering_holds(raw_recipes):
    """RV lower <= HS lower <= HS upper <= RV upper for every recipe/dimension."""
    from compute_bounds import build_dataset_records

    for rec in build_dataset_records(raw_recipes):
        for dim, b in rec["bounds"].items():
            assert b["rv_lower"] <= b["hs_lower"] + 1e-6, (rec["recipe_id"], dim)
            assert b["hs_lower"] <= b["hs_upper"] + 1e-6, (rec["recipe_id"], dim)
            assert b["hs_upper"] <= b["rv_upper"] + 1e-6, (rec["recipe_id"], dim)
