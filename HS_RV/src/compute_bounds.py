#!/usr/bin/env python3
"""Compute Reuss-Voigt and Hashin-Shtrikman bounds for recipe sensory data."""

from __future__ import annotations

import importlib.util
import math
import os
import pprint
from typing import Any

import numpy as np

from env_config import get_raw_recipes_path


SENSORY_DIMENSIONS = ["sweet", "sour", "bitter", "umami", "salty"]
DISPLAY_ORDER = ["bitter", "salty", "sour", "sweet", "umami"]
DISPLAY_DIMENSIONS = {
    "sweet": "sweetness",
    "sour": "sourness",
    "bitter": "bitterness",
    "umami": "umami",
    "salty": "saltiness",
}
EPS = 1e-8
HS_NOTE = "Full N-phase Hashin-Shtrikman bounds computed over all ingredients per the exact specification."

SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
SHARED_DATA_DIR = os.path.join(REPO_ROOT, "data")
HS_OUTPUT_PATH = os.path.join(SHARED_DATA_DIR, "hs_predictions.py")
RV_OUTPUT_PATH = os.path.join(SHARED_DATA_DIR, "rv_predictions.py")


def load_attr_from_py(filepath: str, variable_name: str) -> Any:
    """Load a variable from a Python file without modifying sys.path."""
    spec = importlib.util.spec_from_file_location(variable_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, variable_name)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary numeric input into a finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _round_float(value: float | None, digits: int = 6) -> float | None:
    """Round finite floats for stable intermediate storage."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _round_int(value: float | None) -> int | None:
    """Round values for the human-readable legacy-style HS/RV exports."""
    if value is None or not math.isfinite(value):
        return None
    return int(round(float(value)))


def _clip_score(value: float) -> float:
    """Keep sensory values in the expected dataset range."""
    return float(np.clip(value, 0.0, 100.0))


def normalize_weights(ingredients: list[dict[str, Any]]) -> np.ndarray:
    """Return normalized non-negative ingredient weights."""
    if not ingredients:
        return np.zeros(0, dtype=float)

    weights = np.array([_coerce_float(ingredient.get("weight", 0.0)) for ingredient in ingredients], dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)

    total = float(weights.sum())
    if total <= EPS:
        return np.zeros_like(weights)
    return weights / total


def _extract_dimension_scores(ingredients: list[dict[str, Any]], dimension: str) -> np.ndarray:
    """Extract one sensory dimension across all ingredients as a non-negative vector."""
    values = []
    for ingredient in ingredients:
        sensory_scores = ingredient.get("sensory_scores", {})
        values.append(_coerce_float(sensory_scores.get(dimension, 0.0)))
    array = np.array(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(array, 0.0, None)


def compute_rv(T: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Compute Reuss (lower) and Voigt (upper) bounds for one sensory dimension.

    All N ingredient terms are included. T_i < EPS is clamped to EPS (not skipped).
    Weights v must be pre-normalized (sum to 1).
    """
    rv_upper = float(np.dot(v, T))
    rv_lower = 1.0 / float(np.sum(v / np.maximum(T, EPS)))
    return {"rv_lower": rv_lower, "rv_upper": rv_upper}


def compute_hs(T: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Compute full N-phase Hashin-Shtrikman bounds for one sensory dimension.

    Uses all N ingredient phases — no two-phase approximation.
    Weights v must be pre-normalized (sum to 1).
    """
    idx = np.argsort(T)
    T_s = T[idx]
    v_s = v[idx]

    T1 = float(T_s[0])
    TN = float(T_s[-1])

    alpha1 = 1.0 / (3.0 * max(T1, EPS))
    alphaN = 1.0 / (3.0 * max(TN, EPS))

    # A_1: sum over i=2..N (0-indexed: 1..N-1); clamp diff to >= EPS
    diffs_lo = np.maximum(T_s[1:] - T1, EPS)
    A1 = float(np.sum(v_s[1:] / (1.0 / diffs_lo + alpha1)))

    # A_N: sum over i=1..N-1 (0-indexed: 0..N-2); clamp diff to <= -EPS
    diffs_hi = np.minimum(T_s[:-1] - TN, -EPS)
    AN = float(np.sum(v_s[:-1] / (1.0 / diffs_hi + alphaN)))

    denom_lo = 1.0 - alpha1 * A1
    if abs(denom_lo) < EPS:
        denom_lo = EPS
    hs_lower = T1 + A1 / denom_lo

    denom_hi = 1.0 - alphaN * AN
    if abs(denom_hi) < EPS:
        denom_hi = EPS
    hs_upper = TN + AN / denom_hi

    return {"hs_lower": hs_lower, "hs_upper": hs_upper}


def _compute_taste_bounds(T: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Handle trivial cases then dispatch to compute_rv and compute_hs."""
    N = len(T)
    if N == 0:
        return {"rv_lower": 0.0, "rv_upper": 0.0, "hs_lower": 0.0, "hs_upper": 0.0}
    if N == 1:
        val = float(T[0])
        return {"rv_lower": val, "rv_upper": val, "hs_lower": val, "hs_upper": val}
    if float(np.max(T) - np.min(T)) < EPS:
        val = float(np.dot(v, T))
        return {"rv_lower": val, "rv_upper": val, "hs_lower": val, "hs_upper": val}
    rv = compute_rv(T, v)
    hs = compute_hs(T, v)
    return {**rv, **hs}


def _actual_scores(recipe: dict[str, Any]) -> dict[str, float]:
    """Extract recipe-level target sensory scores in a fixed dimension order."""
    food_scores = recipe.get("food_sensory_scores", {})
    return {
        dimension: _clip_score(_coerce_float(food_scores.get(dimension, 0.0)))
        for dimension in SENSORY_DIMENSIONS
    }


def build_recipe_record(recipe: dict[str, Any]) -> dict[str, Any]:
    """Compute canonical bounds for one recipe."""
    ingredients = recipe.get("ingredients", [])
    weights = normalize_weights(ingredients)
    actual = _actual_scores(recipe)

    bounds: dict[str, dict[str, float | None]] = {}
    for dimension in SENSORY_DIMENSIONS:
        scores = _extract_dimension_scores(ingredients, dimension)
        result = _compute_taste_bounds(scores, weights)
        rv_lo = result["rv_lower"]
        rv_hi = result["rv_upper"]
        hs_lo = result["hs_lower"]
        hs_hi = result["hs_upper"]

        # Step 7: log ordering violations without modifying results
        if not (rv_lo <= hs_lo <= hs_hi <= rv_hi):
            recipe_id = recipe.get("recipe_id", "?")
            print(
                f"[WARN] ordering violated — recipe={recipe_id} dim={dimension}: "
                f"rv_lower={rv_lo:.6g} hs_lower={hs_lo:.6g} "
                f"hs_upper={hs_hi:.6g} rv_upper={rv_hi:.6g}"
            )

        bounds[dimension] = {
            "rv_lower": _round_float(rv_lo),
            "rv_upper": _round_float(rv_hi),
            "hs_lower": _round_float(hs_lo),
            "hs_upper": _round_float(hs_hi),
        }

    return {
        "recipe_id": recipe.get("recipe_id"),
        "recipe_name": recipe.get("recipe_name"),
        "actual": {dimension: _round_float(value) for dimension, value in actual.items()},
        "bounds": bounds,
    }


def build_dataset_records(raw_recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute bounds for every recipe in the dataset."""
    return [build_recipe_record(recipe) for recipe in raw_recipes]


def _recipe_label(record: dict[str, Any]) -> str:
    """Create the readable recipe key used in exported HS/RV files."""
    recipe_id = record.get("recipe_id") or "UNKNOWN"
    recipe_name = record.get("recipe_name") or "Unnamed recipe"
    return f"{recipe_id} - {recipe_name}"


def _least_squares_phi(lowers: list[float], uppers: list[float], actuals: list[float]) -> float:
    """Closed-form least-squares φ clipped to [0, 1].

    φ* = Σ(Δr · (ar - T⁻r)) / Σ(Δr²)
    where Δr = T⁺r - T⁻r (bound gap), ar = actual, T⁻r = lower bound.
    """
    num = 0.0
    den = 0.0
    for lo, hi, a in zip(lowers, uppers, actuals):
        delta = hi - lo
        num += delta * (a - lo)
        den += delta * delta
    if den < EPS:
        return 0.5
    return max(0.0, min(1.0, num / den))


def _loo_calibrated_predictions(
    lowers: list[float], uppers: list[float], actuals: list[float]
) -> list[float]:
    """LOO calibrated predictions: train φ on N-1 recipes, predict on the held-out one."""
    n = len(lowers)
    predictions = []
    for i in range(n):
        loo_lowers = lowers[:i] + lowers[i + 1 :]
        loo_uppers = uppers[:i] + uppers[i + 1 :]
        loo_actuals = actuals[:i] + actuals[i + 1 :]
        phi = _least_squares_phi(loo_lowers, loo_uppers, loo_actuals)
        predictions.append(lowers[i] + phi * (uppers[i] - lowers[i]))
    return predictions


def _pearson_r(ys: list[float], preds: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(ys)
    mean_y = sum(ys) / n
    mean_p = sum(preds) / n
    num = sum((y - mean_y) * (p - mean_p) for y, p in zip(ys, preds))
    den = math.sqrt(
        sum((y - mean_y) ** 2 for y in ys) * sum((p - mean_p) ** 2 for p in preds)
    )
    return 0.0 if den < EPS else num / den


def _r_squared(ys: list[float], preds: list[float]) -> float:
    """Coefficient of determination R²."""
    mean_y = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return 0.0 if ss_tot < EPS else 1.0 - ss_res / ss_tot


def _compute_calibration(
    records: list[dict], lower_key: str, upper_key: str
) -> dict[str, dict]:
    """Compute LOO predictions and final φ for each taste dimension."""
    result: dict[str, dict] = {}
    for dim in SENSORY_DIMENSIONS:
        lowers = [float(record["bounds"][dim][lower_key] or 0.0) for record in records]
        uppers = [float(record["bounds"][dim][upper_key] or 0.0) for record in records]
        actuals = [float(record["actual"][dim] or 0.0) for record in records]
        result[dim] = {
            "loo_preds": _loo_calibrated_predictions(lowers, uppers, actuals),
            "phi": _least_squares_phi(lowers, uppers, actuals),
        }
    return result


def _report_calibration(
    records: list[dict], calibration: dict[str, dict], method_name: str
) -> None:
    """Print final φ, PCC, and R² per taste dimension."""
    n = len(records)
    print(f"\n[{method_name}] Final calibrated φ (fit on all {n} recipes):")
    for dim in SENSORY_DIMENSIONS:
        cal = calibration[dim]
        actuals = [float(record["actual"][dim] or 0.0) for record in records]
        pcc = _pearson_r(actuals, cal["loo_preds"])
        r2 = _r_squared(actuals, cal["loo_preds"])
        print(f"  {dim:8s}: φ={cal['phi']:.4f}  PCC={pcc:.4f}  R²={r2:.4f}")


def _display_scores(values: dict[str, float | None]) -> dict[str, int | None]:
    """Convert canonical sensory names into readable legacy display names."""
    return {
        DISPLAY_DIMENSIONS[dimension]: _round_int(values.get(dimension))
        for dimension in DISPLAY_ORDER
    }


def _legacy_hs_export(
    records: list[dict[str, Any]], calibration: dict[str, dict]
) -> dict[str, Any]:
    """Build the readable HS export with LOO-calibrated predictions keyed by recipe label."""
    payload: dict[str, Any] = {}
    for i, record in enumerate(records):
        lower = {d: record["bounds"][d]["hs_lower"] for d in SENSORY_DIMENSIONS}
        upper = {d: record["bounds"][d]["hs_upper"] for d in SENSORY_DIMENSIONS}
        prediction = {d: calibration[d]["loo_preds"][i] for d in SENSORY_DIMENSIONS}
        payload[_recipe_label(record)] = {
            "Actual": _display_scores(record["actual"]),
            "HS lower bound": _display_scores(lower),
            "HS prediction": _display_scores(prediction),
            "HS upper bound": _display_scores(upper),
        }
    return payload


def _legacy_rv_export(
    records: list[dict[str, Any]], calibration: dict[str, dict]
) -> dict[str, Any]:
    """Build the readable RV export with LOO-calibrated predictions keyed by recipe label."""
    payload: dict[str, Any] = {}
    for i, record in enumerate(records):
        lower = {d: record["bounds"][d]["rv_lower"] for d in SENSORY_DIMENSIONS}
        upper = {d: record["bounds"][d]["rv_upper"] for d in SENSORY_DIMENSIONS}
        prediction = {d: calibration[d]["loo_preds"][i] for d in SENSORY_DIMENSIONS}
        payload[_recipe_label(record)] = {
            "Actual": _display_scores(record["actual"]),
            "RV lower bound": _display_scores(lower),
            "RV prediction": _display_scores(prediction),
            "RV upper bound": _display_scores(upper),
        }
    return payload


def _write_python_export(path: str, variable_name: str, payload: dict[str, Any]) -> None:
    """Write a Python module export for downstream research scripts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Auto-generated by Bounds_project/src/compute_bounds.py\n")
        handle.write(f"# {HS_NOTE}\n")
        handle.write(f"{variable_name} = ")
        handle.write(pprint.pformat(payload, indent=4, width=120, sort_dicts=False))
        handle.write("\n")


def main() -> None:
    """Run the standalone bounds generation pipeline."""
    raw_recipes_path = get_raw_recipes_path()
    raw_recipes = load_attr_from_py(raw_recipes_path, "raw_recipes")
    records = build_dataset_records(raw_recipes)

    hs_calibration = _compute_calibration(records, "hs_lower", "hs_upper")
    rv_calibration = _compute_calibration(records, "rv_lower", "rv_upper")

    _write_python_export(HS_OUTPUT_PATH, "hs_predictions", _legacy_hs_export(records, hs_calibration))
    _write_python_export(RV_OUTPUT_PATH, "rv_predictions", _legacy_rv_export(records, rv_calibration))

    _report_calibration(records, hs_calibration, "HS")
    _report_calibration(records, rv_calibration, "RV")

    print(f"\n[+] Loaded raw recipes from: {raw_recipes_path}")
    print(f"[+] Recipes processed: {len(records)}")
    print(f"[+] Wrote HS export: {HS_OUTPUT_PATH}")
    print(f"[+] Wrote RV export: {RV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
