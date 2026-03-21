#!/usr/bin/env python3
"""Compute Reuss-Voigt and Hashin-Shtrikman bounds for recipe sensory data."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pprint
from typing import Any

import numpy as np

from env_config import get_raw_recipes_path


SENSORY_DIMENSIONS = ["sweet", "sour", "bitter", "umami", "salty"]
EPSILON = 1e-12
HS_APPROXIMATION_NOTE = (
    "For recipes with more than two ingredients, HS is approximated as a two-phase "
    "system that uses the total normalized fraction at the minimum and maximum "
    "ingredient sensory values. Intermediate ingredients are absorbed into the "
    "complementary matrix fraction in the closed-form HS expressions."
)

SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
HS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "hs_predictions.py")
RV_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "rv_predictions.py")
JSON_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "bounds_predictions.json")


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
    """Round finite floats for stable, readable exports."""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), digits)


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
    if total <= EPSILON:
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


def _stable_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Divide safely and return a finite fallback when the ratio is unstable."""
    if not math.isfinite(denominator) or abs(denominator) <= EPSILON:
        return fallback
    value = numerator / denominator
    if not math.isfinite(value):
        return fallback
    return float(value)


def compute_rv(E: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Compute Reuss and Voigt bounds for one sensory dimension."""
    E = np.asarray(E, dtype=float)
    v = np.asarray(v, dtype=float)

    if E.size == 0 or v.size == 0:
        return {"reuss": 0.0, "voigt": 0.0}

    active_mask = v > EPSILON
    if not np.any(active_mask):
        return {"reuss": 0.0, "voigt": 0.0}

    E_active = np.clip(E[active_mask], 0.0, None)
    v_active = v[active_mask]
    v_active = v_active / v_active.sum()

    voigt = _clip_score(float(np.dot(v_active, E_active)))

    if np.any(E_active <= EPSILON):
        reuss = 0.0
    else:
        denominator = float(np.sum(v_active / np.maximum(E_active, EPSILON)))
        reuss = 0.0 if denominator <= EPSILON else float(1.0 / denominator)

    reuss = _clip_score(reuss)
    reuss = min(reuss, voigt)
    return {"reuss": reuss, "voigt": voigt}


def compute_hs(E: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Compute two-phase HS bounds for one sensory dimension.

    For multi-ingredient recipes, this uses a documented approximation:
    the total normalized weight at the minimum and maximum ingredient values is
    retained explicitly, and all intermediate values are absorbed into the
    complementary matrix fraction.
    """
    E = np.asarray(E, dtype=float)
    v = np.asarray(v, dtype=float)

    if E.size == 0 or v.size == 0:
        return {"hs_lower": 0.0, "hs_upper": 0.0}

    active_mask = v > EPSILON
    if not np.any(active_mask):
        return {"hs_lower": 0.0, "hs_upper": 0.0}

    E_active = np.clip(E[active_mask], 0.0, None)
    v_active = v[active_mask]
    v_active = v_active / v_active.sum()

    t_max = float(np.max(E_active))
    t_min = float(np.min(E_active))

    if math.isclose(t_max, t_min, rel_tol=0.0, abs_tol=EPSILON):
        bound = _clip_score(t_max)
        return {"hs_lower": bound, "hs_upper": bound}

    v_max = float(v_active[np.isclose(E_active, t_max, rtol=0.0, atol=EPSILON)].sum())
    v_min = float(v_active[np.isclose(E_active, t_min, rtol=0.0, atol=EPSILON)].sum())

    term_lower = _stable_ratio(1.0, t_min - t_max, fallback=0.0) + _stable_ratio(1.0 - v_min, 3.0 * max(t_max, EPSILON))
    term_upper = _stable_ratio(1.0, t_max - t_min, fallback=0.0) + _stable_ratio(v_min, 3.0 * max(t_min, EPSILON))

    candidate_a = t_max + _stable_ratio(v_min, term_lower, fallback=0.0)
    candidate_b = t_min + _stable_ratio(1.0 - v_min, term_upper, fallback=0.0)

    candidates = np.array([candidate_a, candidate_b], dtype=float)
    candidates = np.nan_to_num(candidates, nan=t_min, posinf=t_max, neginf=t_min)
    candidates = np.clip(candidates, t_min, t_max)

    hs_lower = _clip_score(float(np.min(candidates)))
    hs_upper = _clip_score(float(np.max(candidates)))
    return {"hs_lower": hs_lower, "hs_upper": hs_upper}


def _actual_scores(recipe: dict[str, Any]) -> dict[str, float]:
    """Extract recipe-level target sensory scores in a fixed dimension order."""
    food_scores = recipe.get("food_sensory_scores", {})
    return {
        dimension: _clip_score(_coerce_float(food_scores.get(dimension, 0.0)))
        for dimension in SENSORY_DIMENSIONS
    }


def build_recipe_record(recipe: dict[str, Any]) -> dict[str, Any]:
    """Compute all requested bounds and Voigt prediction errors for one recipe."""
    ingredients = recipe.get("ingredients", [])
    weights = normalize_weights(ingredients)
    actual = _actual_scores(recipe)

    prediction: dict[str, float] = {}
    errors: dict[str, dict[str, float | None]] = {}
    bounds: dict[str, dict[str, float]] = {}

    for dimension in SENSORY_DIMENSIONS:
        scores = _extract_dimension_scores(ingredients, dimension)
        rv_bounds = compute_rv(scores, weights)
        hs_bounds = compute_hs(scores, weights)

        voigt_value = rv_bounds["voigt"]
        actual_value = actual[dimension]
        absolute_error = abs(voigt_value - actual_value)
        squared_error = (voigt_value - actual_value) ** 2

        bounds[dimension] = {
            "reuss": _round_float(rv_bounds["reuss"]),
            "voigt": _round_float(voigt_value),
            "hs_lower": _round_float(hs_bounds["hs_lower"]),
            "hs_upper": _round_float(hs_bounds["hs_upper"]),
        }
        prediction[dimension] = _round_float(voigt_value)
        errors[dimension] = {
            "absolute_error": _round_float(absolute_error),
            "squared_error": _round_float(squared_error),
        }

    return {
        "recipe_id": recipe.get("recipe_id"),
        "recipe_name": recipe.get("recipe_name"),
        "actual": {dimension: _round_float(value) for dimension, value in actual.items()},
        "prediction": prediction,
        "errors": errors,
        "bounds": bounds,
    }


def build_dataset_records(raw_recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute bounds for every recipe in the dataset."""
    return [build_recipe_record(recipe) for recipe in raw_recipes]


def _dataset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Voigt prediction error statistics across the dataset."""
    summary: dict[str, Any] = {
        "recipe_count": len(records),
        "point_prediction": "voigt",
        "error_metrics": {},
    }

    for dimension in SENSORY_DIMENSIONS:
        abs_errors = np.array(
            [record["errors"][dimension]["absolute_error"] for record in records],
            dtype=float,
        )
        sq_errors = np.array(
            [record["errors"][dimension]["squared_error"] for record in records],
            dtype=float,
        )
        summary["error_metrics"][dimension] = {
            "mae": _round_float(float(np.mean(abs_errors))) if abs_errors.size else None,
            "mse": _round_float(float(np.mean(sq_errors))) if sq_errors.size else None,
            "rmse": _round_float(float(np.sqrt(np.mean(sq_errors)))) if sq_errors.size else None,
        }

    return summary


def _payload(method_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the export payload for one bounds family."""
    return {
        "method": method_name,
        "point_prediction": "voigt",
        "hs_approximation": HS_APPROXIMATION_NOTE,
        "summary": _dataset_summary(records),
        "recipes": records,
    }


def _write_python_export(path: str, variable_name: str, payload: dict[str, Any]) -> None:
    """Write a Python module export for downstream research scripts."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Auto-generated by Bounds_project/src/compute_bounds.py\n")
        handle.write("# Point prediction uses the Voigt weighted mixture.\n")
        handle.write(f"{variable_name} = ")
        handle.write(pprint.pformat(payload, indent=4, width=120, sort_dicts=False))
        handle.write("\n")


def _write_json_export(path: str, payload: dict[str, Any]) -> None:
    """Write a JSON export with the same canonical record structure."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    """Run the standalone bounds generation pipeline."""
    raw_recipes_path = get_raw_recipes_path()
    raw_recipes = load_attr_from_py(raw_recipes_path, "raw_recipes")
    records = build_dataset_records(raw_recipes)

    hs_payload = _payload("HS", records)
    rv_payload = _payload("RV", records)
    combined_payload = {
        "data_source": os.path.abspath(raw_recipes_path),
        "point_prediction": "voigt",
        "hs_approximation": HS_APPROXIMATION_NOTE,
        "summary": _dataset_summary(records),
        "recipes": records,
    }

    _write_python_export(HS_OUTPUT_PATH, "hs_predictions", hs_payload)
    _write_python_export(RV_OUTPUT_PATH, "rv_predictions", rv_payload)
    _write_json_export(JSON_OUTPUT_PATH, combined_payload)

    print(f"[+] Loaded raw recipes from: {raw_recipes_path}")
    print(f"[+] Recipes processed: {len(records)}")
    print(f"[+] Wrote HS export: {HS_OUTPUT_PATH}")
    print(f"[+] Wrote RV export: {RV_OUTPUT_PATH}")
    print(f"[+] Wrote JSON export: {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
