#!/usr/bin/env python3

import os
import importlib.util
import numpy as np
from sklearn.preprocessing import StandardScaler
from env_config import get_raw_recipes_path
from plot_config import SENSORY_ORDER, ALIASES

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)

# Column order for X_train.npy / Y_train.npy. This MUST stay identical to the
# order train.py uses to index those arrays, so it is sourced from the single
# shared definition in plot_config rather than redeclared here.
SENSORY_KEYS = list(SENSORY_ORDER)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_recipes(path):
    """Dynamically load the raw_recipes list from a Python module."""
    spec = importlib.util.spec_from_file_location("raw_recipes_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.raw_recipes


def normalize_weights(ingredients):
    """Normalize ingredient weights so they sum to 1."""
    total = sum(ing['weight'] for ing in ingredients)
    return [{
        'name': ing['name'],
        'weight': ing['weight'] / total if total > 0 else 0.0,
        'sensory_scores': ing['sensory_scores']
    } for ing in ingredients]


def compute_input_vector(ingredients):
    """Compute the weighted sensory feature vector for a recipe."""
    vec = np.zeros(len(SENSORY_KEYS), dtype=float)
    for ing in ingredients:
        w = ing['weight']
        s = ing['sensory_scores']
        scores = np.array([float(s[ALIASES.get(k, k)]) for k in SENSORY_KEYS], dtype=float)
        vec += w * scores
    return vec


def compute_target_vector(recipe_scores):
    """Convert the recipe's true sensory scores into a vector."""
    return np.array([float(recipe_scores[ALIASES.get(k, k)]) for k in SENSORY_KEYS], dtype=float)


def standardize_features(X):
    """
    Standardize only the feature matrix X.
    Save scaler parameters for later inverse-transform if needed.
    """
    x_scaler = StandardScaler().fit(X)
    X_std = x_scaler.transform(X)

    np.save(os.path.join(OUTPUT_DIR, "x_scaler_mean.npy"), x_scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "x_scaler_scale.npy"), x_scaler.scale_)

    return X_std


def main():
    raw_data_path = get_raw_recipes_path()
    print(f"[INFO] Loading raw recipes from: {raw_data_path}")
    raw_recipes = load_raw_recipes(raw_data_path)

    X_list, Y_list = [], []
    for recipe in raw_recipes:
        ingredients = normalize_weights(recipe['ingredients'])
        X_list.append(compute_input_vector(ingredients))
        Y_list.append(compute_target_vector(recipe['food_sensory_scores']))

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    print(f"[INFO] Total recipes loaded: {X.shape[0]}")
    print("[INFO] Standardizing features (X only), keeping targets (Y) in original scale")
    X_std = standardize_features(X)

    # Save processed arrays for training
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_std)
    np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y)

    print(f"[✓] Preprocessing complete. Files saved under '{OUTPUT_DIR}'")
    print(f"    X_train.npy shape: {X_std.shape}")
    print(f"    Y_train.npy shape: {Y.shape}")
    print(f"    Y range: min={Y.min():.2f}, max={Y.max():.2f}")


if __name__ == "__main__":
    main()
