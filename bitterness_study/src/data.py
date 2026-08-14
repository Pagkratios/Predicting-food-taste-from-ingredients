#!/usr/bin/env python3
"""Feature/target construction for the bitterness sensitivity study.

READ-ONLY with respect to the rest of the repo. The chemistry-proxy features and
the Hashin-Shtrikman bounds are imported from `Hybrid/src/hybrid_analysis.py` so
this study cannot drift from the numbers reported in the manuscript. Nothing here
writes outside `bitterness_study/`.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
STUDY = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS = os.path.join(STUDY, "results")
FIGURES = os.path.join(STUDY, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# hybrid_analysis imports `compute_bounds` from HS_RV/src, which in turn imports
# `env_config` from the same directory.
sys.path.insert(0, os.path.join(REPO, "HS_RV", "src"))


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hybrid = _load_module(os.path.join(REPO, "Hybrid", "src", "hybrid_analysis.py"),
                       "bitterness_study_hybrid")

TASTES = list(_hybrid.TASTES)                     # sweet, sour, bitter, umami, salty
CHEM_COLS = ['protein_frac', 'sugar_frac', 'maillard_potential', 'salt_frac',
             'water_frac', 'conc_factor', 'allium_frac', 'fermented_frac']


class Dataset:
    """Every feature block the manuscript uses, built once."""

    def __init__(self):
        raw = _hybrid.load_attr_from_py(os.path.join(REPO, "data", "raw_recipes.py"),
                                        "raw_recipes")
        self.raw_recipes = raw
        self.df, self.ingredient_names = _hybrid.build_analysis_df(raw)
        self.n = len(self.df)
        self.recipe_ids = self.df['RP'].tolist()
        self.ing_cols = [c for c in self.df.columns if c.startswith('v_')]

    # ── targets ──────────────────────────────────────────────────────────
    def y(self, taste):
        return self.df[f'{taste}_actual'].values.astype(float)

    # ── feature blocks ───────────────────────────────────────────────────
    def X_voigt5(self, taste=None):
        """The 5-D weighted-average (Voigt) taste vector — the paper's Lasso 5D input."""
        return self.df[[f'{t}_voigt' for t in TASTES]].values.astype(float)

    def X_hybrid(self, taste):
        """HS midpoint + Voigt for this taste + the 8 chemistry proxies (10 features)."""
        return self.df[[f'{taste}_hs_mid', f'{taste}_voigt'] + CHEM_COLS].values.astype(float)

    def X_ingredients(self, taste=None):
        """Per-ingredient mass fractions (115 features)."""
        return self.df[self.ing_cols].values.astype(float)

    def X_chem_only(self, taste=None):
        return self.df[CHEM_COLS].values.astype(float)

    # ── unfitted physical predictors ─────────────────────────────────────
    def hs_mid(self, taste):
        return self.df[f'{taste}_hs_mid'].values.astype(float)

    def voigt(self, taste):
        return self.df[f'{taste}_voigt'].values.astype(float)


_CACHE = {}


def load(force=False):
    if force or 'ds' not in _CACHE:
        _CACHE['ds'] = Dataset()
    return _CACHE['ds']


if __name__ == "__main__":
    ds = load()
    print(f"recipes={ds.n}  ingredients={len(ds.ingredient_names)}")
    for t in TASTES:
        y = ds.y(t)
        print(f"  {t:7s} mean={y.mean():6.2f} sd={y.std(ddof=1):6.2f} "
              f"min={y.min():5.1f} max={y.max():6.1f} levels={len(np.unique(y)):3d} "
              f"frac<=3={np.mean(y <= 3):6.1%}")
