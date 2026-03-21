# Recipe Formulation - Project Guide

## Overview

Predicts five sensory attributes (sweet, bitter, salty, umami, sour) from recipe ingredient compositions. Compares three methods: Hashin-Shtrikman (HS) bounds, Reuss-Voigt (RV) bounds, and Lasso regression (proximal gradient descent with L1 regularization).

## Setup

```bash
pip install -r Lasso_project/requirements.txt
```

Requires Python 3.8+.

## Run Order

All scripts are run from the repo root:

```bash
# 1. Preprocess raw data into feature/target matrices
python3 Lasso_project/src/preprocess.py

# 2. Generate exploratory data analysis plots
python3 Lasso_project/src/data_plots.py

# 3. Train models and produce evaluation plots/metrics
python3 Lasso_project/src/train.py
```

## Project Structure

```
Lasso_project/
  data/
    raw_recipes.py         # Embedded dataset (29 recipes with ingredients + sensory scores)
    data_predictions.py    # Lasso predictions per recipe (updated by train.py)
    hs_predictions.py      # Pre-computed Hashin-Shtrikman (HS) bounds predictions
    rv_predictions.py      # Pre-computed Reuss-Voigt (RV) bounds predictions
  src/
    plot_config.py         # Shared plotting configuration (colors, fonts, DPI, helpers)
    preprocess.py          # Normalize weights, standardize features, save .npy files
    data_plots.py          # EDA: pie chart, KDE/Gaussian, t-SNE, PCA clustering
    train.py               # LOO alpha tuning, Lasso training, metrics, plots
    lasso.py               # Custom Lasso via proximal gradient descent
  results/                 # All generated outputs (gitignored)
    plots_data/            # Ingredient usage, KDE/Gaussian distribution plots
    t-sne/                 # t-SNE clustering: scatter, silhouette, p-values, members
    pca/                   # PCA clustering: scatter, silhouette, loadings, members
    plots/                 # Predicted vs Actual scatter, RMSE boxplot
    models/                # Trained models (.pkl), predictions (.json, .csv)
    metrics/               # Per-method and combined metric tables (.csv, .md, .tex)
    old_plots/             # Legacy plots for comparison (do not use for publications)
  data/processed/          # Preprocessed .npy arrays (gitignored)
  requirements.txt
```

## Key Constants

- `SENSORY_ORDER = ['sweet', 'bitter', 'salty', 'umami', 'sour']`
- `ALIASES` maps alternate names (e.g., `sweetness` -> `sweet`)
- All output paths are constructed from `PROJECT_ROOT` (= `Lasso_project/`)
- All shared constants live in `plot_config.py` (single source of truth)

## Data Format

Recipes are embedded in `raw_recipes.py` as a Python list of dicts:

```python
{
    "recipe_name": "...",
    "food_sensory_scores": {"sweet": 45.0, "bitter": 12.0, ...},
    "ingredients": [
        {"name": "...", "weight": 0.3, "sensory_scores": {"sweet": 60.0, ...}},
        ...
    ]
}
```

## Plotting Conventions

All plots use the shared styling module `Lasso_project/src/plot_config.py`.

### Usage

```python
from plot_config import apply_style, setup_figure, save_figure, style_axes
from plot_config import get_sensory_color, get_method_color, SENSORY_COLORS

apply_style()  # Call once at module load (use_seaborn=True if using seaborn)
fig, ax = setup_figure(size="double")  # "single", "onehalf", "double", "double_wide"
style_axes(ax)
save_figure(fig, "path/to/output.png")
```

### Standards

- **DPI**: 600 (save), 150 (display)
- **Font**: Serif, 9pt base / 10pt labels / 11pt titles / 8pt legend / 7pt annotations
- **Colors**: Tol Bright palette (colorblind-safe). Use `get_sensory_color()` and `get_method_color()`.
  - Sweet=#4477AA, Bitter=#EE6677, Salty=#228833, Umami=#CCBB44, Sour=#AA3377
- **Figure sizes**: Single column (3.5in), 1.5 column (5.5in), double column (7.0in)
- **Spines**: Top and right removed
- **Grid**: Dashed, alpha=0.3, grey (#CCC)
- **Method names**: Use `METHOD_DISPLAY` dict for full names in titles (e.g., "Hashin-Shtrikman (HS)")

### Output Directories

- `results/{plots_data,t-sne,pca,plots}/` -- new styled plots (publication-ready)
- `results/old_plots/` -- legacy plots preserved for comparison

## Notes

- All generated outputs under `Lasso_project/results/` and `Lasso_project/data/processed/` are gitignored. Re-run the scripts to regenerate.
- `train.py` writes Lasso predictions back into `data_predictions.py`.
- HS and RV predictions are loaded from separate files (`hs_predictions.py`, `rv_predictions.py`), not computed in this code.
- The custom Lasso in `lasso.py` uses soft-thresholding (proximal operator) after each gradient step.
