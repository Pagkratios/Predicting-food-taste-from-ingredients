# Recipe Formulation - Project Guide

## Overview

Predicts five sensory attributes (sweet, bitter, salty, umami, sour) from recipe ingredient compositions. Compares three methods: Hashin-Shtrikman (HS) bounds, Reuss-Voigt (RV) bounds, and Lasso regression (proximal gradient descent with L1 regularization).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Run Order

All scripts are run from the repo root:

```bash
# Optional: standalone HS/RV bounds export
python3 HS_RV/src/compute_bounds.py

# 1. Preprocess raw data into feature/target matrices
python3 Lasso/src/preprocess.py

# 2. Generate exploratory data analysis plots
python3 Lasso/src/data_plots.py

# 3. Train models and produce evaluation plots/metrics
python3 Lasso/src/train.py

# Selective plot regeneration without retraining
python3 Lasso/src/data_plots.py --only gaussians combined
python3 Lasso/src/data_plots.py --only pca tsne
python3 Lasso/src/run_all.py --steps plots --plot-groups gaussians combined
```

## Project Structure

```
HS_RV/
  src/
    compute_bounds.py     # Standalone HS/RV computation pipeline
    env_config.py         # RAW_RECIPES_PATH loader for HS_RV

data/
  raw_recipes.py         # Shared default dataset; preferred RAW_RECIPES_PATH target
  data_predictions.py    # Shared Lasso prediction store
  hs_predictions.py      # Generated HS bounds export
  rv_predictions.py      # Generated RV bounds export

Lasso/
  data/
    raw_recipes.py         # Compatibility shim to shared repo-level data/raw_recipes.py
    data_predictions.py    # Compatibility shim to shared repo-level data/data_predictions.py
    hs_predictions.py      # Compatibility shim to shared repo-level data/hs_predictions.py
    rv_predictions.py      # Compatibility shim to shared repo-level data/rv_predictions.py
  src/
    plot_config.py         # Shared plotting configuration (colors, fonts, DPI, helpers)
    preprocess.py          # Normalize weights, standardize features, save .npy files
    data_plots.py          # EDA: pie chart, KDE/Gaussian, t-SNE, PCA clustering
    train.py               # LOO alpha tuning, Lasso training, metrics, plots
    lasso.py               # Custom Lasso via proximal gradient descent
  data/processed/          # Preprocessed .npy arrays (gitignored)

results/                   # All generated outputs (gitignored)
  plots_data/              # Ingredient usage, KDE/Gaussian distribution plots
  t-sne/                   # t-SNE clustering: scatter, silhouette, p-values, members
  pca/                     # PCA clustering: scatter, silhouette, loadings, members
  plots/                   # Predicted vs Actual scatter, RMSE boxplot
  models/                  # Trained models (.pkl), predictions (.json, .csv)
  metrics/                 # Per-method and combined metric tables (.csv, .md, .tex)

webapp/
  app.py                   # Flask web app
  templates/index.html
  static/

requirements.txt           # Universal requirements for the entire repo
```

## Key Constants

- `SENSORY_ORDER = ['sweet', 'bitter', 'salty', 'umami', 'sour']`
- `ALIASES` maps alternate names (e.g., `sweetness` -> `sweet`)
- All output paths are constructed from `PROJECT_ROOT` (= `Lasso/`)
- All shared constants live in `plot_config.py` (single source of truth)

## Data Format

The active raw recipe file is configured by `RAW_RECIPES_PATH` in `.env`. It should point to a Python file that exposes `raw_recipes` as a list of dicts. The preferred shared location is `data/raw_recipes.py`:

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

All plots use the shared styling module `Lasso/src/plot_config.py`.

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

## Notes

- All generated outputs under `results/` and `Lasso/data/processed/` are gitignored. Re-run the scripts to regenerate.
- `HS_RV/src/compute_bounds.py` is intentionally separate from the Lasso pipeline and writes shared HS/RV exports under `data/`.
- `data_plots.py` supports `--only` so you can regenerate just selected plot groups without touching trained artifacts.
- `run_all.py` supports `--steps`, `--plot-groups`, and optional `--clean` for selective reruns.
- `train.py` writes Lasso predictions back into `data/data_predictions.py`.
- HS and RV predictions are loaded from separate shared files (`data/hs_predictions.py`, `data/rv_predictions.py`).
- The custom Lasso in `lasso.py` uses soft-thresholding (proximal operator) after each gradient step.
