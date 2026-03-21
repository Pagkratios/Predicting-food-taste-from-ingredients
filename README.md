# Recipe Sensory Attribute Prediction via Lasso Regression

## Abstract

This project investigates the prediction of five sensory attributes -- sweetness, bitterness, saltiness, umami, and sourness -- from recipe ingredient compositions. We compare three prediction methods: Hashin-Shtrikman (HS) bounds, Reuss-Voigt (RV) bounds, and L1-regularized (Lasso) regression trained via proximal gradient descent. The pipeline includes data preprocessing, exploratory analysis (KDE distributions, t-SNE and PCA clustering), hyperparameter tuning via leave-one-out cross-validation, and comprehensive evaluation using 12 metrics.

## Problem Statement

Predicting how a recipe will taste from its ingredient list is a challenging problem in food science and computational gastronomy. Given a set of ingredients with known sensory profiles and their proportional weights in a recipe, can we accurately predict the overall sensory experience?

Each recipe is represented as a weighted combination of ingredient sensory vectors across five taste dimensions. The goal is to learn a mapping from these input vectors to the recipe's ground-truth sensory scores.

## Methodology

### Data Representation

- **Input (X):** Weighted average of ingredient sensory score vectors, where weights are normalized to sum to 1.0
- **Target (Y):** Recipe-level ground-truth sensory scores (0-100 scale) for each of the five taste dimensions

### Prediction Methods

| Method | Description |
|--------|-------------|
| **HS** (Hashin-Shtrikman bounds) | Composite material bounds applied to sensory prediction |
| **RV** (Reuss-Voigt bounds) | Parallel/series mixture bounds for sensory prediction |
| **Lasso** | L1-regularized linear regression via proximal gradient descent with soft-thresholding |

### Lasso Implementation

The custom Lasso regressor (`lasso.py`) minimizes:

```
L(w) = (1/2n) ||Xw - y||^2 + alpha * ||w||_1
```

using iterative proximal gradient descent with the soft-thresholding operator:

```
S(x, t) = sign(x) * max(|x| - t, 0)
```

Hyperparameter `alpha` is selected per sensory target via leave-one-out cross-validation over 30 log-spaced candidates.

## Dataset

- **29 recipes** with full ingredient compositions
- **5 sensory dimensions:** sweet, bitter, salty, umami, sour
- Each ingredient has a weight (proportion) and its own sensory score vector
- Sensory scores range from 0 to 100

## Pipeline

```
RAW_RECIPES_PATH from .env
        |
        v
  preprocess.py
    - Normalize ingredient weights (L1)
    - Compute weighted-average feature vectors (X)
    - Extract ground-truth sensory targets (Y)
    - Standardize X; save X_train.npy, Y_train.npy
        |
        v
  data_plots.py (exploratory analysis)
    - Ingredient usage pie chart
    - Per-sensory KDE + Gaussian fit (5 plots)
    - Combined distribution (excluding bitter)
    - PCA clustering (silhouette, scatter, PC1 loadings)
    - t-SNE clustering (silhouette, scatter, p-values)
        |
        v
  train.py (model training & evaluation)
    - LOO cross-validation for alpha tuning
    - Train final Lasso models on raw X
    - Extract HS/RV predictions from shared `data/hs_predictions.py` and `data/rv_predictions.py`
    - Generate predicted-vs-actual scatter plots (3x)
    - Generate grouped RMSE boxplot
    - Compute and save 12 evaluation metrics per method
    - Export per-recipe predictions as JSON
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<25"
python -m pip install -r Lasso_project/requirements.txt
```

**Requirements:** Python 3.10 recommended. Dependencies are pinned to a known-good set, including `numpy==1.26.4`, to avoid the NumPy 2.x ABI mismatch that can break `pandas`, `scikit-learn`, `numexpr`, `bottleneck`, and `seaborn`.

## Reproducing Results

```bash
# Standalone HS/RV bounds generation
python Bounds_project/src/compute_bounds.py

# One-command full pipeline
python Lasso_project/src/run_all.py --clean

# Or run each step manually

# Step 1: Preprocess data
python Lasso_project/src/preprocess.py

# Step 2: Generate exploratory plots
python Lasso_project/src/data_plots.py

# Step 3: Train models and evaluate
python Lasso_project/src/train.py

# Only regenerate Gaussian plots + combined distributions
python Lasso_project/src/data_plots.py --only gaussians combined

# Only regenerate PCA and t-SNE outputs
python Lasso_project/src/data_plots.py --only pca tsne

# Through the master runner, without retraining or cleaning everything
python Lasso_project/src/run_all.py --steps plots --plot-groups gaussians combined
python Lasso_project/src/run_all.py --steps plots --plot-groups pca tsne
python Lasso_project/src/run_all.py --steps train
```

`Lasso_project/src/run_all.py` now supports selective execution. It only removes generated outputs when you pass `--clean`.

## Standalone Bounds Pipeline

`Bounds_project/src/compute_bounds.py` is a separate process from the Lasso pipeline. It resolves the dataset through the same `RAW_RECIPES_PATH` value in `.env` and computes:

- Reuss lower bounds
- Voigt weighted-mixture predictions
- Hashin-Shtrikman lower and upper bounds
- Per-dimension absolute and squared errors against `food_sensory_scores`

Outputs are written to:

- `data/hs_predictions.py`
- `data/rv_predictions.py`

Both HS and RV predictions are calibrated using a least-squares φ parameter (see [φ Calibration](#φ-calibration) below). The generated Python files use a compact legacy-style layout keyed by `recipe_id - recipe_name` for easier inspection.

Dataset path configuration:

```bash
cp .env.example .env
```

Set `RAW_RECIPES_PATH` in `.env` to either:
- a repo-relative path such as `data/raw_recipes.py`
- or an absolute path to another `raw_recipes.py` file

The repository includes a GitHub Actions workflow that installs from `Lasso_project/requirements.txt` in a fresh environment and runs all three scripts on each push and pull request.

## φ Calibration

For both HS and RV bounds, a single scalar φ ∈ [0, 1] is learned **per taste dimension** to position the prediction between the lower and upper bound:

```
T* = T⁻ + φ · (T⁺ - T⁻)
```

### Closed-form least-squares solution

φ is chosen to minimise the squared prediction error over all recipes. The closed-form solution is:

```
φ* = Σ(Δr · (ar − T⁻r)) / Σ(Δr²),  clipped to [0, 1]
```

where:
- `Δr = T⁺r − T⁻r` — bound gap for recipe r
- `ar` — actual taste score for recipe r
- `T⁻r` — lower bound for recipe r

When `Σ(Δr²) < EPS` (all bounds are degenerate), φ defaults to 0.5.

### Cross-validation

φ is estimated with **Leave-One-Out (LOO) cross-validation** over the full dataset (70 recipes):

1. For each recipe r, fit φ on the remaining 69 recipes using the closed-form above.
2. Predict recipe r as `T⁻r + φ_loo · Δr`.
3. Repeat for all 70 recipes, producing exactly 70 LOO predictions — matching the Lasso output size exactly.

The **final exported φ** values are fit on all 70 recipes (full-data estimate) and reported alongside PCC and R² for the LOO predictions.

This calibration is applied identically to both the HS bounds (`hs_lower`, `hs_upper`) and the RV bounds (`rv_lower`, `rv_upper`).

## Outputs

All outputs are generated under `Lasso_project/results/`:

### Exploratory Analysis (`results/plots_data/`)
| File | Description |
|------|-------------|
| `ingredient_usage.png` | Pie chart of ingredients appearing in >= 3 recipes |
| `gaussian_<key>.png` | KDE + Gaussian fit per sensory attribute with top-3 outlier annotations |
| `combined_distribution.png` | Pooled KDE across sweet, sour, salty, umami (without bitter) |
| `combined_distribution_with_bitter.png` | Pooled KDE across all five sensory attributes |

### Clustering (`results/pca/` and `results/t-sne/`)
| File | Description |
|------|-------------|
| `pca_silhouette.png` | Silhouette scores vs. k for PCA-space KMeans |
| `pca_bestk_scatter.png` | PCA scatter colored by best-k clusters |
| `pc1_loadings.png` | PC1 feature importance bar plot |
| `clusters_bestk.json` | Cluster membership (recipes + ingredients) |
| `tsne_bestk.png` | t-SNE scatter colored by best-k clusters |
| `silhouette.png` | Silhouette scores for t-SNE-space KMeans |
| `p_values.json` | Mann-Whitney, Welch's t, KS-test p-values per cluster |

### Model Evaluation (`results/plots/` and `results/metrics/`)
| File | Description |
|------|-------------|
| `hs_predicted_vs_actual.png` | HS predictions vs. ground truth with PCC/R^2 |
| `rv_predicted_vs_actual.png` | RV predictions vs. ground truth |
| `lasso_predicted_vs_actual.png` | Lasso predictions vs. ground truth |
| `rmse_boxplot_all_methods.png` | Grouped RMSE comparison across methods and sensory targets |
| `{method}_metrics.csv` | Per-method evaluation metrics |
| `all_metrics_wide.csv` | Combined metrics in wide format |
| `all_metrics.md` / `all_metrics.tex` | Publication-ready metric tables |

### Models (`results/models/`)
| File | Description |
|------|-------------|
| `final_models.pkl` | Serialized trained Lasso models |
| `per_recipe_lasso_predictions.json` | Per-recipe predictions with actual values and diffs |
| `real_vs_predicted.csv` | Long-format prediction results |

## Evaluation Metrics

Each method is evaluated per sensory target and overall using:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| Median AE | Median Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R^2 | Coefficient of Determination |
| Explained Variance | Explained Variance Score |
| Pearson r | Pearson Correlation Coefficient |
| Spearman rho | Spearman Rank Correlation |
| Bias | Mean prediction error |
| Std Error | Standard deviation of errors |
| MAPE | Mean Absolute Percentage Error |
| sMAPE | Symmetric Mean Absolute Percentage Error |

## Project Structure

```
Recipe_formulation/
  README.md
  CLAUDE.md
  .gitignore
  data/
    raw_recipes.py          # Shared default recipe dataset
    data_predictions.py     # Shared prediction store used by training
    hs_predictions.py       # Bounds_project-generated HS export
    rv_predictions.py       # Bounds_project-generated RV export
  Bounds_project/
    src/
      compute_bounds.py      # Standalone HS/RV bounds generator
      env_config.py          # RAW_RECIPES_PATH resolution for Bounds_project
  Lasso_project/
    requirements.txt
    data/
      raw_recipes.py          # Compatibility shim to shared repo-level data/raw_recipes.py
      data_predictions.py     # Compatibility shim to shared repo-level data/data_predictions.py
    src/
      preprocess.py           # Data preprocessing pipeline
      data_plots.py           # Exploratory data analysis plots
      train.py                # Model training and evaluation
      lasso.py                # Custom Lasso regression implementation
    results/                  # Generated outputs (gitignored)
      plots_data/             # KDE, Gaussian, ingredient usage plots
      t-sne/                  # t-SNE clustering analysis
      pca/                    # PCA clustering analysis
      plots/                  # Model evaluation plots
      models/                 # Trained models and predictions
      metrics/                # Performance metric tables
    data/processed/           # Preprocessed arrays (gitignored)
```

## License

TBD
