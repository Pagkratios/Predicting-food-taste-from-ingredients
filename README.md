# Recipe Sensory Attribute Prediction via Lasso Regression

## Abstract

This project investigates the prediction of five sensory attributes — sweetness, bitterness, saltiness, umami, and sourness — from recipe ingredient compositions. We compare three prediction methods: Hashin-Shtrikman (HS) bounds, Reuss-Voigt (RV) bounds, and L1-regularized (Lasso) regression trained via proximal gradient descent. The pipeline includes data preprocessing, exploratory analysis (KDE distributions, t-SNE and PCA clustering), hyperparameter tuning via leave-one-out cross-validation, and comprehensive evaluation using 12 metrics.

## Problem Statement

Predicting how a recipe will taste from its ingredient list is a challenging problem in food science and computational gastronomy. Given a set of ingredients with known sensory profiles and their proportional weights in a recipe, can we accurately predict the overall sensory experience?

Each recipe is represented as a weighted combination of ingredient sensory vectors across five taste dimensions. The goal is to learn a mapping from these input vectors to the recipe's ground-truth sensory scores.

## Methodology

### Data Representation

- **Input (X):** Weighted average of ingredient sensory score vectors, where weights are normalized to sum to 1.0
- **Target (Y):** Recipe-level ground-truth sensory scores (0–100 scale) for each of the five taste dimensions

### Prediction Methods

| Method | Description |
|--------|-------------|
| **HS** (Hashin-Shtrikman bounds) | Composite material bounds applied to sensory prediction |
| **RV** (Reuss-Voigt bounds) | Parallel/series mixture bounds for sensory prediction |
| **Lasso** | L1-regularized linear regression via proximal gradient descent with soft-thresholding |

### Lasso Implementation

The custom Lasso regressor (`Lasso/src/lasso.py`) minimizes:

```
L(w) = (1/2n) ||Xw - y||^2 + alpha * ||w||_1
```

using iterative proximal gradient descent with the soft-thresholding operator:

```
S(x, t) = sign(x) * max(|x| - t, 0)
```

Hyperparameter `alpha` is selected per sensory target via leave-one-out cross-validation over 30 log-spaced candidates.

## Dataset

- **Sensory database:** `data/Supplementary_Data_File_1_v11.xlsx` (sheet: `foods`) — ingredient-level sensory scores on the Spectrum™ scale (0–100) for sweet, sour, bitter, umami, salty
- **Recipe dataset:** `data/raw_recipes.py` — weighted ingredient compositions with ground-truth recipe-level sensory scores
- **5 sensory dimensions:** sweet, bitter, salty, umami, sour
- Sensory scores range from 0 to 100

## Pipeline

```
data/Supplementary_Data_File_1_v11.xlsx  +  data/raw_recipes.py
        |
        v
  Lasso/src/preprocess.py
    - Normalize ingredient weights (L1)
    - Compute weighted-average feature vectors (X)
    - Extract ground-truth sensory targets (Y)
    - Standardize X; save X_train.npy, Y_train.npy → Lasso/data/processed/
        |
        v
  Lasso/src/data_plots.py  (exploratory analysis)
    - Ingredient usage pie chart
    - Per-sensory KDE + Gaussian fit (5 plots)
    - Combined distribution (excluding bitter)
    - PCA clustering (silhouette, scatter, PC1 loadings)
    - t-SNE clustering (silhouette, scatter, p-values)
        |
        v
  Lasso/src/train.py  (model training & evaluation)
    - LOO cross-validation for alpha tuning
    - Train final Lasso models on full dataset
    - Load HS/RV predictions from data/hs_predictions.py and data/rv_predictions.py
    - Generate predicted-vs-actual scatter plots (3×)
    - Generate grouped RMSE boxplot
    - Compute and save 12 evaluation metrics per method
    - Export per-recipe predictions as JSON
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Reproducing Results

All scripts are run from the repo root:

```bash
# Optional: standalone HS/RV bounds generation
python3 HS_RV/src/compute_bounds.py

# Step 1: Preprocess raw data into feature/target matrices
python3 Lasso/src/preprocess.py

# Step 2: Generate exploratory data analysis plots
python3 Lasso/src/data_plots.py

# Step 3: Train models and produce evaluation plots/metrics
python3 Lasso/src/train.py

# Selective plot regeneration without retraining
python3 Lasso/src/data_plots.py --only gaussians combined
python3 Lasso/src/data_plots.py --only pca tsne

# Through the master runner
python3 Lasso/src/run_all.py --steps plots --plot-groups gaussians combined
python3 Lasso/src/run_all.py --steps plots --plot-groups pca tsne
python3 Lasso/src/run_all.py --steps train
python3 Lasso/src/run_all.py --clean   # full pipeline, remove old outputs first
```

## Standalone Bounds Pipeline

`HS_RV/src/compute_bounds.py` is independent of the Lasso pipeline. It resolves the dataset through `RAW_RECIPES_PATH` in `.env` and computes:

- Reuss lower bounds
- Voigt weighted-mixture predictions
- Hashin-Shtrikman lower and upper bounds
- Per-dimension absolute and squared errors against `food_sensory_scores`

Outputs are written to:

- `data/hs_predictions.py`
- `data/rv_predictions.py`

Both HS and RV predictions are calibrated using a least-squares φ parameter. The generated Python files are keyed by `recipe_id - recipe_name`.

Dataset path configuration:

```bash
cp .env.example .env
# Set RAW_RECIPES_PATH to data/raw_recipes.py or an absolute path
```

## φ Calibration

For both HS and RV bounds, a scalar φ ∈ [0, 1] is learned **per taste dimension** to interpolate between the lower and upper bound:

```
T* = T⁻ + φ · (T⁺ - T⁻)
```

φ is chosen to minimise the squared prediction error. The closed-form solution is:

```
φ* = Σ(Δr · (ar − T⁻r)) / Σ(Δr²),  clipped to [0, 1]
```

where `Δr = T⁺r − T⁻r` (bound gap), `ar` is the actual score, `T⁻r` is the lower bound for recipe r.

φ is estimated with **Leave-One-Out (LOO) cross-validation**: for each recipe r, φ is fit on all remaining recipes, then used to predict r. The final exported φ is fit on the full dataset.

## Outputs

All outputs are generated under `results/` (gitignored):

### Exploratory Analysis (`results/plots_data/`)
| File | Description |
|------|-------------|
| `ingredient_usage.png` | Pie chart of ingredients appearing in ≥ 3 recipes |
| `gaussian_<key>.png` | KDE + Gaussian fit per sensory attribute with outlier annotations |
| `combined_distribution.png` | Pooled KDE across sweet, sour, salty, umami |
| `combined_distribution_with_bitter.png` | Pooled KDE across all five attributes |

### Clustering (`results/pca/` and `results/t-sne/`)
| File | Description |
|------|-------------|
| `pca_silhouette.png` | Silhouette scores vs. k for PCA-space KMeans |
| `pca_bestk_scatter.png` | PCA scatter coloured by best-k clusters |
| `pc1_loadings.png` | PC1 feature importance bar plot |
| `clusters_bestk.json` | Cluster membership (recipes + ingredients) |
| `tsne_bestk.png` | t-SNE scatter coloured by best-k clusters |
| `silhouette.png` | Silhouette scores for t-SNE-space KMeans |
| `p_values.json` | Mann-Whitney, Welch's t, KS-test p-values per cluster pair |

### Model Evaluation (`results/plots/` and `results/metrics/`)
| File | Description |
|------|-------------|
| `hs_predicted_vs_actual.png` | HS predictions vs. ground truth |
| `rv_predicted_vs_actual.png` | RV predictions vs. ground truth |
| `lasso_predicted_vs_actual.png` | Lasso predictions vs. ground truth |
| `rmse_boxplot_all_methods.png` | Grouped RMSE comparison across methods |
| `{method}_metrics.csv` | Per-method evaluation metrics |
| `all_metrics_wide.csv` | Combined metrics in wide format |
| `all_metrics.md` / `all_metrics.tex` | Publication-ready metric tables |

### Models (`results/models/`)
| File | Description |
|------|-------------|
| `final_models.pkl` | Serialised trained Lasso models (one per sensory target) |
| `per_recipe_lasso_predictions.json` | Per-recipe predictions with actuals and diffs |
| `real_vs_predicted.csv` | Long-format prediction results |

## Evaluation Metrics

Each method is evaluated per sensory target using:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| Median AE | Median Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Explained Variance | Explained Variance Score |
| Pearson r | Pearson Correlation Coefficient |
| Spearman ρ | Spearman Rank Correlation |
| Bias | Mean prediction error |
| Std Error | Standard deviation of errors |
| MAPE | Mean Absolute Percentage Error |
| sMAPE | Symmetric Mean Absolute Percentage Error |

## Project Structure

```
Recipe_formulation/
  README.md
  CLAUDE.md
  requirements.txt                       # Universal dependencies for the full repo
  .env                                   # RAW_RECIPES_PATH (gitignored)

  data/
    Supplementary_Data_File_1_v11.xlsx   # Ingredient-level sensory database (Excel)
    raw_recipes.py                       # Shared recipe dataset with ground-truth scores
    data_predictions.py                  # Lasso prediction store (written by train.py)
    hs_predictions.py                    # HS bounds export (written by compute_bounds.py)
    rv_predictions.py                    # RV bounds export (written by compute_bounds.py)
    predicted_sensory_data.py            # Additional predicted ingredient scores
    sensory_database.json                # JSON mirror of sensory scores

  HS_RV/
    src/
      compute_bounds.py                  # Standalone HS/RV bounds computation pipeline
      env_config.py                      # RAW_RECIPES_PATH loader for HS_RV

  Lasso/
    data/
      raw_recipes.py                     # Shim → repo-level data/raw_recipes.py
      data_predictions.py                # Shim → repo-level data/data_predictions.py
      hs_predictions.py                  # Shim → repo-level data/hs_predictions.py
      rv_predictions.py                  # Shim → repo-level data/rv_predictions.py
      old_29_recipies_corrected.py       # Legacy 29-recipe dataset (reference only)
      old_receipies_wrong_sensory.py     # Deprecated dataset with incorrect scores
      processed/                         # Preprocessed .npy arrays (gitignored)
    src/
      plot_config.py                     # Shared plotting config (colours, fonts, DPI)
      preprocess.py                      # Normalise weights, standardise features
      data_plots.py                      # EDA: pie, KDE/Gaussian, t-SNE, PCA
      train.py                           # LOO alpha tuning, Lasso training, metrics
      lasso.py                           # Custom Lasso via proximal gradient descent
      run_all.py                         # Master runner (--steps, --plot-groups, --clean)
      env_config.py                      # RAW_RECIPES_PATH loader for Lasso

  results/                               # All generated outputs (gitignored)
    plots_data/
    t-sne/
    pca/
    plots/
    models/
    metrics/
```

## License

TBD
