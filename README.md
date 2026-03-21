# Recipe Sensory Attribute Prediction via Lasso Regression

## Abstract

This project investigates the prediction of five sensory attributes -- sweetness, bitterness, saltiness, umami, and sourness -- from recipe ingredient compositions. We compare three prediction methods: Hyperplane Sampling (HS), Random Voting (RV), and L1-regularized (Lasso) regression trained via proximal gradient descent. The pipeline includes data preprocessing, exploratory analysis (KDE distributions, t-SNE and PCA clustering), hyperparameter tuning via leave-one-out cross-validation, and comprehensive evaluation using 12 metrics.

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
| **HS** (Hyperplane Sampling) | Baseline geometric sampling method |
| **RV** (Random Voting) | Ensemble-based random voting approach |
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
raw_recipes.py (embedded dataset)
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
    - Combined distribution (excluding sour)
    - PCA clustering (silhouette, scatter, PC1 loadings)
    - t-SNE clustering (silhouette, scatter, p-values)
        |
        v
  train.py (model training & evaluation)
    - LOO cross-validation for alpha tuning
    - Train final Lasso models on raw X
    - Extract HS/RV predictions from data_predictions.py
    - Generate predicted-vs-actual scatter plots (3x)
    - Generate grouped RMSE boxplot
    - Compute and save 12 evaluation metrics per method
    - Export per-recipe predictions as JSON
```

## Installation

```bash
pip install -r Lasso_project/requirements.txt
```

**Requirements:** Python 3.8+, NumPy, scikit-learn, Matplotlib, SciPy, pandas, seaborn, tabulate

## Reproducing Results

```bash
# Step 1: Preprocess data
python3 Lasso_project/src/preprocess.py

# Step 2: Generate exploratory plots
python3 Lasso_project/src/data_plots.py

# Step 3: Train models and evaluate
python3 Lasso_project/src/train.py
```

## Outputs

All outputs are generated under `Lasso_project/results/`:

### Exploratory Analysis (`results/plots_data/`)
| File | Description |
|------|-------------|
| `ingredient_usage.png` | Pie chart of ingredients appearing in >= 3 recipes |
| `gaussian_<key>.png` | KDE + Gaussian fit per sensory attribute with top-3 outlier annotations |
| `combined_distribution.png` | Pooled KDE across sweet, bitter, salty, umami |

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
  Lasso_project/
    requirements.txt
    data/
      raw_recipes.py          # Embedded recipe dataset
      data_predictions.py     # HS/RV/Lasso predictions (updated by train.py)
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
