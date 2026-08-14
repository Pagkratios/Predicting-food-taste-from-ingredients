# Recipe Formulation — Sensory Prediction

Predicts five sensory attributes (sweet, bitter, salty, umami, sour) from recipe ingredient compositions, comparing four methods:

| Method | Description |
|--------|-------------|
| **Hashin-Shtrikman (HS)** | Exact N-phase composite material bounds with LOO-calibrated φ |
| **Reuss-Voigt (RV)** | Voigt (weighted average) and Reuss (harmonic mean) bounds |
| **Lasso** | L1-regularised regression via proximal gradient descent |
| **Hybrid** | HS bounds + Voigt + chemistry-proxy features, fitted with LassoCV |

Also includes an **inverse design** module that optimises ingredient fractions to hit a target sensory profile.

---

## Requirements

Python 3.8+

```bash
pip install -r Lasso/requirements.txt
```

---

## Setup

```bash
cp .env.example .env
# Edit .env to set RAW_RECIPES_PATH (defaults to data/raw_recipes.py)
```

---

## Running the Pipeline

All commands are run from the repo root.

### 1. HS / RV bounds

```bash
python3 HS_RV/src/compute_bounds.py
```

Writes `data/hs_predictions.py` and `data/rv_predictions.py`.

### 2. Lasso pipeline

```bash
# Full pipeline (preprocess → EDA plots → train)
python3 Lasso/src/run_all.py

# Step by step
python3 Lasso/src/preprocess.py   # normalise weights, save .npy files
python3 Lasso/src/data_plots.py   # EDA: ingredient usage, KDE, PCA, t-SNE
python3 Lasso/src/train.py        # train, metrics, evaluation plots

# Selective plot regeneration
python3 Lasso/src/data_plots.py --only gaussians combined
python3 Lasso/src/data_plots.py --only pca tsne
python3 Lasso/src/run_all.py --steps plots --plot-groups gaussians combined
```

### 3. Hybrid model

```bash
python3 Hybrid/src/hybrid_analysis.py   # Tables 1 & 2 (coverage + model comparison)
python3 Hybrid/src/hybrid_plots.py      # Hybrid scatter, coverage, MAE, bias, boxplot
python3 Hybrid/src/publication_figures.py  # Figures 1–3 (composite publication figures)
```

### 4. Composite figure (all four methods)

```bash
python3 Lasso/src/composite_figure.py
```

Generates `results/plots/composite_figure.png`: 2×2 predicted-vs-actual scatter (HS, RV, Lasso, Hybrid) + grouped RMSE boxplot across all five sensory attributes.

### 5. Inverse design

```bash
python3 Inverse/src/inverse_design.py
```

Runs three case studies (pea soup salt reduction, chocolate spread sugar reduction, ketchup umami boost) via differential evolution. Writes `results/inverse/inverse_results.json`.

---

## Project Structure

```
├── .env.example
├── data/
│   ├── raw_recipes.py            # Dataset: recipes with ingredient compositions
│   ├── data_predictions.py       # Lasso predictions (written by train.py)
│   ├── hs_predictions.py         # HS predictions (written by compute_bounds.py)
│   └── rv_predictions.py         # RV predictions (written by compute_bounds.py)
│
├── HS_RV/src/
│   ├── compute_bounds.py         # Full N-phase HS/RV computation + LOO calibration
│   └── env_config.py
│
├── Lasso/
│   ├── requirements.txt
│   └── src/
│       ├── plot_config.py        # Shared plotting config (colors, fonts, DPI)
│       ├── preprocess.py
│       ├── lasso.py              # Custom Lasso via proximal gradient descent
│       ├── data_plots.py         # EDA plots
│       ├── train.py              # Nested-LOO evaluation, metrics, plots
│       ├── run_all.py            # Pipeline orchestrator
│       └── composite_figure.py   # 4-method composite figure
│
├── Hybrid/src/
│   ├── hybrid_analysis.py        # Hybrid model + Tables 1 & 2 (single source of truth)
│   ├── hybrid_plots.py           # Coverage, MAE, bias, boxplot
│   ├── overview_panel.py         # Overview panel figure
│   └── publication_figures.py    # Figures 1–3
│
├── Inverse/src/
│   └── inverse_design.py         # Inverse design via differential evolution
│
├── tests/
│   └── test_pipeline_invariants.py  # Regression tests (python3 -m pytest tests/)
│
└── results/                      # All generated outputs (gitignored)
    ├── plots/                    # Scatter plots, composite figure
    ├── plots_data/               # EDA plots
    ├── pca/                      # PCA clustering outputs
    ├── t-sne/                    # t-SNE clustering outputs
    ├── models/                   # Trained models, predictions
    ├── metrics/                  # Per-method metrics (CSV, Markdown, LaTeX)
    ├── hybrid/                   # Table 2, hybrid coefficients
    ├── coverage/                 # Table 1, bound coverage stats
    ├── inverse/                  # Inverse design results
    └── publication_figures/      # Figures 1–3
```

---

## Data Format

`data/raw_recipes.py` exposes a `raw_recipes` list of dicts:

```python
{
    "recipe_id": "RP1",
    "recipe_name": "Bread brown wheat with unsalted butter",
    "food_sensory_scores": {
        "sweet": 8.0, "bitter": 1.0, "salty": 18.0, "umami": 2.0, "sour": 2.0
    },
    "ingredients": [
        {
            "name": "Toast",
            "weight": 0.75,
            "sensory_scores": {"sweet": 3.0, "bitter": 1.0, "salty": 20.0, "umami": 3.0, "sour": 1.0}
        }
    ]
}
```

Weights are normalised to sum to 1. Sensory scores are on a 0–100 scale.

---

## Outputs

| Location | Contents |
|----------|----------|
| `results/plots/` | Predicted vs Actual (per method); composite 4-method figure; RMSE boxplot |
| `results/plots_data/` | Ingredient usage; per-sensory KDE + Gaussian; pooled distributions |
| `results/pca/` | PCA scatter, silhouette, PC1 loadings, cluster membership |
| `results/t-sne/` | t-SNE scatter, silhouette, p-values, cluster membership |
| `results/models/` | `final_models.pkl`; predictions JSON and CSV |
| `results/metrics/` | Per-method and combined metrics (CSV, Markdown, LaTeX) |
| `results/hybrid/` | Table 2 model comparison; Lasso coefficients (Table S3) |
| `results/coverage/` | Table 1 bound coverage statistics |
| `results/inverse/` | Inverse design results JSON |
| `results/publication_figures/` | Figures 1–3 (composite publication figures) |

All plots saved at 600 DPI.

---

## Sensory Attributes

| Attribute | Color |
|-----------|-------|
| Sweet | `#4477AA` |
| Bitter | `#EE6677` |
| Salty | `#228833` |
| Umami | `#CCBB44` |
| Sour | `#AA3377` |

Tol Bright palette (colorblind-safe). Styling centralised in `Lasso/src/plot_config.py`.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_RECIPES_PATH` | `data/raw_recipes.py` | Path to the recipe dataset |

Set via `.env` or OS environment variable.
