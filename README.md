# Recipe_formulation


This project provides tools for analyzing recipe data, generating plots, and training models (HS, RV, and Lasso) to predict sensory attributes such as sweetness, bitterness, saltiness, umami, and sourness.

Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```
(Requires Python 3.8 or higher.)

Preprocessing

Preprocess the raw recipe data to create feature and target matrices:

```bash
python3 Lasso_project/src/preprocess.py
```
This generates the following files in data/processed/:

X_train.npy – standardized features

Y_train.npy – sensory targets

x_scaler_mean.npy and x_scaler_scale.npy – scaler parameters

Data Analysis Plots

Run the script to generate descriptive statistics and publication-ready plots:

```bash
python3 Lasso_project/src/data_plots.py
```
This produces the following figures in results/plots_data/:

ingredient_usage.png – Pie chart of all ingredients used in ≥3 recipes.

tsne_recipes_ingredients_all_labels.png – t-SNE embedding of recipes (squares) and ingredients (circles), with all labels.

tsne_recipes_ingredients_no_labels.png – Same t-SNE embedding, without labels for a clean view.

gaussian_sweet.png, gaussian_bitter.png, gaussian_salty.png, gaussian_umami.png, gaussian_sour.png – For each sensory key, kernel density estimation (KDE) curve and Gaussian fit, with top-3 outlier recipes annotated.

combined_distribution.png – Combined KDE across sweet, bitter, salty, and umami (sour excluded), overlaid with a pooled Gaussian fit.

pca_silhouette.png – Silhouette scores for PCA clustering, used to select best k.

pca_bestk_scatter.png – PCA scatter plot with recipes and ingredients colored by clusters.

pc1_loadings.png – Bar plot showing contribution of each sensory key to the first principal component.

tsne_bestk.png – t-SNE scatter with clusters (best k chosen by silhouette).

Model Training and Evaluation

Train HS, RV, and Lasso models and generate performance plots:

```bash
python3 Lasso_project/src/train.py
```
This produces results in results/plots/, results/models/, and results/metrics/:

hs_predicted_vs_actual.png – Scatter plot of HS predictions vs actual sensory scores.

rv_predicted_vs_actual.png – Scatter plot of RV predictions vs actual sensory scores.

lasso_predicted_vs_actual.png – Scatter plot of Lasso predictions vs actual sensory scores.

rmse_boxplot_all_methods.png – Grouped boxplot of RMSE across all sensory keys and methods.

HS_metrics.csv, RV_metrics.csv, Lasso_metrics.csv – Per-model metrics.

all_metrics_long.csv, all_metrics_wide.csv, all_metrics.md, all_metrics.tex – Combined metrics tables in multiple formats.

per_recipe_lasso_predictions.json – Predictions per recipe.

real_vs_predicted.csv – Long-format prediction results for analysis.

Outputs

Plots – ingredient usage, t-SNE, Gaussian/KDE distributions, PCA, model evaluation

Metrics – per-model and combined performance tables

Models – trained HS, RV, and Lasso models, plus per-recipe predictions
