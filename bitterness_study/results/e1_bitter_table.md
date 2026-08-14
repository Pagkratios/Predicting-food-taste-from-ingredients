# E1 — every model tried on bitterness (leave-one-out, N=70)

`Skill_*` is relative to the leave-one-out constant-mean predictor; positive means better than predicting the mean.
`Wilcoxon_p_vs_const` tests the paired per-recipe absolute errors against that same constant.

| model                             | family       |   MAE |   RMSE |   R2_oos |     PCC |   Skill_MSE |   Skill_MAE |   Wilcoxon_p_vs_const |   Frac_recipes_improved |
|:----------------------------------|:-------------|------:|-------:|---------:|--------:|------------:|------------:|----------------------:|------------------------:|
| Constant (median)                 | baseline     | 0.714 |  1.493 |   -0.172 | nan     |      -0.138 |       0.226 |                 0.000 |                   0.671 |
| log1p-transform Ridge             | floor-aware  | 0.761 |  1.241 |    0.190 |   0.455 |       0.213 |       0.176 |                 0.000 |                   0.671 |
| sqrt-transform Ridge              | floor-aware  | 0.770 |  1.249 |    0.179 |   0.447 |       0.203 |       0.166 |                 0.001 |                   0.671 |
| Ordinal (prop. odds)              | floor-aware  | 0.802 |  1.221 |    0.216 |   0.467 |       0.238 |       0.131 |                 0.004 |                   0.657 |
| Poisson GLM                       | floor-aware  | 0.811 |  1.239 |    0.193 |   0.455 |       0.216 |       0.122 |                 0.010 |                   0.643 |
| Multinomial levels                | floor-aware  | 0.823 |  1.300 |    0.111 |   0.403 |       0.136 |       0.109 |                 0.005 |                   0.686 |
| NegBinomial GLM                   | floor-aware  | 0.825 |  1.259 |    0.167 |   0.459 |       0.191 |       0.107 |                 0.009 |                   0.657 |
| Ridge                             | linear       | 0.833 |  1.237 |    0.195 |   0.445 |       0.218 |       0.098 |                 0.042 |                   0.543 |
| Lasso shrunk to mean              | floor-aware  | 0.835 |  1.280 |    0.138 |   0.387 |       0.162 |       0.096 |                 0.009 |                   0.600 |
| Tobit (censored at 0)             | floor-aware  | 0.847 |  1.249 |    0.179 |   0.453 |       0.203 |       0.083 |                 0.055 |                   0.614 |
| Lasso per-ingredient              | as-published | 0.854 |  1.444 |   -0.096 |   0.098 |      -0.065 |       0.075 |                 0.004 |                   0.686 |
| ElasticNet                        | linear       | 0.854 |  1.254 |    0.174 |   0.440 |       0.197 |       0.075 |                 0.083 |                   0.571 |
| Lasso + clip at 0                 | floor-aware  | 0.861 |  1.261 |    0.163 |   0.437 |       0.187 |       0.068 |                 0.107 |                   0.557 |
| Lasso 5D                          | as-published | 0.861 |  1.262 |    0.163 |   0.437 |       0.187 |       0.068 |                 0.109 |                   0.557 |
| Random Forest                     | nonlinear    | 0.863 |  1.327 |    0.074 |   0.326 |       0.100 |       0.066 |                 0.225 |                   0.571 |
| RV as published (phi-calibrated)  | physical     | 0.886 |  1.444 |   -0.097 |   0.326 |      -0.065 |       0.041 |                 0.005 |                   0.557 |
| HS as published (phi-calibrated)  | physical     | 0.900 |  1.488 |   -0.164 |   0.269 |      -0.131 |       0.025 |                 0.006 |                   0.557 |
| HS midpoint (uncalibrated)        | physical     | 0.920 |  1.395 |   -0.023 |   0.308 |       0.006 |       0.004 |                 0.139 |                   0.586 |
| Constant (mean)                   | baseline     | 0.923 |  1.399 |   -0.029 |  -1.000 |       0.000 |       0.000 |                 1.000 |                   0.000 |
| k-NN                              | nonlinear    | 0.940 |  1.509 |   -0.198 |   0.217 |      -0.164 |      -0.018 |                 0.162 |                   0.586 |
| Gradient Boosting                 | nonlinear    | 0.950 |  1.418 |   -0.057 |   0.340 |      -0.027 |      -0.029 |                 0.667 |                   0.543 |
| Hybrid HS+chem                    | as-published | 0.956 |  1.626 |   -0.389 |  -0.098 |      -0.350 |      -0.035 |                 0.768 |                   0.600 |
| Voigt weighted avg (uncalibrated) | physical     | 1.650 |  2.420 |   -2.078 |   0.391 |      -1.991 |      -0.786 |                 0.002 |                   0.400 |
