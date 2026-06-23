# Summary: 24 Train + 6 Test Baseline Comparison

## Required Statements
- q_scale used: `3008.198194823261`.
- Training set: first 24 rows from `Formulas-train` and `Release-train`.
- Test set: all 6 rows from `Formulas-test` and `Release-test`.
- Excluded 6 rows from original training sheet: `['S10', 'Opt-2', 'Opt-4', 'Opt-6', 'Opt-7', 'Opt-10']`.
- Train Run No values: `['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']`.
- Test Run No values: `['Opt-2-1', 'Opt-2-4', 'Opt-2-5', 'Opt-2-7', 'Opt-2-8', 'Opt-2-10']`.
- Train/test leakage detected: `False`.
- No unconstrained SR, DKC-SR retraining, repeated splits, or bootstrap refitting were run.

## Baseline Models Trained
- Mean train baseline
- Ridge regression
- PLS regression
- Polynomial RSM degree 2
- Random Forest Regressor
- Gaussian Process Regressor

## Endpoint Metrics Table
| model | RMSE_Q6 | MAE_Q6 | R2_Q6 | Spearman_Q6 | Kendall_Q6 | pairwise_accuracy_Q6 | top1_hit_Q6 | top2_hit_Q6 | RMSE_AUC | MAE_AUC | R2_AUC | Spearman_AUC | pairwise_accuracy_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLS regression | 803.029 | 782.177 | -29.4886 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2249.05 | 2127.63 | -11.3245 | -0.6 | 0.266667 |
| Ridge regression | 811.338 | 791.042 | -30.1228 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2274.36 | 2155.4 | -11.6034 | -0.6 | 0.266667 |
| Polynomial RSM degree 2 | 847.474 | 823.959 | -32.9569 | -0.714286 | -0.6 | 0.2 | 0 | 0 | 2385.66 | 2255.96 | -12.8672 | -0.828571 | 0.133333 |
| Random Forest Regressor | 936.833 | 918.829 | -40.4954 | -0.845154 | -0.774597 | 0 | 0 | 0 | 2568.87 | 2449.6 | -15.0788 | -0.845154 | 0 |
| Gaussian Process Regressor | 979.991 | 967.32 | -44.4066 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2795.76 | 2693.32 | -18.0446 | -0.428571 | 0.4 |
| Mean train baseline | 980.145 | 969.295 | -44.4209 | nan | nan | nan | 0 | 1 | 2780.85 | 2706.05 | -17.842 | nan | nan |

## Curve Proxy Metrics Table
- Primary curve metrics exclude `time_h = 0`.
| model | RMSE | MAE | R2 | MSE_normalized_by_q_scale | n_points |
| --- | --- | --- | --- | --- | --- |
| PLS regression | 507.957 | 398.958 | 0.752754 | 0.271323 | 42 |
| Ridge regression | 510.843 | 401.282 | 0.749937 | 0.274414 | 42 |
| Random Forest Regressor | 529.597 | 420.637 | 0.731239 | 0.294933 | 42 |
| Polynomial RSM degree 2 | 538.994 | 428.085 | 0.721617 | 0.305491 | 42 |
| Mean train baseline | 570.881 | 464.56 | 0.687704 | 0.342707 | 42 |
| Gaussian Process Regressor | 590.923 | 470.319 | 0.665391 | 0.367193 | 42 |

## Recommended Manuscript Table
- Use the endpoint baseline table as the main Reviewer #1 comparison because it evaluates formulation-level predictions on the same 24-train/6-test split.
- Recommended columns: model, Q6 RMSE, Q6 MAE, Q6 R2, Q6 Spearman, Q6 pairwise accuracy, Q6 top-1/top-2 hits, AUC RMSE, AUC Spearman, and AUC pairwise accuracy.
- Best endpoint model by held-out Q6 RMSE in this run: `PLS regression`.
- Use the curve proxy table as supplementary evidence only, since those models learn a direct `time_h -> Q` regression rather than an ODE/SR mechanism.

## Review Files
- `revision_validation_24train6test/reports/00_dataset_24train6test_report.md`
- `revision_validation_24train6test/reports/01_static_endpoint_baselines_24train6test_report.md`
- `revision_validation_24train6test/reports/02_curve_proxy_baselines_24train6test_report.md`
- `revision_validation_24train6test/results/static_endpoint_baseline_metrics_24train6test.csv`
- `revision_validation_24train6test/results/static_endpoint_baseline_predictions_24train6test.csv`
- `revision_validation_24train6test/results/curve_proxy_baseline_metrics_24train6test.csv`
- `revision_validation_24train6test/results/curve_proxy_baseline_predictions_24train6test.csv`
- `revision_validation_24train6test/figures/static_q6_parity_24train6test.png`
- `revision_validation_24train6test/figures/static_auc_parity_24train6test.png`
- `revision_validation_24train6test/figures/static_q6_ranking_barplot_24train6test.png`
- `revision_validation_24train6test/figures/curve_proxy_parity_24train6test.png`
