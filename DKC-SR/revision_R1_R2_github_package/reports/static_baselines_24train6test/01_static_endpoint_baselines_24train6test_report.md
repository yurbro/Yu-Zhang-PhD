# Static Endpoint Baselines, 24 Train + 6 Test

## Required Context
- q_scale used: `3008.198194823261`.
- Baseline models were trained on raw physical endpoint targets: `Q6_obs` and `AUC_obs`.
- The selected artifact config was not read for q_scale.
- `MSE_Q6_normalized_by_q_scale` was calculated as `MSE_Q6 / q_scale^2`.
- Training formulations: first 24 rows from `Formulas-train` and `Release-train`.
- Test formulations: all 6 rows from `Formulas-test` and `Release-test`.
- Train Run No values: `['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']`.
- Excluded original training Run No values: `['S10', 'Opt-2', 'Opt-4', 'Opt-6', 'Opt-7', 'Opt-10']`.
- Test Run No values: `['Opt-2-1', 'Opt-2-4', 'Opt-2-5', 'Opt-2-7', 'Opt-2-8', 'Opt-2-10']`.
- Train/test leakage detected: `False`.

## Models
- Mean train baseline
- Ridge regression
- PLS regression
- Polynomial RSM degree 2
- Random Forest Regressor
- Gaussian Process Regressor

## Test Endpoint Metrics
| model | RMSE_Q6 | MAE_Q6 | R2_Q6 | Spearman_Q6 | Kendall_Q6 | pairwise_accuracy_Q6 | top1_hit_Q6 | top2_hit_Q6 | RMSE_AUC | MAE_AUC | R2_AUC | Spearman_AUC | Kendall_AUC | pairwise_accuracy_AUC | top1_hit_AUC | top2_hit_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLS regression | 803.029 | 782.177 | -29.4886 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2249.05 | 2127.63 | -11.3245 | -0.6 | -0.466667 | 0.266667 | 0 | 0 |
| Ridge regression | 811.338 | 791.042 | -30.1228 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2274.36 | 2155.4 | -11.6034 | -0.6 | -0.466667 | 0.266667 | 0 | 0 |
| Polynomial RSM degree 2 | 847.474 | 823.959 | -32.9569 | -0.714286 | -0.6 | 0.2 | 0 | 0 | 2385.66 | 2255.96 | -12.8672 | -0.828571 | -0.733333 | 0.133333 | 0 | 0 |
| Random Forest Regressor | 936.833 | 918.829 | -40.4954 | -0.845154 | -0.774597 | 0 | 0 | 0 | 2568.87 | 2449.6 | -15.0788 | -0.845154 | -0.774597 | 0 | 0 | 0 |
| Gaussian Process Regressor | 979.991 | 967.32 | -44.4066 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2795.76 | 2693.32 | -18.0446 | -0.428571 | -0.2 | 0.4 | 0 | 0 |
| Mean train baseline | 980.145 | 969.295 | -44.4209 | nan | nan | nan | 0 | 1 | 2780.85 | 2706.05 | -17.842 | nan | nan | nan | 0 | 1 |

## Top-k Formulation IDs
- `PLS regression`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Ridge regression`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Polynomial RSM degree 2`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Random Forest Regressor`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Gaussian Process Regressor`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-4"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-4", "Opt-2-10"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-4"].
- `Mean train baseline`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-1"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-1", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-1"].

## Outputs
- `revision_validation_24train6test/results/static_endpoint_baseline_metrics_24train6test.csv`
- `revision_validation_24train6test/results/static_endpoint_baseline_predictions_24train6test.csv`
- `revision_validation_24train6test/figures/static_q6_parity_24train6test.png`
- `revision_validation_24train6test/figures/static_auc_parity_24train6test.png`
- `revision_validation_24train6test/figures/static_q6_ranking_barplot_24train6test.png`
