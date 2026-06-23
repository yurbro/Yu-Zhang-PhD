# Unified Endpoint Table With DKC-SR

## Context
- q_scale used for DKC-SR replay: `3008.198194823261`.
- Static baseline rows were reused from `revision_validation_24train6test`; they were not regenerated here.
- DKC-SR was not retrained; the selected expression was replayed and existing prediction files were audited/remapped by `Q_obs` curves.
- Existing `pred_test-six.csv` mapped to canonical 6-test curves: `True`.
- Existing `pred_test-six.csv` row order matched canonical order: `False`.

## Test Endpoint Metrics
| model | RMSE_Q6 | MAE_Q6 | R2_Q6 | Spearman_Q6 | Kendall_Q6 | pairwise_accuracy_Q6 | top1_hit_Q6 | top2_hit_Q6 | RMSE_AUC | MAE_AUC | R2_AUC | Spearman_AUC | Kendall_AUC | pairwise_accuracy_AUC | top1_hit_AUC | top2_hit_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLS regression | 803.029 | 782.177 | -29.4886 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2249.05 | 2127.63 | -11.3245 | -0.6 | -0.466667 | 0.266667 | 0 | 0 |
| Ridge regression | 811.338 | 791.042 | -30.1228 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2274.36 | 2155.4 | -11.6034 | -0.6 | -0.466667 | 0.266667 | 0 | 0 |
| Polynomial RSM degree 2 | 847.474 | 823.959 | -32.9569 | -0.714286 | -0.6 | 0.2 | 0 | 0 | 2385.66 | 2255.96 | -12.8672 | -0.828571 | -0.733333 | 0.133333 | 0 | 0 |
| Random Forest Regressor | 936.833 | 918.829 | -40.4954 | -0.845154 | -0.774597 | 0 | 0 | 0 | 2568.87 | 2449.6 | -15.0788 | -0.845154 | -0.774597 | 0 | 0 | 0 |
| DKC-SR existing pred_test-six.csv | 959.296 | 948.207 | -42.5091 | 0.2 | 0.2 | 0.6 | 0 | 1 | 2446.82 | 2361.46 | -13.5873 | -0.2 | -0.0666667 | 0.466667 | 0 | 0 |
| DKC-SR replayed equation, q_scale=975.178 | 959.383 | 948.207 | -42.517 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 1764.26 | 1642.83 | -6.58398 | -0.428571 | -0.2 | 0.4 | 0 | 0 |
| Gaussian Process Regressor | 979.991 | 967.32 | -44.4066 | -0.6 | -0.466667 | 0.266667 | 0 | 0 | 2795.76 | 2693.32 | -18.0446 | -0.428571 | -0.2 | 0.4 | 0 | 0 |
| Mean train baseline | 980.145 | 969.295 | -44.4209 | nan | nan | nan | 0 | 1 | 2780.85 | 2706.05 | -17.842 | nan | nan | nan | 0 | 1 |

## Top-k Sets
- `PLS regression`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Ridge regression`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Polynomial RSM degree 2`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `Random Forest Regressor`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-10"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-10", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-10"].
- `DKC-SR existing pred_test-six.csv`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-8"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-8", "Opt-2-7"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-8"].
- `DKC-SR replayed equation, q_scale=975.178`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-4"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-4", "Opt-2-10"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-4"].
- `Gaussian Process Regressor`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-4"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-4", "Opt-2-10"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-4"].
- `Mean train baseline`: Q6 true_top1=["Opt-2-7"], pred_top1=["Opt-2-1"], true_top2=["Opt-2-7", "Opt-2-1"], pred_top2=["Opt-2-1", "Opt-2-4"]; AUC true_top1=["Opt-2-5"], pred_top1=["Opt-2-1"].

## Outputs
- `revision_validation_24train6test_dkcsr/results/unified_endpoint_metrics_with_dkcsr_24train6test.csv`
- `revision_validation_24train6test_dkcsr/results/unified_endpoint_predictions_with_dkcsr_24train6test.csv`
- `revision_validation_24train6test_dkcsr/figures/unified_q6_parity_with_dkcsr_24train6test.png`
- `revision_validation_24train6test_dkcsr/figures/unified_auc_parity_with_dkcsr_24train6test.png`
- `revision_validation_24train6test_dkcsr/figures/unified_q6_ranking_barplot_with_dkcsr_24train6test.png`
