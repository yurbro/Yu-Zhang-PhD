# Summary: DKC-SR And Static Baselines, 24 Train + 6 Test

## Required Statements
- q_scale used: `3008.198194823261`.
- The same 24 train + 6 test formulation-level split from `revision_validation_24train6test` was used.
- DKC-SR was added from existing prediction files and replay of the selected expression only; no SR search or DKC-SR retraining was run.
- Existing DKC-SR prediction files were audited by curve-set matching because some row orders differ from the first-24/test Excel order.

## DKC-SR Prediction Provenance
| file_path | appears_to_correspond_to | can_be_safely_used_for_24plus6 | short_notes |
| --- | --- | --- | --- |
| evaluation/pred_train.csv | 24 training formulations | True | all 24 canonical training curves match after Q_obs curve remapping; row order differs from first-24 Excel order |
| evaluation/pred_test-six.csv | 6 test formulations | True | all six canonical test curves match after Q_obs curve remapping; row order differs from the 24+6 canonical test file |
| evaluation/pred_test.csv | 12 test/ranking formulations | False | contains 12 curves; first six may match the held-out test set but the file is not a clean 6-test file |
| artifacts/archive/ivrt-pair-251007/pred_train.csv | 30 training formulations | False | matches the older 30-training canonical split, not the 24-training split as a whole |
| artifacts/archive/ivrt-pair-251007/pred_test.csv | 6 test formulations | True | all six canonical test curves match after Q_obs curve remapping; row order differs from the 24+6 canonical test file |

## Unified Endpoint Metrics
| model | RMSE_Q6 | MAE_Q6 | R2_Q6 | Spearman_Q6 | pairwise_accuracy_Q6 | top1_hit_Q6 | top2_hit_Q6 | RMSE_AUC | Spearman_AUC | pairwise_accuracy_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLS regression | 803.029 | 782.177 | -29.4886 | -0.6 | 0.266667 | 0 | 0 | 2249.05 | -0.6 | 0.266667 |
| Ridge regression | 811.338 | 791.042 | -30.1228 | -0.6 | 0.266667 | 0 | 0 | 2274.36 | -0.6 | 0.266667 |
| Polynomial RSM degree 2 | 847.474 | 823.959 | -32.9569 | -0.714286 | 0.2 | 0 | 0 | 2385.66 | -0.828571 | 0.133333 |
| Random Forest Regressor | 936.833 | 918.829 | -40.4954 | -0.845154 | 0 | 0 | 0 | 2568.87 | -0.845154 | 0 |
| DKC-SR existing pred_test-six.csv | 959.296 | 948.207 | -42.5091 | 0.2 | 0.6 | 0 | 1 | 2446.82 | -0.2 | 0.466667 |
| DKC-SR replayed equation, q_scale=3008.198 | 959.383 | 948.207 | -42.517 | -0.6 | 0.266667 | 0 | 0 | 1764.26 | -0.428571 | 0.4 |
| Gaussian Process Regressor | 979.991 | 967.32 | -44.4066 | -0.6 | 0.266667 | 0 | 0 | 2795.76 | -0.428571 | 0.4 |
| Mean train baseline | 980.145 | 969.295 | -44.4209 | nan | nan | 0 | 1 | 2780.85 | nan | nan |

## Unified Curve Metrics
- Primary curve metrics exclude `time_h = 0`.
| model | RMSE | MAE | R2 | MSE_normalized_by_q_scale | n_points |
| --- | --- | --- | --- | --- | --- |
| DKC-SR replayed equation, q_scale=3008.198 | 461.257 | 325.575 | 0.796126 | 0.223727 | 42 |
| PLS regression | 507.957 | 398.958 | 0.752754 | 0.271323 | 42 |
| Ridge regression | 510.843 | 401.282 | 0.749937 | 0.274414 | 42 |
| DKC-SR existing pred_test-six.csv | 526.031 | 439.627 | 0.734846 | 0.290974 | 42 |
| Random Forest Regressor | 529.597 | 420.637 | 0.731239 | 0.294933 | 42 |
| Polynomial RSM degree 2 | 538.994 | 428.085 | 0.721617 | 0.305491 | 42 |
| Mean train baseline | 570.881 | 464.56 | 0.687704 | 0.342707 | 42 |
| Gaussian Process Regressor | 590.923 | 470.319 | 0.665391 | 0.367193 | 42 |

## Physical Plausibility Audit
| model | negative_Q_prediction_rate | non_monotonic_curve_rate | extreme_Q6_rate | numerical_failure_rate | negative_RHS_rate | top_predicted_Q6 | top_predicted_is_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PLS regression | 0.076 | 0 | 0 | 0 | nan | 2327.03 | True |
| Ridge regression | 0.072 | 0 | 0 | 0 | nan | 2321.14 | True |
| Polynomial RSM degree 2 | 0.075 | 0 | 0 | 0 | nan | 2336.33 | True |
| Random Forest Regressor | 0 | 0 | 0 | 0 | nan | 2577.3 | False |
| Gaussian Process Regressor | 0.098 | 0 | 0 | 0 | nan | 3144.7 | True |
| DKC-SR replayed equation, q_scale=3008.198 | 0 | 0 | 0 | 0 | 0 | 2424.21 | True |

## Recommended Manuscript Interpretation
- The unified endpoint table should be used to show that conventional static baselines are useful comparator models on the same 24+6 split.
- The curve table and physical audit should be used to clarify that DKC-SR is not only an endpoint regressor: it supplies a compact ODE form and can be checked for dynamic admissibility.
- These results support the interpretation that static baselines provide useful endpoint predictors, but they do not encode dynamic release structure or physical constraints. The selected DKC-SR model should therefore be evaluated not only by endpoint RMSE, but also by its compact ODE structure and physically admissible release dynamics.
- In this audit, DKC-SR grid negative-Q rate was `0.0` and non-monotonic curve rate was `0.0`.

## Review Files
- `revision_validation_24train6test_dkcsr/reports/00_dkcsr_24train6test_provenance_report.md`
- `revision_validation_24train6test_dkcsr/reports/01_unified_endpoint_with_dkcsr_report.md`
- `revision_validation_24train6test_dkcsr/reports/02_unified_curve_with_dkcsr_report.md`
- `revision_validation_24train6test_dkcsr/reports/03_physical_plausibility_audit_report.md`
- `revision_validation_24train6test_dkcsr/reports/summary_dkcsr_static_baseline_24train6test_report.md`
- `revision_validation_24train6test_dkcsr/results/unified_endpoint_metrics_with_dkcsr_24train6test.csv`
- `revision_validation_24train6test_dkcsr/results/unified_curve_metrics_with_dkcsr_24train6test.csv`
- `revision_validation_24train6test_dkcsr/results/physical_plausibility_audit_static_vs_dkcsr.csv`
- `revision_validation_24train6test_dkcsr/figures/unified_q6_parity_with_dkcsr_24train6test.png`
- `revision_validation_24train6test_dkcsr/figures/unified_auc_parity_with_dkcsr_24train6test.png`
- `revision_validation_24train6test_dkcsr/figures/unified_curve_parity_with_dkcsr_24train6test.png`
- `revision_validation_24train6test_dkcsr/figures/physical_plausibility_summary_barplot.png`
