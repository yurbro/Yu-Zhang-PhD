# Unified Curve Table With DKC-SR

## Context
- q_scale used: `3008.198194823261`.
- Static curve proxy rows were reused from `revision_validation_24train6test`.
- Primary curve metrics exclude `time_h = 0`.
- DKC-SR was not retrained; existing and replayed DKC-SR predictions were added on the same six held-out formulations.
- Existing `pred_test-six.csv` mapped to canonical 6-test curves: `True`.

## Test Curve Metrics, Excluding t=0
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

## Outputs
- `revision_validation_24train6test_dkcsr/results/unified_curve_metrics_with_dkcsr_24train6test.csv`
- `revision_validation_24train6test_dkcsr/results/unified_curve_predictions_with_dkcsr_24train6test.csv`
- `revision_validation_24train6test_dkcsr/figures/unified_curve_parity_with_dkcsr_24train6test.png`
