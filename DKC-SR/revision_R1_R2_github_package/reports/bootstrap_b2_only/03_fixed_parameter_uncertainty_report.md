# Fixed-Parameter Prediction Uncertainty Report

The constant-refit bootstrap was not used because the full-training refit did not recover the final equation constant. Therefore, uncertainty was reported for the fixed published equation without claiming bootstrap constant stability.

| Item | Value |
| --- | --- |
| a_fixed | 2.0 |
| b2_fixed | 2.3550290604627118 |
| train_curve_RMSE_fixed_equation | 574.13392134745595 |
| test_curve_RMSE_fixed_equation | 461.25732513162308 |
| test_curve_R2_fixed_equation | 0.79612597062273305 |
| test_Q6_RMSE_fixed_equation | 959.38312005305374 |
| test_AUC_RMSE_fixed_equation | 1764.2617735097558 |
| heldout_point_interval_coverage | 1 |
| residual_interval_method | empirical time-specific training residual quantiles; t=0 fixed at zero residual |

## Outputs

- `revision_validation_bootstrap_correction_24train6test/results/fixed_parameter_prediction_uncertainty.csv`
- `revision_validation_bootstrap_correction_24train6test/figures/fixed_parameter_prediction_uncertainty.png`
