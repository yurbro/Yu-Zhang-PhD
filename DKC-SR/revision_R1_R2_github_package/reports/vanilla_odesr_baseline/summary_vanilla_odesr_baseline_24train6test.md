# Summary: Vanilla ODE-SR Baseline, 24 Train + 6 Test

- q_scale used: `3008.198194823261`.
- Vanilla ODE-SR used conventional primitives only: add, sub, mul, protected div, pow2.
- `softplus` and DKC-SR physical/domain constraints were absent.
- The best conventional static/curve baseline from the existing 24+6 validation is included for predictive context; structural ODE validity metrics are not applicable to that row.
- The previous constraint-relaxed operator-matched SR is reported only as a supplementary ablation.

## Table 1. Predictive and Structural Comparison

| model_or_group | n_successful_seeds | test_curve_RMSE_mean | test_curve_RMSE_sd | test_curve_RMSE_best | test_curve_R2_best | Q6_RMSE_best | formulation_variable_count | active_sensitivity_count | Q6_range_design_grid | structural_class | valid_for_formulation_optimisation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DKC-SR selected equation | 1 | 461.257 | 0 | 461.257 | 0.796126 | 959.383 | 3 | 3 | 2425.75 | structurally_valid | True |
| Best conventional static/curve baseline (PLS regression) | 1 | 507.957 | 0 | 507.957 | 0.752754 | 803.029 | not_applicable | not_applicable | not_applicable | static_baseline | not_applicable |
| Vanilla ODE-SR mean +/- SD | 5 | 572.86 | 14.4117 | 549.949 | 0.710186 | 914.479 | 0 | 0 | 0 | degenerate_for_optimisation | 0/5 |
| Best vanilla ODE-SR seed 0 by curve RMSE | 1 | 549.949 | 0 | 549.949 | 0.710186 | 914.479 | 0 | 0 | 0 | degenerate_for_optimisation | False |
| Constraint-relaxed operator-matched SR (supplementary) best seed 2 | 1 | 240.566 | 0 | 240.566 | 0.944545 | 373.67 | 3 | 3 | 1396.45 | structurally_valid | True |

## Table 2. Physical and Optimisation Safety

| model_or_group | negative_Q_prediction_rate | non_monotonic_curve_rate | negative_RHS_rate | positive_dfdQ_rate | numerical_failure_rate | extreme_Q6_rate | best_Q6 | best_is_boundary | optimisation_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DKC-SR selected equation | 0 | 0 | 0 | 0.429 | 0 | 0 | 2425.75 | False | 0 |
| Vanilla ODE-SR mean +/- SD | 0 | 0 | 0 | 0 | 0 | 0 | 2191.57 | 5/5 | 0 |
| Best vanilla ODE-SR seed 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2230.74 | True | 0 |
| Constraint-relaxed operator-matched SR (supplementary) best seed 2 | 0 | 0 | 0 | nan | 0 | 0 | 2965.88 | True | 0 |

## Reviewer-Facing Answers

1. Vanilla ODE-SR is a cleaner reviewer-facing conventional SR baseline than the previous operator-matched ablation because it removes the positivity-oriented `softplus` primitive.
2. Best vanilla ODE-SR test curve RMSE was `549.949`.
3. The best vanilla ODE-SR structural class was `degenerate_for_optimisation`.
4. Basic physical checks are reported in Table 2; violations should be interpreted by the rates, not by RMSE alone.
5. The best vanilla ODE-SR positive df/dQ rate was `0`; under this finite-difference audit it does not violate df/dQ <= 0, but this is because the fitted RHS is formulation-independent and constant.
6. The best vanilla ODE-SR grid optimum boundary status was `True`.
7. The previous constraint-relaxed SR should be described as an operator-matched supplementary ablation, not as the main vanilla unconstrained SR baseline.
8. Suggested wording: A conventional vanilla ODE-SR baseline using generic arithmetic primitives was added on the same 24/6 split. This baseline evaluates whether unconstrained ODE symbolic regression can match predictive accuracy while retaining structurally and physically usable formulation dependence. The operator-matched softplus ablation is retained only as supplementary evidence because its operator set still contains positivity-oriented design choices.

## Outputs

- `revision_validation_vanilla_odesr_24train6test/results/table_vanilla_odesr_predictive_structural_comparison.csv`
- `revision_validation_vanilla_odesr_24train6test/results/table_vanilla_odesr_physical_optimisation_comparison.csv`
