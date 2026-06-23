# Vanilla ODE-SR Multiseed Run Report

- q_scale used: `3008.198194823261`.
- Run mode: `full`.
- Seeds: `[0, 1, 2, 3, 4]`.
- Population: `300`; generations: `30`; Hall of Fame: `20`.
- Training used only the 24 canonical training formulations; the 6 held-out test formulations were replayed after training.
- Primitive set excludes `softplus`.

## Run Summary

| seed | status | runtime_seconds | complexity | train_RMSE_curve | test_RMSE_curve | test_R2_curve | numerical_failure_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | success | 34.0262 | 3 | 175.314 | 549.949 | 0.710186 | 0 | softplus_absent=True; sympy_export=ok |
| 1 | success | 39.4715 | 2 | 174.832 | 588.755 | 0.667842 | 0 | softplus_absent=True; sympy_export=ok |
| 2 | success | 46.5154 | 3 | 174.001 | 576.553 | 0.681467 | 0 | softplus_absent=True; sympy_export=ok |
| 3 | success | 35.2779 | 3 | 173.922 | 570.397 | 0.688233 | 0 | softplus_absent=True; sympy_export=ok |
| 4 | success | 40.1959 | 3 | 174.079 | 578.648 | 0.679148 | 0 | softplus_absent=True; sympy_export=ok |

## Outputs

- `revision_validation_vanilla_odesr_24train6test/results/vanilla_odesr_multiseed_run_summary.csv`
- `revision_validation_vanilla_odesr_24train6test/runs/seed_<seed>/...`
