# Physical and Optimisation Safety: Vanilla ODE-SR vs DKC-SR

- q_scale used: `3008.198194823261`.
- Physical rates used a 5 x 5 x 5 grid with 8 replay time points.
- Optimisation safety used a 21 x 21 x 21 grid over the formulation design space.
- Extreme Q6 upper bound: `4957.703371007806`.
- `positive_dfdQ_rate` reports finite-difference violations of the intended nonpositive df/dQ dynamic prior.
- The DKC-SR row is a replay audit of the selected equation, not a reread of training-time constraint logs.

## Audit Metrics

| model | seed | negative_Q_prediction_rate | non_monotonic_curve_rate | negative_RHS_rate | positive_dfdQ_rate | numerical_failure_rate | extreme_Q6_rate | best_C1 | best_C2 | best_C3 | best_Q6 | best_is_boundary | failure_rate | extreme_Q6_rate_optimisation_grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DKC-SR selected equation | selected | 0 | 0 | 0 | 0.429 | 0 | 0 | 22.5 | 15 | 11.5 | 2425.75 | False | 0 | 0 |
| Vanilla ODE-SR | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 10 | 2230.74 | True | 0 | 0 |
| Vanilla ODE-SR | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 10 | 2164.42 | True | 0 | 0 |
| Vanilla ODE-SR | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 10 | 2185.25 | True | 0 | 0 |
| Vanilla ODE-SR | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 10 | 2195.76 | True | 0 | 0 |
| Vanilla ODE-SR | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 10 | 2181.67 | True | 0 | 0 |

## Outputs

- `revision_validation_vanilla_odesr_24train6test/results/vanilla_vs_dkcsr_physical_optimisation_audit.csv`
- `revision_validation_vanilla_odesr_24train6test/figures/vanilla_vs_dkcsr_physical_rates_barplot.png`
- `revision_validation_vanilla_odesr_24train6test/figures/vanilla_vs_dkcsr_best_q6_barplot.png`
