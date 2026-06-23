# Vanilla ODE-SR Structural Validity Audit

- q_scale used for replay: `3008.198194823261`.
- Design grid: `21 x 21 x 21` over C1, C2, C3.
- Validity rule: Q present after simplification, at least two formulation variables present, at least two active formulation sensitivities, and non-trivial Q6 variation.

## Structural Metrics

| seed | variables_present_after_simplification | formulation_variable_count | active_sensitivity_count | Q6_range_design_grid | Q6_std_design_grid | structural_class | valid_for_formulation_optimisation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 |  | 0 | 0 | 0 | 4.54747e-13 | degenerate_for_optimisation | False |
| 1 |  | 0 | 0 | 0 | 9.09495e-13 | degenerate_for_optimisation | False |
| 2 |  | 0 | 0 | 0 | 0 | degenerate_for_optimisation | False |
| 3 |  | 0 | 0 | 0 | 4.54747e-13 | degenerate_for_optimisation | False |
| 4 |  | 0 | 0 | 0 | 0 | degenerate_for_optimisation | False |

## Outputs

- `results/vanilla_odesr_structural_validity.csv`
- `figures/vanilla_odesr_q6_range_barplot.png`
- `figures/vanilla_odesr_active_sensitivity_barplot.png`
