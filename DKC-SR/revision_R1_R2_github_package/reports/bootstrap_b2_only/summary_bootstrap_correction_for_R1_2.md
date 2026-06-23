# Bootstrap Correction Summary For R1-2

1. Does the selected expression numerator correspond to `softplus(2*Qtilde)`?
Yes: `True`. The selected sources contain `Q + Q` and/or `2*Q` in the softplus numerator.

2. Was `a` fixed at 2 and excluded from refitting?
Yes. `a_fixed = 2.0` and `a_refitted = False`.

3. What is the original final-equation value of `b2`?
`b2_original = 2.3550290604627118`.

4. Does the full-training refit recover the original `b2`?
No. The best full-training refit was `2.67187`, with absolute difference `0.31684` and relative difference `0.134538`, exceeding the required closeness rule.

5. If yes, what is the corrected bootstrap 95% interval for `b2`, and does it include the original `b2`?
Not applicable. The corrected bootstrap was skipped because the full-training sanity check failed.

6. If no, why should constant-refit bootstrap not be used as constant-stability evidence?
Because the independent fixed-structure refit, with `a` fixed at 2 and only `b2` optimized, did not reproduce the selected final-equation constant on the full training set. A bootstrap around that refit would characterize the refit pipeline's different point estimate, not stability of the published constant.

7. What wording should be used in the response to Reviewer #1 Comment 2?
A constant-refit bootstrap was attempted, but the independent refit pipeline did not reproduce the final-equation constant on the full training set. Therefore, we did not use the constant-refit bootstrap as evidence of constant stability. Instead, we report uncertainty for the fixed final equation and acknowledge this limitation.

8. What wording should be used in the manuscript?
A constant-refit bootstrap was attempted, but the independent refit pipeline did not reproduce the final-equation constant on the full training set. Therefore, we did not use the constant-refit bootstrap as evidence of constant stability. Instead, we report uncertainty for the fixed final equation and acknowledge this limitation.

9. Which previous bootstrap results should be superseded?
The previous `revision_validation_robustness_24train6test` fixed-structure bootstrap results should be superseded for R1-2 because they refitted both `a` and `b2`. In particular, do not use the earlier constants, metrics, interval plots, or report text that describe refitting `a` in `softplus(a * Qtilde)`.

## Files

- `revision_validation_bootstrap_correction_24train6test/reports/00_final_equation_reconstruction_report.md`
- `revision_validation_bootstrap_correction_24train6test/reports/01_full_train_b2_refit_sanity_check_report.md`
- `revision_validation_bootstrap_correction_24train6test/reports/02_corrected_fixed_structure_bootstrap_SKIPPED.md`
- `revision_validation_bootstrap_correction_24train6test/reports/03_fixed_parameter_uncertainty_report.md`
- `revision_validation_bootstrap_correction_24train6test/results/table_corrected_bootstrap_summary.csv`
