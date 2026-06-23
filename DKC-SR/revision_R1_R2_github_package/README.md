# Revision evidence package for DKC-SR dermal formulation manuscript

## Purpose

This folder contains the final evidence files supporting the revised manuscript and response to reviewers, especially R1-2 (small dataset robustness and fixed-structure bootstrap refitting) and R1-5 (same-split baseline comparisons against static regressors and vanilla ODE-SR).

## Folder Structure

- `manuscript/`: manuscript and response documents when available.
- `reports/static_baselines_24train6test/`: same-split static baseline reports.
- `reports/dkcsr_baseline_comparison/`: DKC-SR versus static baseline comparison reports.
- `reports/vanilla_odesr_baseline/`: vanilla ODE-SR baseline reports.
- `reports/bootstrap_b2_only/`: corrected final-equation and b2-only/fixed-parameter robustness evidence.
- `tables/`: selected CSV tables used as revision evidence.
- `figures/`: selected manuscript/revision figures.
- `code/`: scripts and helper/config files used to generate the current revision evidence.

## Key Evidence Summary

### R1-2 Robustness

- Corrected fixed-structure bootstrap refitting was performed with the selected symbolic structure fixed.
- The numerator term `softplus(2 Qtilde)` was treated as structural and was not refitted.
- Only the denominator offset `b2` was refitted.
- Bootstrap success rate: 500/500.
- Bootstrap b2: 2.564 +/- 0.199.
- 95% interval: 2.212 to 2.942.
- Selected final-equation b2 = 2.355029 lies within this interval.
- Bootstrap-refit test curve RMSE: 499.990 +/- 37.032.

### R1-5 Baseline Comparison

- Same 24-training/6-test formulation-level split was used.
- Static baselines included PLS, ridge, polynomial RSM degree 2, RF, and GPR.
- DKC-SR achieved the lowest test curve RMSE among the compared models.
- DKC-SR also achieved the lowest AUC RMSE among the compared models.
- PLS achieved the lowest Q6 endpoint RMSE.
- Vanilla ODE-SR candidates partially fitted the curves but collapsed to formulation-independent expressions.

## Copied File Counts

- bootstrap_b2_only: 4
- dkcsr_comparison: 4
- figure: 4
- static_baseline: 3
- table: 4
- vanilla_odesr: 6

## Code included in this package

The `code/` folder contains scripts used to generate the current revision evidence.

The code files are included for revision traceability and reproducibility of the reported baseline comparisons, vanilla ODE-SR baseline, and corrected b2-only fixed-structure bootstrap analysis. Raw confidential experimental data are not included unless explicitly intended for repository sharing.

| Revision evidence | Code folder |
|---|---|
| Static baseline comparison | `code/static_baselines_24train6test/` |
| DKC-SR vs baseline comparison | `code/dkcsr_baseline_comparison/` |
| Vanilla ODE-SR baseline | `code/vanilla_odesr_baseline/` |
| Corrected b2-only bootstrap | `code/bootstrap_b2_only/` |
| Diagnostic plots and audits | `code/plotting_and_audits/` |

### Code File Counts

- bootstrap_b2_only: 7
- dkcsr_comparison: 4
- packaging_tool: 1
- plotting_audit: 1
- static_baseline: 4
- vanilla_odesr: 8

## Missing Files

- manuscript/Manuscript.docx
- manuscript/Response to Reviewers.docx
- manuscript/Supplementary Material.docx

## Files Requiring Manual Review

- revision_validation_robustness_24train6test/results/dkcsr_fixed_structure_bootstrap_constants.csv (Bootstrap-like CSV outside corrected b2-only package; not copied automatically.)
- revision_validation_robustness_24train6test/results/dkcsr_fixed_structure_bootstrap_metrics.csv (Bootstrap-like CSV outside corrected b2-only package; not copied automatically.)
- revision_validation_robustness_24train6test/results/dkcsr_fixed_structure_bootstrap_test_predictions.csv (Bootstrap-like CSV outside corrected b2-only package; not copied automatically.)
- revision_validation_robustness_24train6test/results/table_bootstrap_constant_stability.csv (Bootstrap-like CSV outside corrected b2-only package; not copied automatically.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/results/30train_b2_bootstrap_constants.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/results/30train_b2_bootstrap_metrics.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/results/30train_b2_bootstrap_test6_prediction_intervals.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/results/table_24train_vs_30train_b2_comparison.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/results/table_30train_b2_bootstrap_summary.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_all30_diagnostic/results/all30_b2_bootstrap_constants.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_all30_diagnostic/results/all30_b2_bootstrap_metrics.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_all30_diagnostic/results/all30_b2_bootstrap_prediction_intervals.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_all30_diagnostic/results/table_all30_b2_bootstrap_summary.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_robustness_24train6test/results/table_dfdq_consistency_summary.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_robustness_24train6test/results/table_repeated_split_baseline_summary.csv (Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/00_build_30train6test_dataset.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/01_30train_b2_refit_sanity_check.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/02_30train_b2_bootstrap.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/03_compare_24train_vs_30train_b2.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/04_make_30train_b2_bootstrap_summary.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_30train6test_diagnostic/_30train_b2_common.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_all30_diagnostic/00_build_all30_dataset.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_all30_diagnostic/01_all30_b2_refit_sanity_check.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_all30_diagnostic/02_all30_b2_bootstrap.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_all30_diagnostic/03_compare_all30_vs_24train_sanity.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_all30_diagnostic/04_make_all30_b2_bootstrap_summary.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_bootstrap_b2_all30_diagnostic/_all30_b2_common.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_qscale3008/run_qscale3008_baseline.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_robustness_24train6test/00_fixed_structure_dkcsr_bootstrap.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_robustness_24train6test/01_repeated_split_static_baselines.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_robustness_24train6test/02_dkcsr_dfdq_consistency_check.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_robustness_24train6test/03_make_robustness_summary.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_robustness_24train6test/_robustness_common.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/00_prepare_unconstrained_sr_config.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/01_run_unconstrained_sr_multiseed.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/02_evaluate_unconstrained_sr_candidates.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/03_physical_plausibility_unconstrained_vs_dkcsr.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/04_optimisation_safety_unconstrained_vs_dkcsr.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/05_make_unconstrained_ablation_summary.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_24train6test/_ucsr_common.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_structural_audit/00_expression_variable_usage_audit.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_structural_audit/01_design_space_sensitivity_audit.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_structural_audit/02_predictive_vs_structural_summary.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)
- revision_validation_unconstrained_sr_structural_audit/_structural_common.py (Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.)

## Notes

This package is intended for manuscript revision traceability. It should not include raw confidential experimental data unless the repository is private and sharing is intentional.

Obsolete and exploratory diagnostics, including all30 bootstrap diagnostics, 30train6test bootstrap diagnostics, old qscale3008 exploratory files, skipped bootstrap reports, and bootstrap versions where the numerator coefficient was refitted, were not copied automatically.
