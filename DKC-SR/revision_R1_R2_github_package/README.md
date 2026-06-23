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
- static_baseline: 4
- vanilla_odesr: 8



