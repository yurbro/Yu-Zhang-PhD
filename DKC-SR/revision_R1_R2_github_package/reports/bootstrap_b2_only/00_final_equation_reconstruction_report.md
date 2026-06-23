# Final Equation Reconstruction Report

## Reconstruction Result

| Item | Value |
| --- | --- |
| selected numerator is softplus(2*Qtilde) | True |
| a fixed and not refitted | True |
| original b2 | 2.3550290604627118 |
| q_scale | 3008.198194823261 |
| b2 confirmed in selected sources | True |
| excluded rows present in canonical train/test | [] |

## Selected Expression Sources

- `artifacts/archive/ivrt-pair-251007/best_sympy.txt`: `log(exp(2*ARG0) + 1)/((ARG0**2 - ARG1/ARG3 + ARG2)**2 + 2.3550290604627118)`
- `artifacts/archive/ivrt-pair-251007/best_infix.txt`: `(log(1+exp((Q + Q))) / (((C2 - ((C1 / C3) - (Q)**2)))**2 + (-1.5346103937034676)**2))`
- `evaluation/used_expr.txt`: `log(exp(2*Q) + 1)/((Q**2 - C1/C3 + C2)**2 + 2.3550290604627118)`

## Data Files Used

- curve_train: `revision_validation_24train6test/data/canonical_curve_train_24.csv`
- curve_test: `revision_validation_24train6test/data/canonical_curve_test_6.csv`
- endpoint_train: `revision_validation_24train6test/data/canonical_endpoint_train_24.csv`
- endpoint_test: `revision_validation_24train6test/data/canonical_endpoint_test_6.csv`

## Replay Implementation

- Source: `revision_validation_robustness_24train6test/_robustness_common.py` for the normalized RK4 replay and metric conventions, wrapped locally in `revision_validation_bootstrap_correction_24train6test/_corrected_bootstrap_common.py`.
- Corrected replay function: `simulate_corrected_record(rec, b2)` calls the existing replay with `a=2.0` and variable `b2`.
- Normalisation: `Q_raw = Q_SCALE * Qtilde`, with `Q_SCALE = 3008.198194823261`, and normalized `C1n`, `C2n`, `C3n` from the canonical CSVs.
- Integration convention: RK4 stepping between measured time points using the existing revision-validation helper.

## Differences From The Previous Fixed-Structure Bootstrap

- The previous bootstrap refitted both `a` and `b2`.
- This corrected workflow fixes `a = 2.0`, because the selected numerator is `softplus(Q + Q)`, equivalently `softplus(2*Qtilde)`.
- Only the denominator offset `b2` may be refitted in the sanity check or conditional bootstrap.

## Outputs

- `revision_validation_bootstrap_correction_24train6test/config/corrected_bootstrap_config.json`
