# 24 Train + 6 Test Dataset Report

## Split Definition
- q_scale used for revision-validation context: `3008.198194823261`.
- Training set: first 24 rows from `Formulas-train` and first 24 rows from `Release-train`.
- Test set: all 6 rows from `Formulas-test` and all 6 rows from `Release-test`.
- The remaining 6 rows from the original training sheets were excluded from training.
- Split level: formulation-level by `Run No`; each formulation contributes an entire curve to one split.

## Run No Values
- First 24 training formulations used: `['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']`.
- Excluded 6 original training rows: `['S10', 'Opt-2', 'Opt-4', 'Opt-6', 'Opt-7', 'Opt-10']`.
- Held-out 6 test formulations: `['Opt-2-1', 'Opt-2-4', 'Opt-2-5', 'Opt-2-7', 'Opt-2-8', 'Opt-2-10']`.

## Leakage Checks
- Train/test `Run No` overlap: `[]`.
- Train/test formulation-condition overlap: `[]`.
- Any train/test leakage detected: `no`.
- Excluded rows accidentally present in train: `[]`.
- Excluded rows present in held-out test: `[]`.

## Canonical Data
- Time points: `[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`.
- `Q(0)=0` was added internally to every curve.
- `R_t1..R_t7` were mapped to `0.5, 1, 2, 3, 4, 5, 6` hours.
- AUC was calculated by trapezoidal integration over the full curve including `Q(0)=0`.
- Train curve rows: `192`; train endpoint rows: `24`.
- Test curve rows: `48`; test endpoint rows: `6`.

## Output Files
- `revision_validation_24train6test/data/canonical_curve_train_24.csv`
- `revision_validation_24train6test/data/canonical_curve_test_6.csv`
- `revision_validation_24train6test/data/canonical_endpoint_train_24.csv`
- `revision_validation_24train6test/data/canonical_endpoint_test_6.csv`
- `revision_validation_24train6test/config/revision_24train6test_config.json`
- `revision_validation_24train6test/reports/00_dataset_24train6test_report.md`

## Endpoint Preview
| dataset | run_no | C1 | C2 | C3 | Q6_obs | AUC_obs |
| --- | --- | --- | --- | --- | --- | --- |
| train | F1 | 30 | 20 | 15 | 1842.92 | 5863.34 |
| train | F2 | 30 | 15 | 10 | 2082.6 | 6393.19 |
| train | F3 | 25 | 20 | 20 | 1845.68 | 5903.51 |
| train | F4 | 25 | 15 | 15 | 2295.68 | 6500.79 |
| train | F5 | 25 | 10 | 20 | 2016.72 | 6165.21 |
| test | Opt-2-1 | 20 | 10.1 | 19.99 | 3239.85 | 9904.09 |
| test | Opt-2-4 | 20 | 10 | 15.43 | 3058.49 | 8909.05 |
| test | Opt-2-5 | 20 | 10.08 | 19.97 | 3234.4 | 10071.7 |
| test | Opt-2-7 | 20 | 10.11 | 19.97 | 3305.14 | 9322.18 |
| test | Opt-2-8 | 20 | 10.15 | 19.98 | 3092.06 | 9329.64 |
| test | Opt-2-10 | 20 | 10.07 | 12.53 | 2871.56 | 8137.9 |
