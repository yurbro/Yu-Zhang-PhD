# DKC-SR Prediction Provenance Audit, 24 Train + 6 Test

## Scope
- q_scale used for replay: `3008.198194823261`.
- The selected artifact config q_scale was not read or used.
- Replayed selected expression from `artifacts/archive/ivrt-pair-251007/best_sympy.txt`.
- Normalized expression used for replay: `softplus(2*Q)/((Q**2 - C1/C3 + C2)**2 + 2.3550290604627118)`.
- No symbolic-regression search or DKC-SR retraining was run.

## Prediction File Audit
| file_path | number_of_rows | unique_record_indices | unique_time_points | matches_canonical_24_train_curve_set | matches_canonical_6_test_curve_set | matches_canonical_30_train_curve_set | matches_canonical_order_24_train | matches_canonical_order_6_test | appears_to_correspond_to | can_be_safely_used_for_24plus6 | short_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation/pred_train.csv | 192 | 24 | 8 | True | False | False | False | False | 24 training formulations | True | all 24 canonical training curves match after Q_obs curve remapping; row order differs from first-24 Excel order |
| evaluation/pred_test-six.csv | 48 | 6 | 8 | False | True | False | False | False | 6 test formulations | True | all six canonical test curves match after Q_obs curve remapping; row order differs from the 24+6 canonical test file |
| evaluation/pred_test.csv | 96 | 12 | 8 | False | False | False | False | False | 12 test/ranking formulations | False | contains 12 curves; first six may match the held-out test set but the file is not a clean 6-test file |
| artifacts/archive/ivrt-pair-251007/pred_train.csv | 240 | 30 | 8 | False | False | True | False | False | 30 training formulations | False | matches the older 30-training canonical split, not the 24-training split as a whole |
| artifacts/archive/ivrt-pair-251007/pred_test.csv | 48 | 6 | 8 | False | True | False | False | False | 6 test formulations | True | all six canonical test curves match after Q_obs curve remapping; row order differs from the 24+6 canonical test file |

## Replay Versus Existing pred_test-six.csv
- `evaluation/pred_test-six.csv` can be mapped to the same six canonical test formulations by matching `Q_obs` curves.
- Row order matches canonical 24+6 test order: `False`.
- Matched Run No order in file: `['Opt-2-1', 'Opt-2-10', 'Opt-2-4', 'Opt-2-5', 'Opt-2-7', 'Opt-2-8']`.
- Max absolute prediction difference over full curve: `363.9417198757935`.
- Mean absolute prediction difference over full curve: `111.35472909737727`.
- Max absolute prediction difference at Q6: `3.2490172011421237`.
- Q6 agrees within 5 Q units: `True`.
- Full curve agrees within small tolerance: `False`.

## Outputs
- `revision_validation_24train6test_dkcsr/results/dkcsr_prediction_file_audit.csv`
- `revision_validation_24train6test_dkcsr/results/dkcsr_qscale975_replay_vs_existing.csv`
