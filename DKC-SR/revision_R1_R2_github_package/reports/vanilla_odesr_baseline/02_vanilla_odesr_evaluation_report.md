# Vanilla ODE-SR Candidate Evaluation

- q_scale used: `3008.198194823261`.
- Every successful seed was replayed on the 24 train and 6 held-out test formulations.
- Primary curve metrics exclude `time_h = 0`.
- Baseline primitive set excludes `softplus`.

## Test Curve Metrics

| seed | RMSE | MAE | R2 | MSE_normalized_by_q_scale | tree_size | tree_depth | number_of_operators |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 549.949 | 439.417 | 0.710186 | 0.318036 | 3 | 1 | 1 |
| 3 | 570.397 | 457.046 | 0.688233 | 0.342126 | 3 | 1 | 1 |
| 2 | 576.553 | 462.346 | 0.681467 | 0.349551 | 3 | 1 | 1 |
| 4 | 578.648 | 464.149 | 0.679148 | 0.352096 | 3 | 1 | 1 |
| 1 | 588.755 | 472.906 | 0.667842 | 0.364503 | 2 | 1 | 1 |

## Test Endpoint Metrics

| seed | RMSE_Q6 | MAE_Q6 | R2_Q6 | Spearman_Q6 | Kendall_Q6 | pairwise_accuracy_Q6 | top1_hit_Q6 | top2_hit_Q6 | RMSE_AUC | Spearman_AUC | Kendall_AUC | pairwise_accuracy_AUC | tree_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 914.479 | 902.84 | -38.5387 | nan | nan | nan | 0 | 1 | 2665.01 | nan | nan | nan | 3 |
| 3 | 949.029 | 937.82 | -41.5828 | nan | nan | nan | 0 | 1 | 2766.99 | nan | nan | nan | 3 |
| 2 | 959.423 | 948.336 | -42.5206 | nan | nan | nan | 0 | 1 | 2797.69 | nan | nan | nan | 3 |
| 4 | 962.959 | 951.913 | -42.842 | nan | nan | nan | 0 | 1 | 2808.14 | nan | nan | nan | 3 |
| 1 | 980.013 | 969.161 | -44.4086 | nan | nan | nan | 0 | 1 | 2858.54 | nan | nan | nan | 2 |

## Outputs
- `revision_validation_vanilla_odesr_24train6test/results/vanilla_odesr_curve_metrics_by_seed.csv`
- `revision_validation_vanilla_odesr_24train6test/results/vanilla_odesr_endpoint_metrics_by_seed.csv`
- `revision_validation_vanilla_odesr_24train6test/results/vanilla_odesr_predictions_curve_all_seeds.csv`
- `revision_validation_vanilla_odesr_24train6test/results/vanilla_odesr_predictions_endpoint_all_seeds.csv`
