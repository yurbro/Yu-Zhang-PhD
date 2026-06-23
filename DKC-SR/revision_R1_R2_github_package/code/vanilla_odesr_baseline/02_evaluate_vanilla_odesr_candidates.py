from __future__ import annotations

import pandas as pd

from _vanilla_common import (
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    curve_metrics_table,
    endpoint_from_curve,
    endpoint_metrics_table,
    ensure_dirs,
    expression_complexity,
    get_successful_seed_dirs,
    load_canonical,
    md_table,
    predictions_for_expr,
    rel,
)

MODEL_LABEL = "Vanilla ODE-SR"


def main() -> None:
    ensure_dirs()
    canon = load_canonical()
    curve_rows = []
    endpoint_rows = []
    complexity_rows = []
    for seed_dir in get_successful_seed_dirs():
        seed = int(seed_dir.name.split("_")[-1])
        expr = (seed_dir / "best_expression_infix.txt").read_text(encoding="utf-8").strip()
        train_pred, _ = predictions_for_expr(expr, canon["curve_train"], "train", MODEL_LABEL, seed)
        test_pred, _ = predictions_for_expr(expr, canon["curve_test"], "test", MODEL_LABEL, seed)
        curve_rows.extend([train_pred, test_pred])
        endpoint_rows.append(endpoint_from_curve(pd.concat([train_pred, test_pred], ignore_index=True)))
        complexity_rows.append({"seed": seed, "best_expression": expr, **expression_complexity(expr)})
    if not curve_rows:
        raise RuntimeError("No successful vanilla ODE-SR seed directories found.")
    pred_curve = pd.concat(curve_rows, ignore_index=True)
    pred_endpoint = pd.concat(endpoint_rows, ignore_index=True)
    complexity = pd.DataFrame(complexity_rows)
    curve_metrics = curve_metrics_table(pred_curve).merge(complexity, on="seed", how="left")
    endpoint_metrics = endpoint_metrics_table(pred_endpoint).merge(complexity, on="seed", how="left")

    curve_metrics.to_csv(RESULTS_DIR / "vanilla_odesr_curve_metrics_by_seed.csv", index=False)
    endpoint_metrics.to_csv(RESULTS_DIR / "vanilla_odesr_endpoint_metrics_by_seed.csv", index=False)
    pred_curve.to_csv(RESULTS_DIR / "vanilla_odesr_predictions_curve_all_seeds.csv", index=False)
    pred_endpoint.to_csv(RESULTS_DIR / "vanilla_odesr_predictions_endpoint_all_seeds.csv", index=False)

    test_curve = curve_metrics[(curve_metrics["dataset"] == "test") & (curve_metrics["scope"] == "overall_excluding_t0")].sort_values("RMSE")
    test_endpoint = endpoint_metrics[endpoint_metrics["dataset"] == "test"].sort_values("RMSE_Q6")
    lines = [
        "# Vanilla ODE-SR Candidate Evaluation",
        "",
        f"- q_scale used: `{Q_SCALE}`.",
        "- Every successful seed was replayed on the 24 train and 6 held-out test formulations.",
        "- Primary curve metrics exclude `time_h = 0`.",
        "- Baseline primitive set excludes `softplus`.",
        "",
        "## Test Curve Metrics",
        "",
        *md_table(test_curve, ["seed", "RMSE", "MAE", "R2", "MSE_normalized_by_q_scale", "tree_size", "tree_depth", "number_of_operators"]),
        "",
        "## Test Endpoint Metrics",
        "",
        *md_table(test_endpoint, ["seed", "RMSE_Q6", "MAE_Q6", "R2_Q6", "Spearman_Q6", "Kendall_Q6", "pairwise_accuracy_Q6", "top1_hit_Q6", "top2_hit_Q6", "RMSE_AUC", "Spearman_AUC", "Kendall_AUC", "pairwise_accuracy_AUC", "tree_size"]),
        "",
        "## Outputs",
    ]
    for path in [
        RESULTS_DIR / "vanilla_odesr_curve_metrics_by_seed.csv",
        RESULTS_DIR / "vanilla_odesr_endpoint_metrics_by_seed.csv",
        RESULTS_DIR / "vanilla_odesr_predictions_curve_all_seeds.csv",
        RESULTS_DIR / "vanilla_odesr_predictions_endpoint_all_seeds.csv",
    ]:
        lines.append(f"- `{rel(path)}`")
    (REPORT_DIR / "02_vanilla_odesr_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {rel(RESULTS_DIR / 'vanilla_odesr_curve_metrics_by_seed.csv')}")


if __name__ == "__main__":
    main()
