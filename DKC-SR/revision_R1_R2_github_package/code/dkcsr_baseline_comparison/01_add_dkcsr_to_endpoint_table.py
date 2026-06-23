from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _dkcsr_common import (
    EXISTING_PRED_TEST_SIX,
    FIGURES_DIR,
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    SOURCE_24_RESULTS,
    add_identity,
    endpoint_from_curve,
    endpoint_metrics_table,
    ensure_dirs,
    load_24_canonical,
    map_prediction_file_to_canonical,
    md_table,
    rel,
    replay_dkc_for_curve,
)


def build_dkcsr_endpoint_predictions() -> tuple[pd.DataFrame, dict[str, object]]:
    canon = load_24_canonical()
    curve_test = canon["curve_test"]
    meta: dict[str, object] = {}
    curves: list[pd.DataFrame] = []

    existing, existing_info = map_prediction_file_to_canonical(
        EXISTING_PRED_TEST_SIX,
        curve_test,
        "DKC-SR existing pred_test-six.csv",
    )
    meta["existing_pred_test_six"] = existing_info
    if existing is not None:
        curves.append(existing)

    replay = replay_dkc_for_curve(curve_test, "DKC-SR replayed equation, q_scale=3008.198194823261")
    curves.append(replay)
    dkc_curve = pd.concat(curves, ignore_index=True)
    dkc_endpoint = endpoint_from_curve(dkc_curve)
    return dkc_endpoint, meta


def make_parity(pred: pd.DataFrame, target: str, path) -> None:
    test = pred[pred["dataset"] == "test"]
    plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()
    for model, g in test.groupby("model", sort=False):
        ax.scatter(g[f"{target}_obs"], g[f"{target}_pred"], s=40, alpha=0.75, label=model)
    add_identity(ax, test[f"{target}_obs"].to_numpy(float), test[f"{target}_pred"].to_numpy(float))
    ax.set_xlabel(f"Observed {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.legend(fontsize=6.5, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_ranking_plot(metrics: pd.DataFrame, path) -> None:
    test = metrics[metrics["dataset"] == "test"].copy().sort_values("RMSE_Q6")
    plt.figure(figsize=(8.4, 5.1))
    ax = plt.gca()
    ax.barh(test["model"], test["RMSE_Q6"], color="#4C78A8")
    ax.set_xlabel("Q6 RMSE on 6 held-out formulations")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_report(metrics: pd.DataFrame, meta: dict[str, object]) -> None:
    test = metrics[metrics["dataset"] == "test"].sort_values("RMSE_Q6")
    text: list[str] = []
    text.append("# Unified Endpoint Table With DKC-SR")
    text.append("")
    text.append("## Context")
    text.append(f"- q_scale used for DKC-SR replay: `{Q_SCALE}`.")
    text.append("- Static baseline rows were reused from `revision_validation_24train6test`; they were not regenerated here.")
    text.append("- DKC-SR was not retrained; the selected expression was replayed and existing prediction files were audited/remapped by `Q_obs` curves.")
    text.append(f"- Existing `pred_test-six.csv` mapped to canonical 6-test curves: `{meta.get('existing_pred_test_six', {}).get('loaded')}`.")
    text.append(f"- Existing `pred_test-six.csv` row order matched canonical order: `{meta.get('existing_pred_test_six', {}).get('sequence_matches_canonical_order')}`.")
    text.append("")
    text.append("## Test Endpoint Metrics")
    cols = [
        "model",
        "RMSE_Q6",
        "MAE_Q6",
        "R2_Q6",
        "Spearman_Q6",
        "Kendall_Q6",
        "pairwise_accuracy_Q6",
        "top1_hit_Q6",
        "top2_hit_Q6",
        "RMSE_AUC",
        "MAE_AUC",
        "R2_AUC",
        "Spearman_AUC",
        "Kendall_AUC",
        "pairwise_accuracy_AUC",
        "top1_hit_AUC",
        "top2_hit_AUC",
    ]
    text.extend(md_table(test, cols))
    text.append("")
    text.append("## Top-k Sets")
    for _, row in test.iterrows():
        text.append(
            f"- `{row['model']}`: Q6 true_top1={row['true_top1_Q6']}, pred_top1={row['pred_top1_Q6']}, "
            f"true_top2={row['true_top2_Q6']}, pred_top2={row['pred_top2_Q6']}; "
            f"AUC true_top1={row['true_top1_AUC']}, pred_top1={row['pred_top1_AUC']}."
        )
    text.append("")
    text.append("## Outputs")
    for path in [
        RESULTS_DIR / "unified_endpoint_metrics_with_dkcsr_24train6test.csv",
        RESULTS_DIR / "unified_endpoint_predictions_with_dkcsr_24train6test.csv",
        FIGURES_DIR / "unified_q6_parity_with_dkcsr_24train6test.png",
        FIGURES_DIR / "unified_auc_parity_with_dkcsr_24train6test.png",
        FIGURES_DIR / "unified_q6_ranking_barplot_with_dkcsr_24train6test.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "01_unified_endpoint_with_dkcsr_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    static_pred = pd.read_csv(SOURCE_24_RESULTS / "static_endpoint_baseline_predictions_24train6test.csv")
    static_metrics = pd.read_csv(SOURCE_24_RESULTS / "static_endpoint_baseline_metrics_24train6test.csv")
    dkc_endpoint, meta = build_dkcsr_endpoint_predictions()
    dkc_metrics = endpoint_metrics_table(dkc_endpoint)

    unified_pred = pd.concat([static_pred, dkc_endpoint[static_pred.columns]], ignore_index=True)
    unified_metrics = pd.concat([static_metrics, dkc_metrics[static_metrics.columns]], ignore_index=True)
    unified_pred.to_csv(RESULTS_DIR / "unified_endpoint_predictions_with_dkcsr_24train6test.csv", index=False)
    unified_metrics.to_csv(RESULTS_DIR / "unified_endpoint_metrics_with_dkcsr_24train6test.csv", index=False)

    make_parity(unified_pred, "Q6", FIGURES_DIR / "unified_q6_parity_with_dkcsr_24train6test.png")
    make_parity(unified_pred, "AUC", FIGURES_DIR / "unified_auc_parity_with_dkcsr_24train6test.png")
    make_ranking_plot(unified_metrics, FIGURES_DIR / "unified_q6_ranking_barplot_with_dkcsr_24train6test.png")
    write_report(unified_metrics, meta)
    print("[OK] Unified endpoint table with DKC-SR complete.")
    print(f"[OK] Wrote {rel(RESULTS_DIR / 'unified_endpoint_metrics_with_dkcsr_24train6test.csv')}")


if __name__ == "__main__":
    main()
