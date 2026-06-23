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
    curve_metrics_table,
    ensure_dirs,
    load_24_canonical,
    map_prediction_file_to_canonical,
    md_table,
    rel,
    replay_dkc_for_curve,
)


def build_dkcsr_curve_predictions() -> tuple[pd.DataFrame, dict[str, object]]:
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
    curves.append(replay_dkc_for_curve(curve_test, "DKC-SR replayed equation, q_scale=3008.198194823261"))
    return pd.concat(curves, ignore_index=True), meta


def make_curve_parity(pred: pd.DataFrame, path) -> None:
    test = pred[(pred["dataset"] == "test") & (~np.isclose(pred["time_h"], 0.0))]
    plt.figure(figsize=(7.6, 6.2))
    ax = plt.gca()
    for model, g in test.groupby("model", sort=False):
        ax.scatter(g["Q_obs"], g["Q_pred"], s=25, alpha=0.68, label=model)
    add_identity(ax, test["Q_obs"].to_numpy(float), test["Q_pred"].to_numpy(float))
    ax.set_xlabel("Observed Q")
    ax.set_ylabel("Predicted Q")
    ax.legend(fontsize=6.5, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_report(metrics: pd.DataFrame, meta: dict[str, object]) -> None:
    test = metrics[(metrics["dataset"] == "test") & (metrics["scope"] == "overall_excluding_t0")].sort_values("RMSE")
    text: list[str] = []
    text.append("# Unified Curve Table With DKC-SR")
    text.append("")
    text.append("## Context")
    text.append(f"- q_scale used: `{Q_SCALE}`.")
    text.append("- Static curve proxy rows were reused from `revision_validation_24train6test`.")
    text.append("- Primary curve metrics exclude `time_h = 0`.")
    text.append("- DKC-SR was not retrained; existing and replayed DKC-SR predictions were added on the same six held-out formulations.")
    text.append(f"- Existing `pred_test-six.csv` mapped to canonical 6-test curves: `{meta.get('existing_pred_test_six', {}).get('loaded')}`.")
    text.append("")
    text.append("## Test Curve Metrics, Excluding t=0")
    text.extend(md_table(test, ["model", "RMSE", "MAE", "R2", "MSE_normalized_by_q_scale", "n_points"]))
    text.append("")
    text.append("## Outputs")
    for path in [
        RESULTS_DIR / "unified_curve_metrics_with_dkcsr_24train6test.csv",
        RESULTS_DIR / "unified_curve_predictions_with_dkcsr_24train6test.csv",
        FIGURES_DIR / "unified_curve_parity_with_dkcsr_24train6test.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "02_unified_curve_with_dkcsr_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    static_pred = pd.read_csv(SOURCE_24_RESULTS / "curve_proxy_baseline_predictions_24train6test.csv")
    static_metrics = pd.read_csv(SOURCE_24_RESULTS / "curve_proxy_baseline_metrics_24train6test.csv")
    dkc_curve, meta = build_dkcsr_curve_predictions()
    dkc_metrics = curve_metrics_table(dkc_curve)
    unified_pred = pd.concat([static_pred, dkc_curve[static_pred.columns]], ignore_index=True)
    unified_metrics = pd.concat([static_metrics, dkc_metrics[static_metrics.columns]], ignore_index=True)
    unified_pred.to_csv(RESULTS_DIR / "unified_curve_predictions_with_dkcsr_24train6test.csv", index=False)
    unified_metrics.to_csv(RESULTS_DIR / "unified_curve_metrics_with_dkcsr_24train6test.csv", index=False)
    make_curve_parity(unified_pred, FIGURES_DIR / "unified_curve_parity_with_dkcsr_24train6test.png")
    write_report(unified_metrics, meta)
    print("[OK] Unified curve table with DKC-SR complete.")
    print(f"[OK] Wrote {rel(RESULTS_DIR / 'unified_curve_metrics_with_dkcsr_24train6test.csv')}")


if __name__ == "__main__":
    main()
