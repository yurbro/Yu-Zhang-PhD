from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _dkcsr_common import (
    FEATURES_CURVE,
    FIGURES_DIR,
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    STATIC_PLAUSIBILITY_MODELS,
    build_dkc_callable,
    build_grid,
    ensure_dirs,
    fit_static_curve_models,
    is_boundary_row,
    load_24_canonical,
    md_table,
    rel,
    simulate_dkc_times,
)


DKC_MODEL = "DKC-SR replayed equation, q_scale=3008.198194823261"


def predict_static_grid(curve_train: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    models = fit_static_curve_models(curve_train)
    rows: list[pd.DataFrame] = []
    X_grid = grid[FEATURES_CURVE].to_numpy(float)
    for model_name in STATIC_PLAUSIBILITY_MODELS:
        pred = np.asarray(models[model_name].predict(X_grid), dtype=float).reshape(-1)
        base = grid.copy()
        base["model"] = model_name
        base["Q_pred"] = pred
        rows.append(base)
    return pd.concat(rows, ignore_index=True)


def predict_dkcsr_grid(grid: pd.DataFrame, qcap_raw: float = 1_000_000.0) -> tuple[pd.DataFrame, float]:
    f, _, _ = build_dkc_callable()
    rows: list[pd.DataFrame] = []
    rhs_values: list[float] = []
    for grid_id, g in grid.groupby("grid_id", sort=True):
        gg = g.sort_values("time_h").copy()
        pred = simulate_dkc_times(
            f,
            gg["time_h"].to_numpy(float),
            float(gg["C1n"].iloc[0]),
            float(gg["C2n"].iloc[0]),
            float(gg["C3n"].iloc[0]),
            qcap_raw=qcap_raw,
        )
        gg["model"] = DKC_MODEL
        gg["Q_pred"] = pred
        for q_raw in pred:
            if np.isfinite(q_raw):
                rhs_values.append(float(f(float(q_raw) / Q_SCALE, float(gg["C1n"].iloc[0]), float(gg["C2n"].iloc[0]), float(gg["C3n"].iloc[0]))))
        rows.append(gg)
    rhs = np.asarray(rhs_values, dtype=float)
    negative_rhs_rate = float(np.mean(rhs < -1e-12)) if rhs.size else float("nan")
    return pd.concat(rows, ignore_index=True), negative_rhs_rate


def plausibility_table(grid_pred: pd.DataFrame, observed_q6_upper: float, dkc_negative_rhs_rate: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, g in grid_pred.groupby("model", sort=False):
        finite = np.isfinite(g["Q_pred"].to_numpy(float))
        negative_q = (g["Q_pred"].to_numpy(float) < -1e-12) & finite
        numerical_failure_rate = float(np.mean(~finite))

        nonmono_count = 0
        curve_count = 0
        q6_extreme_count = 0
        q6_count = 0
        q6_rows = []
        for grid_id, gg in g.groupby("grid_id", sort=True):
            curve_count += 1
            s = gg.sort_values("time_h")
            q = s["Q_pred"].to_numpy(float)
            q_finite = np.isfinite(q)
            if np.all(q_finite) and np.any(np.diff(q) < -1e-9):
                nonmono_count += 1
            q6 = s.loc[np.isclose(s["time_h"], 6.0), "Q_pred"]
            if len(q6):
                q6_count += 1
                q6_val = float(q6.iloc[0])
                if np.isfinite(q6_val) and (q6_val < 0.0 or q6_val > observed_q6_upper):
                    q6_extreme_count += 1
                row = s.loc[np.isclose(s["time_h"], 6.0)].iloc[0].copy()
                q6_rows.append(row)

        q6_df = pd.DataFrame(q6_rows)
        q6_finite = q6_df[np.isfinite(q6_df["Q_pred"].to_numpy(float))].copy()
        if len(q6_finite):
            top = q6_finite.sort_values("Q_pred", ascending=False).iloc[0]
            top_grid_id = int(top["grid_id"])
            top_c1 = float(top["C1"])
            top_c2 = float(top["C2"])
            top_c3 = float(top["C3"])
            top_q6 = float(top["Q_pred"])
            top_is_boundary = bool(is_boundary_row(top))
        else:
            top_grid_id = -1
            top_c1 = top_c2 = top_c3 = top_q6 = float("nan")
            top_is_boundary = False

        rows.append(
            {
                "model": model,
                "n_grid_formulations": int(curve_count),
                "n_grid_predictions": int(len(g)),
                "negative_Q_prediction_rate": float(np.mean(negative_q)) if len(g) else float("nan"),
                "non_monotonic_curve_rate": float(nonmono_count / curve_count) if curve_count else float("nan"),
                "extreme_Q6_rate": float(q6_extreme_count / q6_count) if q6_count else float("nan"),
                "numerical_failure_rate": numerical_failure_rate,
                "top_predicted_grid_id": top_grid_id,
                "top_predicted_C1": top_c1,
                "top_predicted_C2": top_c2,
                "top_predicted_C3": top_c3,
                "top_predicted_Q6": top_q6,
                "top_predicted_is_boundary": top_is_boundary,
                "negative_RHS_rate": dkc_negative_rhs_rate if model == DKC_MODEL else float("nan"),
                "observed_Q6_upper_bound": observed_q6_upper,
            }
        )
    return pd.DataFrame(rows)


def make_summary_barplot(audit: pd.DataFrame) -> None:
    metrics = ["negative_Q_prediction_rate", "non_monotonic_curve_rate", "extreme_Q6_rate", "numerical_failure_rate"]
    plot = audit[["model"] + metrics].melt(id_vars="model", var_name="check", value_name="rate")
    pivot = plot.pivot(index="model", columns="check", values="rate").fillna(0.0)
    ax = pivot.plot(kind="bar", figsize=(10.5, 5.5), width=0.78)
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(pivot.to_numpy())) * 1.1))
    ax.legend(fontsize=7, loc="upper right")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physical_plausibility_summary_barplot.png", dpi=300)
    plt.close()


def write_physical_report(audit: pd.DataFrame, observed_q6_upper: float) -> None:
    text: list[str] = []
    text.append("# Physical Plausibility Audit")
    text.append("")
    text.append("## Context")
    text.append(f"- q_scale used for DKC-SR replay: `{Q_SCALE}`.")
    text.append("- Static curve proxy models were refit on the same 24 training formulations only to predict the design-space grid.")
    text.append("- No symbolic-regression search or DKC-SR retraining was run.")
    text.append("- Grid: 5 x 5 x 5 formulation points over C1 in [20, 30], C2 in [10, 20], C3 in [10, 20], with 8 time points.")
    text.append(f"- Extreme Q6 upper bound: `1.5 * max observed train+test Q6 = {observed_q6_upper}`.")
    text.append("")
    text.append("## Plausibility Metrics")
    cols = [
        "model",
        "negative_Q_prediction_rate",
        "non_monotonic_curve_rate",
        "extreme_Q6_rate",
        "numerical_failure_rate",
        "negative_RHS_rate",
        "top_predicted_C1",
        "top_predicted_C2",
        "top_predicted_C3",
        "top_predicted_Q6",
        "top_predicted_is_boundary",
    ]
    text.extend(md_table(audit, cols))
    text.append("")
    text.append("## Notes")
    text.append("- The DKC-SR negative RHS rate is evaluated at DKC-SR-predicted grid states.")
    text.append("- Static curve proxy models do not provide an ODE RHS, so `negative_RHS_rate` is not applicable for them.")
    text.append("- Boundary optimum checks are descriptive; a boundary optimum is not automatically invalid, but it is useful context for optimisation claims.")
    text.append("")
    text.append("## Outputs")
    for path in [
        RESULTS_DIR / "physical_plausibility_audit_static_vs_dkcsr.csv",
        RESULTS_DIR / "grid_predictions_static_vs_dkcsr.csv",
        FIGURES_DIR / "physical_plausibility_summary_barplot.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "03_physical_plausibility_audit_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_summary_report() -> None:
    endpoint = pd.read_csv(RESULTS_DIR / "unified_endpoint_metrics_with_dkcsr_24train6test.csv")
    curve = pd.read_csv(RESULTS_DIR / "unified_curve_metrics_with_dkcsr_24train6test.csv")
    physical = pd.read_csv(RESULTS_DIR / "physical_plausibility_audit_static_vs_dkcsr.csv")
    provenance = pd.read_csv(RESULTS_DIR / "dkcsr_prediction_file_audit.csv")

    endpoint_test = endpoint[endpoint["dataset"] == "test"].sort_values("RMSE_Q6")
    curve_test = curve[(curve["dataset"] == "test") & (curve["scope"] == "overall_excluding_t0")].sort_values("RMSE")
    dkc_physical = physical[physical["model"].eq(DKC_MODEL)]
    dkc_nonmono = float(dkc_physical["non_monotonic_curve_rate"].iloc[0]) if len(dkc_physical) else math.nan
    dkc_negative = float(dkc_physical["negative_Q_prediction_rate"].iloc[0]) if len(dkc_physical) else math.nan

    text: list[str] = []
    text.append("# Summary: DKC-SR And Static Baselines, 24 Train + 6 Test")
    text.append("")
    text.append("## Required Statements")
    text.append(f"- q_scale used: `{Q_SCALE}`.")
    text.append("- The same 24 train + 6 test formulation-level split from `revision_validation_24train6test` was used.")
    text.append("- DKC-SR was added from existing prediction files and replay of the selected expression only; no SR search or DKC-SR retraining was run.")
    text.append("- Existing DKC-SR prediction files were audited by curve-set matching because some row orders differ from the first-24/test Excel order.")
    text.append("")
    text.append("## DKC-SR Prediction Provenance")
    prov_cols = ["file_path", "appears_to_correspond_to", "can_be_safely_used_for_24plus6", "short_notes"]
    text.extend(md_table(provenance, prov_cols))
    text.append("")
    text.append("## Unified Endpoint Metrics")
    endpoint_cols = [
        "model",
        "RMSE_Q6",
        "MAE_Q6",
        "R2_Q6",
        "Spearman_Q6",
        "pairwise_accuracy_Q6",
        "top1_hit_Q6",
        "top2_hit_Q6",
        "RMSE_AUC",
        "Spearman_AUC",
        "pairwise_accuracy_AUC",
    ]
    text.extend(md_table(endpoint_test, endpoint_cols))
    text.append("")
    text.append("## Unified Curve Metrics")
    text.append("- Primary curve metrics exclude `time_h = 0`.")
    text.extend(md_table(curve_test, ["model", "RMSE", "MAE", "R2", "MSE_normalized_by_q_scale", "n_points"]))
    text.append("")
    text.append("## Physical Plausibility Audit")
    physical_cols = [
        "model",
        "negative_Q_prediction_rate",
        "non_monotonic_curve_rate",
        "extreme_Q6_rate",
        "numerical_failure_rate",
        "negative_RHS_rate",
        "top_predicted_Q6",
        "top_predicted_is_boundary",
    ]
    text.extend(md_table(physical, physical_cols))
    text.append("")
    text.append("## Recommended Manuscript Interpretation")
    text.append("- The unified endpoint table should be used to show that conventional static baselines are useful comparator models on the same 24+6 split.")
    text.append("- The curve table and physical audit should be used to clarify that DKC-SR is not only an endpoint regressor: it supplies a compact ODE form and can be checked for dynamic admissibility.")
    text.append(
        "- These results support the interpretation that static baselines provide useful endpoint predictors, but they do not encode dynamic release structure or physical constraints. "
        "The selected DKC-SR model should therefore be evaluated not only by endpoint RMSE, but also by its compact ODE structure and physically admissible release dynamics."
    )
    text.append(f"- In this audit, DKC-SR grid negative-Q rate was `{dkc_negative}` and non-monotonic curve rate was `{dkc_nonmono}`.")
    text.append("")
    text.append("## Review Files")
    for path in [
        REPORT_DIR / "00_dkcsr_24train6test_provenance_report.md",
        REPORT_DIR / "01_unified_endpoint_with_dkcsr_report.md",
        REPORT_DIR / "02_unified_curve_with_dkcsr_report.md",
        REPORT_DIR / "03_physical_plausibility_audit_report.md",
        REPORT_DIR / "summary_dkcsr_static_baseline_24train6test_report.md",
        RESULTS_DIR / "unified_endpoint_metrics_with_dkcsr_24train6test.csv",
        RESULTS_DIR / "unified_curve_metrics_with_dkcsr_24train6test.csv",
        RESULTS_DIR / "physical_plausibility_audit_static_vs_dkcsr.csv",
        FIGURES_DIR / "unified_q6_parity_with_dkcsr_24train6test.png",
        FIGURES_DIR / "unified_auc_parity_with_dkcsr_24train6test.png",
        FIGURES_DIR / "unified_curve_parity_with_dkcsr_24train6test.png",
        FIGURES_DIR / "physical_plausibility_summary_barplot.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "summary_dkcsr_static_baseline_24train6test_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    canon = load_24_canonical()
    curve_train = canon["curve_train"]
    endpoint_all = pd.concat([canon["endpoint_train"], canon["endpoint_test"]], ignore_index=True)
    observed_q6_upper = 1.5 * float(endpoint_all["Q6_obs"].max())
    grid = build_grid()
    static_grid = predict_static_grid(curve_train, grid)
    dkc_grid, dkc_negative_rhs_rate = predict_dkcsr_grid(grid)
    grid_pred = pd.concat([static_grid, dkc_grid], ignore_index=True)
    audit = plausibility_table(grid_pred, observed_q6_upper, dkc_negative_rhs_rate)
    grid_pred.to_csv(RESULTS_DIR / "grid_predictions_static_vs_dkcsr.csv", index=False)
    audit.to_csv(RESULTS_DIR / "physical_plausibility_audit_static_vs_dkcsr.csv", index=False)
    make_summary_barplot(audit)
    write_physical_report(audit, observed_q6_upper)
    write_summary_report()
    print("[OK] Physical plausibility audit complete.")
    print(f"[OK] Wrote {rel(RESULTS_DIR / 'physical_plausibility_audit_static_vs_dkcsr.csv')}")


if __name__ == "__main__":
    main()
