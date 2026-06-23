from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _corrected_bootstrap_common import (
    A_FIXED,
    B2_ORIGINAL,
    FIGURES_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    curve_predictions,
    ensure_dirs,
    full_metric_bundle,
    load_canonical,
    md_table,
    records_from_curve,
)


def sanity_passed() -> bool:
    path = RESULTS_DIR / "full_train_b2_refit_sanity_check.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    return bool(df["sanity_pass"].iloc[0]) if "sanity_pass" in df.columns and len(df) else False


def empirical_quantiles(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 50.0)),
        float(np.percentile(values, 97.5)),
    )


def main() -> None:
    ensure_dirs()
    if sanity_passed():
        print("[SKIP] fixed-parameter uncertainty not needed because Part B passed")
        return

    canonical = load_canonical()
    train_records = records_from_curve(canonical["curve_train"])
    test_records = records_from_curve(canonical["curve_test"])
    train_m, train_pred, _ = full_metric_bundle(train_records, b2=B2_ORIGINAL, dataset="train")
    test_m, test_pred, _ = full_metric_bundle(test_records, b2=B2_ORIGINAL, dataset="test")

    train_residuals = train_pred.copy()
    train_residuals["residual"] = train_residuals["Q_obs"] - train_residuals["Q_pred"]
    nonzero = train_residuals.loc[~np.isclose(train_residuals["time_h"], 0.0), "residual"].to_numpy(float)
    overall_stats = empirical_quantiles(nonzero)
    by_time: dict[float, tuple[float, float, float, float, float]] = {}
    for time_h, g in train_residuals.groupby("time_h", sort=True):
        vals = g["residual"].to_numpy(float)
        if np.isclose(float(time_h), 0.0):
            vals = np.asarray([0.0])
        by_time[float(time_h)] = empirical_quantiles(vals)

    rows = []
    for _, row in test_pred.iterrows():
        time_h = float(row["time_h"])
        stats = by_time.get(time_h, overall_stats)
        mean_res, sd_res, lo_res, med_res, hi_res = stats
        lo = float(row["Q_pred"] + lo_res)
        med = float(row["Q_pred"] + med_res)
        hi = float(row["Q_pred"] + hi_res)
        rows.append(
            {
                "Run No": row["run_no"],
                "time_h": time_h,
                "Q_obs": float(row["Q_obs"]),
                "Q_pred_fixed_b2": float(row["Q_pred"]),
                "residual_mean_train_at_time": mean_res,
                "residual_sd_train_at_time": sd_res,
                "Q_pred_p2_5_fixed_parameter_residual": lo,
                "Q_pred_p50_fixed_parameter_residual": med,
                "Q_pred_p97_5_fixed_parameter_residual": hi,
                "interval_contains_Q_obs": bool(lo <= float(row["Q_obs"]) <= hi),
                "a_fixed": A_FIXED,
                "b2_fixed": B2_ORIGINAL,
            }
        )
    uncertainty = pd.DataFrame(rows)
    uncertainty.to_csv(RESULTS_DIR / "fixed_parameter_prediction_uncertainty.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (run_no, g) in zip(axes.ravel(), uncertainty.groupby("Run No", sort=False)):
        gg = g.sort_values("time_h")
        ax.fill_between(
            gg["time_h"],
            gg["Q_pred_p2_5_fixed_parameter_residual"],
            gg["Q_pred_p97_5_fixed_parameter_residual"],
            color="#4c78a8",
            alpha=0.2,
        )
        ax.plot(gg["time_h"], gg["Q_pred_fixed_b2"], color="#4c78a8", linewidth=1.5)
        ax.scatter(gg["time_h"], gg["Q_obs"], color="black", s=18)
        ax.set_title(str(run_no), fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("time_h")
    for ax in axes[:, 0]:
        ax.set_ylabel("Q")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fixed_parameter_prediction_uncertainty.png", dpi=300)
    plt.close(fig)

    coverage = float(uncertainty["interval_contains_Q_obs"].mean())
    lines = [
        "# Fixed-Parameter Prediction Uncertainty Report",
        "",
        "The constant-refit bootstrap was not used because the full-training refit did not recover the final equation constant. Therefore, uncertainty was reported for the fixed published equation without claiming bootstrap constant stability.",
        "",
    ]
    lines.extend(
        md_table(
            [
                ("a_fixed", "2.0"),
                ("b2_fixed", "2.3550290604627118"),
                ("train_curve_RMSE_fixed_equation", train_m["train_curve_RMSE"]),
                ("test_curve_RMSE_fixed_equation", test_m["test_curve_RMSE"]),
                ("test_curve_R2_fixed_equation", test_m["test_curve_R2"]),
                ("test_Q6_RMSE_fixed_equation", test_m["test_Q6_RMSE"]),
                ("test_AUC_RMSE_fixed_equation", test_m["test_AUC_RMSE"]),
                ("heldout_point_interval_coverage", coverage),
                ("residual_interval_method", "empirical time-specific training residual quantiles; t=0 fixed at zero residual"),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `revision_validation_bootstrap_correction_24train6test/results/fixed_parameter_prediction_uncertainty.csv`",
            "- `revision_validation_bootstrap_correction_24train6test/figures/fixed_parameter_prediction_uncertainty.png`",
        ]
    )
    (REPORT_DIR / "03_fixed_parameter_uncertainty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] fixed-parameter uncertainty complete")


if __name__ == "__main__":
    main()
