from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _corrected_bootstrap_common import (
    A_FIXED,
    B2_ORIGINAL,
    FIGURES_DIR,
    N_BOOTSTRAP,
    RANDOM_SEED,
    REPORT_DIR,
    RESULTS_DIR,
    curve_metric_dict,
    curve_predictions,
    ensure_dirs,
    fit_b2_from_initial,
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


def write_skipped_report(reason: str) -> None:
    lines = [
        "# Corrected Fixed-Structure Bootstrap Skipped",
        "",
        reason,
        "",
        "The corrected bootstrap was conditional on the full-training refit recovering the selected final-equation value of `b2` while keeping `a = 2` fixed. Because that condition was not met, no constant-refit bootstrap results are reported as constant-stability evidence.",
    ]
    (REPORT_DIR / "02_corrected_fixed_structure_bootstrap_SKIPPED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not sanity_passed():
        write_skipped_report(
            "Part B did not pass the closeness rule, so the 500-replicate constant-refit bootstrap was not run."
        )
        print("[SKIP] corrected fixed-structure bootstrap")
        return

    start = time.time()
    canonical = load_canonical()
    train_records = records_from_curve(canonical["curve_train"])
    test_records = records_from_curve(canonical["curve_test"])
    rng = np.random.default_rng(RANDOM_SEED)
    constants_rows: list[dict] = []
    metrics_rows: list[dict] = []
    test_pred_long: list[pd.DataFrame] = []
    n_train = len(train_records)

    for boot_id in range(N_BOOTSTRAP):
        sample_idx = rng.integers(0, n_train, size=n_train)
        sampled = [train_records[int(i)] for i in sample_idx]
        fit = fit_b2_from_initial(sampled, B2_ORIGINAL, maxiter=80)
        b2 = float(fit["b2_refit"])
        success = bool(fit["fit_success"])
        constants_rows.append(
            {
                "bootstrap_id": boot_id,
                "fit_success": success,
                "failure_reason": "" if success else fit["failure_reason"],
                "a_fixed": A_FIXED,
                "b2_refit": b2,
                "objective_normalized_mse": fit["objective_normalized_mse"],
                "n_unique_train_formulations": len(set(map(int, sample_idx))),
            }
        )
        row = {
            "bootstrap_id": boot_id,
            "fit_success": success,
            "failure_reason": "" if success else fit["failure_reason"],
            "b2_refit": b2,
        }
        if success and np.isfinite(b2):
            boot_curve, _ = curve_metric_dict(sampled, b2=b2, dataset="bootstrap_sample")
            train_m, _, _ = full_metric_bundle(train_records, b2=b2, dataset="train")
            test_m, test_pred, _ = full_metric_bundle(test_records, b2=b2, dataset="test")
            row.update(
                {
                    "train_curve_RMSE_bootstrap_sample": boot_curve["RMSE"],
                    "train_curve_RMSE_full_train": train_m["train_curve_RMSE"],
                    "test_curve_RMSE": test_m["test_curve_RMSE"],
                    "test_curve_R2": test_m["test_curve_R2"],
                    "test_Q6_RMSE": test_m["test_Q6_RMSE"],
                    "test_AUC_RMSE": test_m["test_AUC_RMSE"],
                    "test_Q6_pairwise_accuracy": test_m["test_pairwise_accuracy_Q6"],
                    "test_Q6_top2_hit": test_m["test_top2_hit_Q6"],
                }
            )
            test_pred_long.append(test_pred.assign(bootstrap_id=boot_id))
        metrics_rows.append(row)
        if (boot_id + 1) % 50 == 0:
            print(f"[bootstrap] {boot_id + 1}/{N_BOOTSTRAP}", flush=True)

    constants = pd.DataFrame(constants_rows)
    metrics = pd.DataFrame(metrics_rows)
    constants.to_csv(RESULTS_DIR / "corrected_bootstrap_b2_constants.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "corrected_bootstrap_metrics.csv", index=False)

    original_pred = curve_predictions(test_records, b2=B2_ORIGINAL, dataset="test_original")
    all_test_pred = pd.concat(test_pred_long, ignore_index=True) if test_pred_long else pd.DataFrame()
    if len(all_test_pred):
        interval = (
            all_test_pred.groupby(["run_no", "time_h", "Q_obs"], as_index=False)["Q_pred"]
            .agg(
                Q_pred_mean_bootstrap="mean",
                Q_pred_sd_bootstrap="std",
                Q_pred_p2_5_bootstrap=lambda x: float(np.percentile(x, 2.5)),
                Q_pred_p50_bootstrap=lambda x: float(np.percentile(x, 50.0)),
                Q_pred_p97_5_bootstrap=lambda x: float(np.percentile(x, 97.5)),
            )
            .rename(columns={"run_no": "Run No"})
        )
        original_keep = original_pred[["run_no", "time_h", "Q_pred"]].rename(
            columns={"run_no": "Run No", "Q_pred": "Q_pred_original_b2"}
        )
        interval = interval.merge(original_keep, on=["Run No", "time_h"], how="left")
        interval = interval[
            [
                "Run No",
                "time_h",
                "Q_obs",
                "Q_pred_original_b2",
                "Q_pred_mean_bootstrap",
                "Q_pred_sd_bootstrap",
                "Q_pred_p2_5_bootstrap",
                "Q_pred_p50_bootstrap",
                "Q_pred_p97_5_bootstrap",
            ]
        ]
    else:
        interval = pd.DataFrame()
    interval.to_csv(RESULTS_DIR / "corrected_bootstrap_test_prediction_intervals.csv", index=False)

    ok = constants[constants["fit_success"]].copy()
    plt.figure(figsize=(6.0, 4.0))
    plt.hist(ok["b2_refit"].dropna(), bins=30, color="#4c78a8", alpha=0.85)
    plt.axvline(B2_ORIGINAL, color="black", linestyle="--", linewidth=1)
    plt.xlabel("b2")
    plt.ylabel("Bootstrap count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "corrected_bootstrap_b2_hist.png", dpi=300)
    plt.close()

    if len(interval):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for ax, (run_no, g) in zip(axes.ravel(), interval.groupby("Run No", sort=False)):
            gg = g.sort_values("time_h")
            ax.fill_between(gg["time_h"], gg["Q_pred_p2_5_bootstrap"], gg["Q_pred_p97_5_bootstrap"], color="#4c78a8", alpha=0.2)
            ax.plot(gg["time_h"], gg["Q_pred_mean_bootstrap"], color="#4c78a8", linewidth=1.5)
            ax.plot(gg["time_h"], gg["Q_pred_original_b2"], color="#f58518", linestyle="--", linewidth=1.2)
            ax.scatter(gg["time_h"], gg["Q_obs"], color="black", s=18)
            ax.set_title(str(run_no), fontsize=9)
        for ax in axes[-1, :]:
            ax.set_xlabel("time_h")
        for ax in axes[:, 0]:
            ax.set_ylabel("Q")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "corrected_bootstrap_test_curve_intervals.png", dpi=300)
        plt.close(fig)

    b2_vals = ok["b2_refit"].dropna().to_numpy(float)
    test_rmse = metrics.loc[metrics["fit_success"], "test_curve_RMSE"].dropna().to_numpy(float)
    b2_p2_5 = float(np.percentile(b2_vals, 2.5))
    b2_p97_5 = float(np.percentile(b2_vals, 97.5))
    original_inside = bool(b2_p2_5 <= B2_ORIGINAL <= b2_p97_5)
    lines = [
        "# Corrected Fixed-Structure Bootstrap Report",
        "",
    ]
    lines.extend(
        md_table(
            [
                ("bootstrap_success_rate", float(constants["fit_success"].mean())),
                ("b2_original", B2_ORIGINAL),
                ("b2_mean", float(np.mean(b2_vals))),
                ("b2_sd", float(np.std(b2_vals, ddof=1))),
                ("b2_median", float(np.median(b2_vals))),
                ("b2_p2_5", b2_p2_5),
                ("b2_p97_5", b2_p97_5),
                ("whether_original_b2_is_inside_95_interval", original_inside),
                ("test_curve_RMSE_mean", float(np.mean(test_rmse))),
                ("test_curve_RMSE_sd", float(np.std(test_rmse, ddof=1))),
                ("test_curve_RMSE_p2_5", float(np.percentile(test_rmse, 2.5))),
                ("test_curve_RMSE_p97_5", float(np.percentile(test_rmse, 97.5))),
                ("runtime_seconds", time.time() - start),
            ]
        )
    )
    lines.extend(
        [
            "",
            f"The original final-equation value `b2_original = {B2_ORIGINAL}` lies within the bootstrap 95% interval: `{original_inside}`.",
        ]
    )
    (REPORT_DIR / "02_corrected_fixed_structure_bootstrap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] corrected fixed-structure bootstrap complete")


if __name__ == "__main__":
    main()

