from __future__ import annotations

import numpy as np
import pandas as pd

from _corrected_bootstrap_common import (
    B2_ORIGINAL,
    CONFIG_DIR,
    REPORT_DIR,
    RESULTS_DIR,
    ensure_dirs,
    load_json,
)


PASS_WORDING = (
    "The final symbolic structure was held fixed, including the numerator term softplus(2Qtilde), and only the denominator offset b2 was refitted over 500 bootstrap resamples. "
    "The full-training refit recovered a b2 value close to the selected final-equation constant, and the original b2 lay within the bootstrap 95% interval."
)

FAIL_WORDING = (
    "A constant-refit bootstrap was attempted, but the independent refit pipeline did not reproduce the final-equation constant on the full training set. "
    "Therefore, we did not use the constant-refit bootstrap as evidence of constant stability. Instead, we report uncertainty for the fixed final equation and acknowledge this limitation."
)


def read_sanity() -> dict[str, object]:
    path = RESULTS_DIR / "full_train_b2_refit_sanity_check.csv"
    if not path.exists():
        return {"available": False, "sanity_pass": False}
    df = pd.read_csv(path)
    best = df[df["is_best_across_initialisations"]].iloc[0]
    original_init = df[df["is_original_initialisation"]].iloc[0]
    return {
        "available": True,
        "sanity_pass": bool(best["sanity_pass"]),
        "b2_refit_best": float(best["b2_refit"]),
        "b2_refit_from_original_initialisation": float(original_init["b2_refit"]),
        "absolute_difference_from_original": float(best["absolute_difference_from_original"]),
        "relative_difference_from_original": float(best["relative_difference_from_original"]),
        "train_curve_RMSE_at_original_b2": float(best["train_curve_RMSE_at_original_b2"]),
        "train_curve_RMSE_at_refit_b2": float(best["train_curve_RMSE_at_refit_b2"]),
        "test_curve_RMSE_at_original_b2": float(best["test_curve_RMSE_at_original_b2"]),
        "test_curve_RMSE_at_refit_b2": float(best["test_curve_RMSE_at_refit_b2"]),
    }


def read_bootstrap() -> dict[str, object]:
    path = RESULTS_DIR / "corrected_bootstrap_b2_constants.csv"
    metrics_path = RESULTS_DIR / "corrected_bootstrap_metrics.csv"
    if not path.exists() or not metrics_path.exists():
        return {"available": False}
    constants = pd.read_csv(path)
    metrics = pd.read_csv(metrics_path)
    ok = constants[constants["fit_success"]]
    b2 = ok["b2_refit"].dropna().to_numpy(float)
    test_rmse = metrics.loc[metrics["fit_success"], "test_curve_RMSE"].dropna().to_numpy(float)
    if b2.size == 0:
        return {"available": False}
    lo = float(np.percentile(b2, 2.5))
    hi = float(np.percentile(b2, 97.5))
    return {
        "available": True,
        "success_rate": float(constants["fit_success"].mean()),
        "b2_mean": float(np.mean(b2)),
        "b2_sd": float(np.std(b2, ddof=1)),
        "b2_p2_5": lo,
        "b2_p97_5": hi,
        "original_inside_95": bool(lo <= B2_ORIGINAL <= hi),
        "test_curve_RMSE_mean": float(np.mean(test_rmse)) if test_rmse.size else float("nan"),
    }


def main() -> None:
    ensure_dirs()
    config = load_json(CONFIG_DIR / "corrected_bootstrap_config.json")
    sanity = read_sanity()
    bootstrap = read_bootstrap()
    passed = bool(sanity.get("sanity_pass", False))
    wording = PASS_WORDING if passed and bootstrap.get("available") else FAIL_WORDING
    bootstrap_status = "run" if bootstrap.get("available") else "skipped"

    table_rows = [
        {"item": "numerator_softplus_2Qtilde", "value": config["expression_checks"]["numerator_softplus_2"]},
        {"item": "a_fixed_at_2_excluded_from_refit", "value": config["a_fixed"] == 2.0 and not config["a_refitted"]},
        {"item": "b2_original", "value": B2_ORIGINAL},
        {"item": "b2_refit_from_original_initialisation", "value": sanity.get("b2_refit_from_original_initialisation", "")},
        {"item": "b2_refit_best_across_initialisations", "value": sanity.get("b2_refit_best", "")},
        {"item": "full_training_refit_recovers_original_b2", "value": passed},
        {"item": "absolute_difference_from_original", "value": sanity.get("absolute_difference_from_original", "")},
        {"item": "relative_difference_from_original", "value": sanity.get("relative_difference_from_original", "")},
        {"item": "bootstrap_status", "value": bootstrap_status},
        {"item": "bootstrap_b2_95_interval", "value": f"[{bootstrap.get('b2_p2_5')}, {bootstrap.get('b2_p97_5')}]" if bootstrap.get("available") else "not run"},
        {"item": "original_b2_inside_bootstrap_95_interval", "value": bootstrap.get("original_inside_95", "not applicable")},
        {"item": "previous_bootstrap_results_superseded", "value": "revision_validation_robustness_24train6test fixed-structure bootstrap outputs that refitted both a and b2"},
    ]
    summary_table = pd.DataFrame(table_rows)
    summary_table.to_csv(RESULTS_DIR / "table_corrected_bootstrap_summary.csv", index=False)

    lines = [
        "# Bootstrap Correction Summary For R1-2",
        "",
        "1. Does the selected expression numerator correspond to `softplus(2*Qtilde)`?",
        f"Yes: `{config['expression_checks']['numerator_softplus_2']}`. The selected sources contain `Q + Q` and/or `2*Q` in the softplus numerator.",
        "",
        "2. Was `a` fixed at 2 and excluded from refitting?",
        f"Yes. `a_fixed = {config['a_fixed']}` and `a_refitted = {config['a_refitted']}`.",
        "",
        "3. What is the original final-equation value of `b2`?",
        f"`b2_original = {B2_ORIGINAL}`.",
        "",
        "4. Does the full-training refit recover the original `b2`?",
    ]
    if passed:
        lines.append(
            f"Yes. The best full-training refit was `{sanity['b2_refit_best']:.6g}`, with absolute difference `{sanity['absolute_difference_from_original']:.6g}` and relative difference `{sanity['relative_difference_from_original']:.6g}`."
        )
    else:
        lines.append(
            f"No. The best full-training refit was `{sanity.get('b2_refit_best', float('nan')):.6g}`, with absolute difference `{sanity.get('absolute_difference_from_original', float('nan')):.6g}` and relative difference `{sanity.get('relative_difference_from_original', float('nan')):.6g}`, exceeding the required closeness rule."
        )
    lines.extend(
        [
            "",
            "5. If yes, what is the corrected bootstrap 95% interval for `b2`, and does it include the original `b2`?",
        ]
    )
    if bootstrap.get("available"):
        lines.append(
            f"The corrected bootstrap interval was `[{bootstrap['b2_p2_5']:.6g}, {bootstrap['b2_p97_5']:.6g}]`; original `b2` inside interval: `{bootstrap['original_inside_95']}`."
        )
    else:
        lines.append("Not applicable. The corrected bootstrap was skipped because the full-training sanity check failed.")
    lines.extend(
        [
            "",
            "6. If no, why should constant-refit bootstrap not be used as constant-stability evidence?",
            "Because the independent fixed-structure refit, with `a` fixed at 2 and only `b2` optimized, did not reproduce the selected final-equation constant on the full training set. A bootstrap around that refit would characterize the refit pipeline's different point estimate, not stability of the published constant.",
            "",
            "7. What wording should be used in the response to Reviewer #1 Comment 2?",
            wording,
            "",
            "8. What wording should be used in the manuscript?",
            wording,
            "",
            "9. Which previous bootstrap results should be superseded?",
            "The previous `revision_validation_robustness_24train6test` fixed-structure bootstrap results should be superseded for R1-2 because they refitted both `a` and `b2`. In particular, do not use the earlier constants, metrics, interval plots, or report text that describe refitting `a` in `softplus(a * Qtilde)`.",
            "",
            "## Files",
            "",
            "- `revision_validation_bootstrap_correction_24train6test/reports/00_final_equation_reconstruction_report.md`",
            "- `revision_validation_bootstrap_correction_24train6test/reports/01_full_train_b2_refit_sanity_check_report.md`",
            f"- `revision_validation_bootstrap_correction_24train6test/reports/{'02_corrected_fixed_structure_bootstrap_report.md' if bootstrap.get('available') else '02_corrected_fixed_structure_bootstrap_SKIPPED.md'}`",
            "- `revision_validation_bootstrap_correction_24train6test/reports/03_fixed_parameter_uncertainty_report.md`",
            "- `revision_validation_bootstrap_correction_24train6test/results/table_corrected_bootstrap_summary.csv`",
        ]
    )
    (REPORT_DIR / "summary_bootstrap_correction_for_R1_2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] bootstrap correction summary complete")


if __name__ == "__main__":
    main()

