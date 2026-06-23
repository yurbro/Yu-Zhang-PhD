from __future__ import annotations

import numpy as np
import pandas as pd

from _corrected_bootstrap_common import (
    A_FIXED,
    B2_ORIGINAL,
    INITIAL_VALUES,
    REPORT_DIR,
    RESULTS_DIR,
    curve_metric_dict,
    ensure_dirs,
    fit_b2_from_initial,
    load_canonical,
    md_table,
    records_from_curve,
    sanity_pass,
)


def main() -> None:
    ensure_dirs()
    canonical = load_canonical()
    train_records = records_from_curve(canonical["curve_train"])
    test_records = records_from_curve(canonical["curve_test"])

    rows = [fit_b2_from_initial(train_records, initial) for initial in INITIAL_VALUES]
    results = pd.DataFrame(rows)
    results["a_fixed"] = A_FIXED
    results["b2_original"] = B2_ORIGINAL
    results["absolute_difference_from_original"] = (results["b2_refit"] - B2_ORIGINAL).abs()
    results["relative_difference_from_original"] = results["absolute_difference_from_original"] / B2_ORIGINAL
    results["is_original_initialisation"] = np.isclose(results["initial_value"], B2_ORIGINAL)

    ok = results[results["fit_success"] & np.isfinite(results["objective_normalized_mse"])].copy()
    if ok.empty:
        best_idx = results.index[0]
    else:
        best_idx = ok["objective_normalized_mse"].idxmin()
    results["is_best_across_initialisations"] = results.index == best_idx
    best = results.loc[best_idx]
    original_init = results.loc[results["is_original_initialisation"]].iloc[0]

    train_original, _ = curve_metric_dict(train_records, b2=B2_ORIGINAL, dataset="train")
    test_original, _ = curve_metric_dict(test_records, b2=B2_ORIGINAL, dataset="test")
    train_refit, _ = curve_metric_dict(train_records, b2=float(best["b2_refit"]), dataset="train")
    test_refit, _ = curve_metric_dict(test_records, b2=float(best["b2_refit"]), dataset="test")

    abs_diff = float(best["absolute_difference_from_original"])
    rel_diff = float(best["relative_difference_from_original"])
    passed = sanity_pass(abs_diff, rel_diff)
    results["sanity_pass"] = passed
    results["train_curve_RMSE_at_original_b2"] = train_original["RMSE"]
    results["train_curve_RMSE_at_refit_b2"] = train_refit["RMSE"]
    results["test_curve_RMSE_at_original_b2"] = test_original["RMSE"]
    results["test_curve_RMSE_at_refit_b2"] = test_refit["RMSE"]
    results.to_csv(RESULTS_DIR / "full_train_b2_refit_sanity_check.csv", index=False)

    decision = "PASS" if passed else "FAIL"
    lines: list[str] = [
        "# Full-Training b2 Refit Sanity Check",
        "",
        f"Decision: `{decision}`.",
        "",
    ]
    lines.extend(
        md_table(
            [
                ("b2_original", "2.3550290604627118"),
                ("a_fixed", "2.0"),
                ("b2_refit_from_original_initialisation", float(original_init["b2_refit"])),
                ("b2_refit_best_across_initialisations", float(best["b2_refit"])),
                ("absolute_difference_from_original", abs_diff),
                ("relative_difference_from_original", rel_diff),
                ("train_curve_RMSE_at_original_b2", train_original["RMSE"]),
                ("train_curve_RMSE_at_refit_b2", train_refit["RMSE"]),
                ("test_curve_RMSE_at_original_b2", test_original["RMSE"]),
                ("test_curve_RMSE_at_refit_b2", test_refit["RMSE"]),
                ("absolute threshold", "<= 0.15"),
                ("relative threshold", "<= 7.5%"),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Initialisation Sensitivity",
            "",
            "| initial_value | b2_refit | objective_normalized_mse | optimizer_success | is_best |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in results.iterrows():
        lines.append(
            f"| {row['initial_value']:.12g} | {row['b2_refit']:.12g} | {row['objective_normalized_mse']:.12g} | {row['optimizer_success']} | {row['is_best_across_initialisations']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if passed:
        lines.append(
            "The full-training refit recovered a value close to the selected final-equation constant, so the corrected constant-refit bootstrap may proceed."
        )
    else:
        lines.append(
            "The full-training refit did not recover a value close to the selected final-equation constant under the required absolute and relative thresholds. The constant-refit bootstrap should therefore not be presented as evidence of constant stability for the final equation."
        )
    lines.extend(
        [
            "",
            "## Output",
            "",
            "- `revision_validation_bootstrap_correction_24train6test/results/full_train_b2_refit_sanity_check.csv`",
        ]
    )
    (REPORT_DIR / "01_full_train_b2_refit_sanity_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] full-train b2 refit sanity check: {decision}")


if __name__ == "__main__":
    main()
