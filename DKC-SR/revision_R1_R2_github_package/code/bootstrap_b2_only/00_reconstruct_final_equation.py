from __future__ import annotations

from _corrected_bootstrap_common import (
    A_FIXED,
    B2_BOUNDS,
    B2_ORIGINAL,
    CONFIG_DIR,
    EXCLUDED_ROWS,
    INITIAL_VALUES,
    N_BOOTSTRAP,
    Q_SCALE,
    RANDOM_SEED,
    REPORT_DIR,
    SOURCE_DATA,
    ensure_dirs,
    expression_checks,
    load_canonical,
    md_table,
    read_expression_sources,
    rel,
    write_json,
)


def main() -> None:
    ensure_dirs()
    expressions = read_expression_sources()
    checks = expression_checks(expressions)
    canonical = load_canonical()
    data_files = {
        "curve_train": SOURCE_DATA / "canonical_curve_train_24.csv",
        "curve_test": SOURCE_DATA / "canonical_curve_test_6.csv",
        "endpoint_train": SOURCE_DATA / "canonical_endpoint_train_24.csv",
        "endpoint_test": SOURCE_DATA / "canonical_endpoint_test_6.csv",
    }
    present_runs = sorted(
        set(canonical["curve_train"]["run_no"].astype(str))
        | set(canonical["curve_test"]["run_no"].astype(str))
    )
    excluded_present = sorted(set(EXCLUDED_ROWS) & set(present_runs))

    config = {
        "q_scale": Q_SCALE,
        "a_fixed": A_FIXED,
        "a_refitted": False,
        "b2_original": B2_ORIGINAL,
        "b2_refit_bounds": list(B2_BOUNDS),
        "initial_values_for_full_train_sanity": INITIAL_VALUES,
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "data_files": {key: rel(path) for key, path in data_files.items()},
        "excluded_rows_kept_excluded": EXCLUDED_ROWS,
        "excluded_rows_present_in_canonical_train_or_test": excluded_present,
        "expression_sources": expressions,
        "expression_checks": checks,
        "replay_implementation_source": "revision_validation_robustness_24train6test/_robustness_common.py wrapped by revision_validation_bootstrap_correction_24train6test/_corrected_bootstrap_common.py",
        "replay_equation": "dQtilde/dt = softplus(2*Qtilde) / ((Qtilde**2 - C1n/C3n + C2n)**2 + b2)",
        "replay_differences_from_previous_bootstrap": [
            "a is fixed at 2.0 and never optimized.",
            "Only b2 is eligible for refitting.",
            "No symbolic structure search or DKC-SR retraining is performed.",
        ],
    }
    write_json(CONFIG_DIR / "corrected_bootstrap_config.json", config)

    lines: list[str] = [
        "# Final Equation Reconstruction Report",
        "",
        "## Reconstruction Result",
        "",
    ]
    lines.extend(
        md_table(
            [
                ("selected numerator is softplus(2*Qtilde)", checks["numerator_softplus_2"]),
                ("a fixed and not refitted", True),
                ("original b2", "2.3550290604627118"),
                ("q_scale", "3008.198194823261"),
                ("b2 confirmed in selected sources", checks["b2_original_confirmed"]),
                ("excluded rows present in canonical train/test", excluded_present),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Selected Expression Sources",
            "",
        ]
    )
    for source, expr in expressions.items():
        lines.append(f"- `{source}`: `{expr}`")
    lines.extend(
        [
            "",
            "## Data Files Used",
            "",
        ]
    )
    for key, path in data_files.items():
        lines.append(f"- {key}: `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Replay Implementation",
            "",
            "- Source: `revision_validation_robustness_24train6test/_robustness_common.py` for the normalized RK4 replay and metric conventions, wrapped locally in `revision_validation_bootstrap_correction_24train6test/_corrected_bootstrap_common.py`.",
            "- Corrected replay function: `simulate_corrected_record(rec, b2)` calls the existing replay with `a=2.0` and variable `b2`.",
            "- Normalisation: `Q_raw = Q_SCALE * Qtilde`, with `Q_SCALE = 3008.198194823261`, and normalized `C1n`, `C2n`, `C3n` from the canonical CSVs.",
            "- Integration convention: RK4 stepping between measured time points using the existing revision-validation helper.",
            "",
            "## Differences From The Previous Fixed-Structure Bootstrap",
            "",
            "- The previous bootstrap refitted both `a` and `b2`.",
            "- This corrected workflow fixes `a = 2.0`, because the selected numerator is `softplus(Q + Q)`, equivalently `softplus(2*Qtilde)`.",
            "- Only the denominator offset `b2` may be refitted in the sanity check or conditional bootstrap.",
            "",
            "## Outputs",
            "",
            "- `revision_validation_bootstrap_correction_24train6test/config/corrected_bootstrap_config.json`",
        ]
    )
    (REPORT_DIR / "00_final_equation_reconstruction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] final equation reconstruction complete")


if __name__ == "__main__":
    main()
