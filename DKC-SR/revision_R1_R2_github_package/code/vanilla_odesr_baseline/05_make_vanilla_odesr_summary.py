from __future__ import annotations

import numpy as np
import pandas as pd

from _vanilla_common import (
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    SOURCE_DKCSR,
    SOURCE_UCSR,
    SOURCE_UCSR_STRUCT,
    ensure_dirs,
    md_table,
    rel,
)

DKC_METRIC_LABEL = "DKC-SR replayed equation, q_scale=3008.198194823261"
DKC_LABEL = "DKC-SR selected equation"
VANILLA_LABEL = "Vanilla ODE-SR"
UCSR_SUPP_LABEL = "Constraint-relaxed operator-matched SR (supplementary)"


def first_row(df: pd.DataFrame) -> pd.Series | None:
    return df.iloc[0] if len(df) else None


def mean_or_nan(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan


def sd_or_zero(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan


def make_predictive_structural_table() -> pd.DataFrame:
    curve = pd.read_csv(RESULTS_DIR / "vanilla_odesr_curve_metrics_by_seed.csv")
    endpoint = pd.read_csv(RESULTS_DIR / "vanilla_odesr_endpoint_metrics_by_seed.csv")
    structural = pd.read_csv(RESULTS_DIR / "vanilla_odesr_structural_validity.csv")
    v_curve = curve[(curve["dataset"] == "test") & (curve["scope"] == "overall_excluding_t0")].copy()
    v_endpoint = endpoint[endpoint["dataset"] == "test"].copy()
    merged = v_curve.merge(v_endpoint, on=["model", "seed", "dataset"], suffixes=("_curve", "_endpoint")).merge(structural, on=["model", "seed"], how="left")
    best = merged.sort_values("RMSE").iloc[0] if len(merged) else None
    valid = merged[merged["valid_for_formulation_optimisation"] == True].copy()
    best_valid = valid.sort_values("RMSE").iloc[0] if len(valid) else None

    dkc_curve = pd.read_csv(SOURCE_DKCSR / "results" / "unified_curve_metrics_with_dkcsr_24train6test.csv")
    dkc_endpoint = pd.read_csv(SOURCE_DKCSR / "results" / "unified_endpoint_metrics_with_dkcsr_24train6test.csv")
    dkc_c = first_row(dkc_curve[(dkc_curve["model"] == DKC_METRIC_LABEL) & (dkc_curve["dataset"] == "test") & (dkc_curve["scope"] == "overall_excluding_t0")])
    dkc_e = first_row(dkc_endpoint[(dkc_endpoint["model"] == DKC_METRIC_LABEL) & (dkc_endpoint["dataset"] == "test")])
    dkc_struct = first_row(pd.read_csv(SOURCE_UCSR_STRUCT / "results" / "predictive_vs_structural_summary.csv").query("model == 'DKC-SR selected equation'"))

    rows = []
    if dkc_c is not None and dkc_e is not None and dkc_struct is not None:
        rows.append(
            {
                "model_or_group": DKC_LABEL,
                "n_successful_seeds": 1,
                "test_curve_RMSE_mean": dkc_c["RMSE"],
                "test_curve_RMSE_sd": 0.0,
                "test_curve_RMSE_best": dkc_c["RMSE"],
                "test_curve_R2_best": dkc_c["R2"],
                "Q6_RMSE_best": dkc_e["RMSE_Q6"],
                "formulation_variable_count": dkc_struct["formulation_variable_count"],
                "active_sensitivity_count": dkc_struct["active_sensitivity_count"],
                "Q6_range_design_grid": dkc_struct["Q6_range"],
                "structural_class": dkc_struct["structural_class"],
                "valid_for_formulation_optimisation": dkc_struct["valid_for_formulation_optimisation"],
            }
        )
    static_curve = dkc_curve[
        (dkc_curve["dataset"] == "test")
        & (dkc_curve["scope"] == "overall_excluding_t0")
        & (~dkc_curve["model"].astype(str).str.contains("DKC-SR", regex=False))
    ].copy()
    if len(static_curve):
        static_best = static_curve.sort_values("RMSE").iloc[0]
        static_endpoint = first_row(dkc_endpoint[(dkc_endpoint["dataset"] == "test") & (dkc_endpoint["model"] == static_best["model"])])
        rows.append(
            {
                "model_or_group": f"Best conventional static/curve baseline ({static_best['model']})",
                "n_successful_seeds": 1,
                "test_curve_RMSE_mean": static_best["RMSE"],
                "test_curve_RMSE_sd": 0.0,
                "test_curve_RMSE_best": static_best["RMSE"],
                "test_curve_R2_best": static_best["R2"],
                "Q6_RMSE_best": static_endpoint["RMSE_Q6"] if static_endpoint is not None else np.nan,
                "formulation_variable_count": "not_applicable",
                "active_sensitivity_count": "not_applicable",
                "Q6_range_design_grid": "not_applicable",
                "structural_class": "static_baseline",
                "valid_for_formulation_optimisation": "not_applicable",
            }
        )
    rows.append(
        {
            "model_or_group": "Vanilla ODE-SR mean +/- SD",
            "n_successful_seeds": int(merged["seed"].nunique()),
            "test_curve_RMSE_mean": mean_or_nan(merged["RMSE"]),
            "test_curve_RMSE_sd": sd_or_zero(merged["RMSE"]),
            "test_curve_RMSE_best": float(merged["RMSE"].min()) if len(merged) else np.nan,
            "test_curve_R2_best": float(merged.loc[merged["RMSE"].idxmin(), "R2"]) if len(merged) else np.nan,
            "Q6_RMSE_best": float(merged["RMSE_Q6"].min()) if len(merged) else np.nan,
            "formulation_variable_count": mean_or_nan(merged["formulation_variable_count"]),
            "active_sensitivity_count": mean_or_nan(merged["active_sensitivity_count"]),
            "Q6_range_design_grid": mean_or_nan(merged["Q6_range_design_grid"]),
            "structural_class": "mixed" if merged["structural_class"].nunique() > 1 else (merged["structural_class"].iloc[0] if len(merged) else ""),
            "valid_for_formulation_optimisation": f"{int(merged['valid_for_formulation_optimisation'].sum())}/{len(merged)}",
        }
    )
    if best is not None:
        rows.append(
            {
                "model_or_group": f"Best vanilla ODE-SR seed {int(best['seed'])} by curve RMSE",
                "n_successful_seeds": 1,
                "test_curve_RMSE_mean": best["RMSE"],
                "test_curve_RMSE_sd": 0.0,
                "test_curve_RMSE_best": best["RMSE"],
                "test_curve_R2_best": best["R2"],
                "Q6_RMSE_best": best["RMSE_Q6"],
                "formulation_variable_count": best["formulation_variable_count"],
                "active_sensitivity_count": best["active_sensitivity_count"],
                "Q6_range_design_grid": best["Q6_range_design_grid"],
                "structural_class": best["structural_class"],
                "valid_for_formulation_optimisation": best["valid_for_formulation_optimisation"],
            }
        )
    if best_valid is not None:
        rows.append(
            {
                "model_or_group": f"Best structurally valid vanilla ODE-SR seed {int(best_valid['seed'])}",
                "n_successful_seeds": 1,
                "test_curve_RMSE_mean": best_valid["RMSE"],
                "test_curve_RMSE_sd": 0.0,
                "test_curve_RMSE_best": best_valid["RMSE"],
                "test_curve_R2_best": best_valid["R2"],
                "Q6_RMSE_best": best_valid["RMSE_Q6"],
                "formulation_variable_count": best_valid["formulation_variable_count"],
                "active_sensitivity_count": best_valid["active_sensitivity_count"],
                "Q6_range_design_grid": best_valid["Q6_range_design_grid"],
                "structural_class": best_valid["structural_class"],
                "valid_for_formulation_optimisation": best_valid["valid_for_formulation_optimisation"],
            }
        )
    ucsr = pd.read_csv(SOURCE_UCSR_STRUCT / "results" / "predictive_vs_structural_summary.csv")
    ucsr_best = ucsr[ucsr["model"] == "Unconstrained SR"].sort_values("test_curve_RMSE").iloc[0]
    rows.append(
        {
            "model_or_group": f"{UCSR_SUPP_LABEL} best seed {int(ucsr_best['seed'])}",
            "n_successful_seeds": 1,
            "test_curve_RMSE_mean": ucsr_best["test_curve_RMSE"],
            "test_curve_RMSE_sd": 0.0,
            "test_curve_RMSE_best": ucsr_best["test_curve_RMSE"],
            "test_curve_R2_best": ucsr_best["test_curve_R2"],
            "Q6_RMSE_best": ucsr_best["Q6_RMSE"],
            "formulation_variable_count": ucsr_best["formulation_variable_count"],
            "active_sensitivity_count": ucsr_best["active_sensitivity_count"],
            "Q6_range_design_grid": ucsr_best["Q6_range"],
            "structural_class": ucsr_best["structural_class"],
            "valid_for_formulation_optimisation": ucsr_best["valid_for_formulation_optimisation"],
        }
    )
    return pd.DataFrame(rows)


def make_physical_table() -> pd.DataFrame:
    audit = pd.read_csv(RESULTS_DIR / "vanilla_vs_dkcsr_physical_optimisation_audit.csv")
    curve = pd.read_csv(RESULTS_DIR / "vanilla_odesr_curve_metrics_by_seed.csv")
    v_curve = curve[(curve["dataset"] == "test") & (curve["scope"] == "overall_excluding_t0")]
    best_seed = int(v_curve.sort_values("RMSE").iloc[0]["seed"]) if len(v_curve) else None
    rows = []
    dkc = first_row(audit[audit["model"] == DKC_LABEL])
    if dkc is not None:
        rows.append(row_from_physical(DKC_LABEL, dkc))
    vanilla = audit[audit["model"] == VANILLA_LABEL].copy()
    if len(vanilla):
        rows.append(
            {
                "model_or_group": "Vanilla ODE-SR mean +/- SD",
                "negative_Q_prediction_rate": mean_or_nan(vanilla["negative_Q_prediction_rate"]),
                "non_monotonic_curve_rate": mean_or_nan(vanilla["non_monotonic_curve_rate"]),
                "negative_RHS_rate": mean_or_nan(vanilla["negative_RHS_rate"]),
                "positive_dfdQ_rate": mean_or_nan(vanilla["positive_dfdQ_rate"]),
                "numerical_failure_rate": mean_or_nan(vanilla["numerical_failure_rate"]),
                "extreme_Q6_rate": mean_or_nan(vanilla["extreme_Q6_rate"]),
                "best_Q6": mean_or_nan(vanilla["best_Q6"]),
                "best_is_boundary": f"{int(vanilla['best_is_boundary'].astype(bool).sum())}/{len(vanilla)}",
                "optimisation_failure_rate": mean_or_nan(vanilla["failure_rate"]),
            }
        )
    if best_seed is not None:
        best = first_row(vanilla[vanilla["seed"].astype(int) == best_seed])
        if best is not None:
            rows.append(row_from_physical(f"Best vanilla ODE-SR seed {best_seed}", best))

    u_phys = pd.read_csv(SOURCE_UCSR / "results" / "physical_plausibility_unconstrained_vs_dkcsr.csv")
    u_opt = pd.read_csv(SOURCE_UCSR / "results" / "optimisation_safety_unconstrained_vs_dkcsr.csv")
    u_summary = pd.read_csv(SOURCE_UCSR_STRUCT / "results" / "predictive_vs_structural_summary.csv")
    u_best_seed = int(u_summary[u_summary["model"] == "Unconstrained SR"].sort_values("test_curve_RMSE").iloc[0]["seed"])
    u_phys = u_phys.copy()
    u_opt = u_opt.copy()
    u_phys["_seed_numeric"] = pd.to_numeric(u_phys["seed"], errors="coerce")
    u_opt["_seed_numeric"] = pd.to_numeric(u_opt["seed"], errors="coerce")
    u_row = first_row(u_phys[(u_phys["model"] == "Unconstrained SR") & (u_phys["_seed_numeric"] == u_best_seed)])
    u_opt_row = first_row(u_opt[(u_opt["model"] == "Unconstrained SR") & (u_opt["_seed_numeric"] == u_best_seed)])
    if u_row is not None and u_opt_row is not None:
        rows.append(
            {
                "model_or_group": f"{UCSR_SUPP_LABEL} best seed {u_best_seed}",
                "negative_Q_prediction_rate": u_row["negative_Q_prediction_rate"],
                "non_monotonic_curve_rate": u_row["non_monotonic_curve_rate"],
                "negative_RHS_rate": u_row["negative_RHS_rate"],
                "positive_dfdQ_rate": np.nan,
                "numerical_failure_rate": u_row["numerical_failure_rate"],
                "extreme_Q6_rate": u_row["extreme_Q6_rate"],
                "best_Q6": u_opt_row["best_Q6"],
                "best_is_boundary": u_opt_row["best_is_boundary"],
                "optimisation_failure_rate": u_opt_row["failure_rate"],
            }
        )
    return pd.DataFrame(rows)


def row_from_physical(label: str, row: pd.Series) -> dict:
    return {
        "model_or_group": label,
        "negative_Q_prediction_rate": row["negative_Q_prediction_rate"],
        "non_monotonic_curve_rate": row["non_monotonic_curve_rate"],
        "negative_RHS_rate": row["negative_RHS_rate"],
        "positive_dfdQ_rate": row["positive_dfdQ_rate"],
        "numerical_failure_rate": row["numerical_failure_rate"],
        "extreme_Q6_rate": row["extreme_Q6_rate"],
        "best_Q6": row["best_Q6"],
        "best_is_boundary": row["best_is_boundary"],
        "optimisation_failure_rate": row["failure_rate"],
    }


def main() -> None:
    ensure_dirs()
    pred_struct = make_predictive_structural_table()
    phys = make_physical_table()
    pred_path = RESULTS_DIR / "table_vanilla_odesr_predictive_structural_comparison.csv"
    phys_path = RESULTS_DIR / "table_vanilla_odesr_physical_optimisation_comparison.csv"
    pred_struct.to_csv(pred_path, index=False)
    phys.to_csv(phys_path, index=False)

    vanilla_rows = pred_struct[pred_struct["model_or_group"].str.contains("Vanilla", na=False)]
    best_vanilla = pred_struct[pred_struct["model_or_group"].str.startswith("Best vanilla")]
    physical_vanilla = phys[phys["model_or_group"].str.startswith("Best vanilla")]
    best_rmse = float(best_vanilla["test_curve_RMSE_best"].iloc[0]) if len(best_vanilla) else np.nan
    best_struct = str(best_vanilla["structural_class"].iloc[0]) if len(best_vanilla) else "not available"
    best_boundary = str(physical_vanilla["best_is_boundary"].iloc[0]) if len(physical_vanilla) else "not available"
    best_positive_dfdq = float(physical_vanilla["positive_dfdQ_rate"].iloc[0]) if len(physical_vanilla) else np.nan

    lines = [
        "# Summary: Vanilla ODE-SR Baseline, 24 Train + 6 Test",
        "",
        f"- q_scale used: `{Q_SCALE}`.",
        "- Vanilla ODE-SR used conventional primitives only: add, sub, mul, protected div, pow2.",
        "- `softplus` and DKC-SR physical/domain constraints were absent.",
        "- The best conventional static/curve baseline from the existing 24+6 validation is included for predictive context; structural ODE validity metrics are not applicable to that row.",
        "- The previous constraint-relaxed operator-matched SR is reported only as a supplementary ablation.",
        "",
        "## Table 1. Predictive and Structural Comparison",
        "",
        *md_table(pred_struct, list(pred_struct.columns)),
        "",
        "## Table 2. Physical and Optimisation Safety",
        "",
        *md_table(phys, list(phys.columns)),
        "",
        "## Reviewer-Facing Answers",
        "",
        "1. Vanilla ODE-SR is a cleaner reviewer-facing conventional SR baseline than the previous operator-matched ablation because it removes the positivity-oriented `softplus` primitive.",
        f"2. Best vanilla ODE-SR test curve RMSE was `{best_rmse:.3f}`.",
        f"3. The best vanilla ODE-SR structural class was `{best_struct}`.",
        "4. Basic physical checks are reported in Table 2; violations should be interpreted by the rates, not by RMSE alone.",
        f"5. The best vanilla ODE-SR positive df/dQ rate was `{best_positive_dfdq:.6g}`; under this finite-difference audit it does not violate df/dQ <= 0, but this is because the fitted RHS is formulation-independent and constant.",
        f"6. The best vanilla ODE-SR grid optimum boundary status was `{best_boundary}`.",
        "7. The previous constraint-relaxed SR should be described as an operator-matched supplementary ablation, not as the main vanilla unconstrained SR baseline.",
        "8. Suggested wording: A conventional vanilla ODE-SR baseline using generic arithmetic primitives was added on the same 24/6 split. This baseline evaluates whether unconstrained ODE symbolic regression can match predictive accuracy while retaining structurally and physically usable formulation dependence. The operator-matched softplus ablation is retained only as supplementary evidence because its operator set still contains positivity-oriented design choices.",
        "",
        "## Outputs",
        "",
        f"- `{rel(pred_path)}`",
        f"- `{rel(phys_path)}`",
    ]
    (REPORT_DIR / "summary_vanilla_odesr_baseline_24train6test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {rel(REPORT_DIR / 'summary_vanilla_odesr_baseline_24train6test.md')}")


if __name__ == "__main__":
    main()
