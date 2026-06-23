from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _vanilla_common import (
    FIGURES_DIR,
    FORMULATION_VARS,
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    VanillaConfig,
    compile_expr,
    ensure_dirs,
    get_successful_seed_dirs,
    md_table,
    qcap_norm_from_observed,
    simplify_expression,
    simulate_record,
    variables_in_text,
)

MODEL_LABEL = "Vanilla ODE-SR"
GRID_N = 21


def q6_grid_fast(f, n: int, qcap_norm: float) -> pd.DataFrame:
    cfg = VanillaConfig(mode="grid", seeds=[], pop_size=0, ngen=0, hall_of_fame_size=0)
    rows = []
    grid_id = 0
    for c1 in np.linspace(20.0, 30.0, n):
        c1n = (c1 - 20.0) / 10.0
        for c2 in np.linspace(10.0, 20.0, n):
            c2n = (c2 - 10.0) / 10.0
            for c3 in np.linspace(10.0, 20.0, n):
                c3n = (c3 - 10.0) / 10.0
                rec = {
                    "time_h": np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float),
                    "Qtilde_obs": np.zeros(8, dtype=float),
                    "C1n": c1n,
                    "C2n": c2n,
                    "C3n": c3n,
                }
                pred = simulate_record(f, rec, cfg, qcap_norm=qcap_norm)
                rows.append({"grid_id": grid_id, "C1": c1, "C2": c2, "C3": c3, "Q6": float(pred[-1] * Q_SCALE) if np.all(np.isfinite(pred)) else np.nan})
                grid_id += 1
    return pd.DataFrame(rows)


def sensitivity_metrics(q6_grid: pd.DataFrame) -> dict:
    q6 = q6_grid["Q6"].to_numpy(float)
    finite = q6[np.isfinite(q6)]
    if finite.size == 0:
        return {"Q6_range_design_grid": np.nan, "Q6_std_design_grid": np.nan, "active_sensitivity_count": 0, "active_sensitivity_C1": False, "active_sensitivity_C2": False, "active_sensitivity_C3": False, "structural_class": "degenerate_for_optimisation", "nontrivial_q6_variation": False}
    q6_range = float(np.nanmax(finite) - np.nanmin(finite))
    q6_std = float(np.nanstd(finite))
    tol = 1e-6 * max(1.0, float(np.nanmedian(np.abs(finite))))
    ordered = q6_grid.sort_values(["C1", "C2", "C3"])
    arr = ordered["Q6"].to_numpy(float).reshape((GRID_N, GRID_N, GRID_N))
    grads = np.gradient(arr, np.linspace(20.0, 30.0, GRID_N), np.linspace(10.0, 20.0, GRID_N), np.linspace(10.0, 20.0, GRID_N), edge_order=1)
    mean_abs = [float(np.nanmean(np.abs(g))) for g in grads]
    max_abs = [float(np.nanmax(np.abs(g))) for g in grads]
    threshold = 1e-3 * max(1.0, q6_range)
    active = [bool(v > threshold) if np.isfinite(v) else False for v in mean_abs]
    active_count = int(sum(active))
    nontrivial = bool(q6_range > tol)
    if active_count >= 2 and nontrivial:
        structural_class = "structurally_valid"
    elif active_count == 1:
        structural_class = "structurally_weak"
    else:
        structural_class = "degenerate_for_optimisation"
    return {
        "Q6_min_design_grid": float(np.nanmin(finite)),
        "Q6_max_design_grid": float(np.nanmax(finite)),
        "Q6_range_design_grid": q6_range,
        "Q6_std_design_grid": q6_std,
        "nontrivial_q6_variation": nontrivial,
        "q6_nontrivial_tolerance": tol,
        "mean_abs_dQ6_dC1": mean_abs[0],
        "mean_abs_dQ6_dC2": mean_abs[1],
        "mean_abs_dQ6_dC3": mean_abs[2],
        "max_abs_dQ6_dC1": max_abs[0],
        "max_abs_dQ6_dC2": max_abs[1],
        "max_abs_dQ6_dC3": max_abs[2],
        "active_sensitivity_C1": active[0],
        "active_sensitivity_C2": active[1],
        "active_sensitivity_C3": active[2],
        "active_sensitivity_count": active_count,
        "structural_class": structural_class,
    }


def main() -> None:
    ensure_dirs()
    qcap_norm, _ = qcap_norm_from_observed()
    rows = []
    for seed_dir in get_successful_seed_dirs():
        seed = int(seed_dir.name.split("_")[-1])
        expr = (seed_dir / "best_expression_infix.txt").read_text(encoding="utf-8").strip()
        sympy_text = (seed_dir / "best_expression_sympy.txt").read_text(encoding="utf-8").strip()
        before = variables_in_text(expr)
        simplified, after = simplify_expression(sympy_text if sympy_text else expr)
        formulation_after = [v for v in FORMULATION_VARS if v in after]
        f, _, _ = compile_expr(expr)
        sens = sensitivity_metrics(q6_grid_fast(f, GRID_N, qcap_norm))
        valid = bool("Q" in after and len(formulation_after) >= 2 and sens["active_sensitivity_count"] >= 2 and sens["nontrivial_q6_variation"])
        rows.append(
            {
                "model": MODEL_LABEL,
                "seed": seed,
                "expression": expr,
                "simplified_expression": simplified,
                "variables_present_before_simplification": ",".join(before),
                "variables_present_after_simplification": ",".join(after),
                "Q_present": "Q" in after,
                "C1_present": "C1" in after,
                "C2_present": "C2" in after,
                "C3_present": "C3" in after,
                "formulation_variable_count": len(formulation_after),
                **sens,
                "valid_for_formulation_optimisation": valid,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS_DIR / "vanilla_odesr_structural_validity.csv", index=False)

    plot = audit.copy()
    plot["label"] = plot["seed"].map(lambda s: f"seed {s}")
    plt.figure(figsize=(7.2, 4.5))
    plt.bar(plot["label"], plot["Q6_range_design_grid"], color="#4c78a8")
    plt.ylabel("Q6 range over design grid")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vanilla_odesr_q6_range_barplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 4.5))
    plt.bar(plot["label"], plot["active_sensitivity_count"], color="#f58518")
    plt.ylabel("Active formulation sensitivities")
    plt.ylim(0, 3.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vanilla_odesr_active_sensitivity_barplot.png", dpi=300)
    plt.close()

    lines = [
        "# Vanilla ODE-SR Structural Validity Audit",
        "",
        f"- q_scale used for replay: `{Q_SCALE}`.",
        f"- Design grid: `{GRID_N} x {GRID_N} x {GRID_N}` over C1, C2, C3.",
        "- Validity rule: Q present after simplification, at least two formulation variables present, at least two active formulation sensitivities, and non-trivial Q6 variation.",
        "",
        "## Structural Metrics",
        "",
        *md_table(audit, ["seed", "variables_present_after_simplification", "formulation_variable_count", "active_sensitivity_count", "Q6_range_design_grid", "Q6_std_design_grid", "structural_class", "valid_for_formulation_optimisation"]),
        "",
        "## Outputs",
        "",
        "- `results/vanilla_odesr_structural_validity.csv`",
        "- `figures/vanilla_odesr_q6_range_barplot.png`",
        "- `figures/vanilla_odesr_active_sensitivity_barplot.png`",
    ]
    (REPORT_DIR / "03_vanilla_odesr_structural_validity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] results/vanilla_odesr_structural_validity.csv")


if __name__ == "__main__":
    main()
