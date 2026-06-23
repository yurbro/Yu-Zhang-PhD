from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _vanilla_common import (
    FIGURES_DIR,
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    compile_expr,
    dkc_callable,
    ensure_dirs,
    get_successful_seed_dirs,
    grid_predictions_for_model,
    make_grid,
    md_table,
    optimisation_rows_from_grid,
    qcap_norm_from_observed,
    rel,
    safety_rows_from_grid,
)

DKC_LABEL = "DKC-SR selected equation"
VANILLA_LABEL = "Vanilla ODE-SR"


def main() -> None:
    ensure_dirs()
    qcap_norm, observed_q6_upper = qcap_norm_from_observed()
    f_lookup = {}

    physical_grid = make_grid(5)
    physical_preds = []
    fd = dkc_callable()
    physical_preds.append(grid_predictions_for_model(DKC_LABEL, "selected", fd, physical_grid, qcap_norm))
    f_lookup[(DKC_LABEL, "selected")] = fd

    for seed_dir in get_successful_seed_dirs():
        seed = int(seed_dir.name.split("_")[-1])
        expr = (seed_dir / "best_expression_infix.txt").read_text(encoding="utf-8").strip()
        f, _, _ = compile_expr(expr)
        physical_preds.append(grid_predictions_for_model(VANILLA_LABEL, seed, f, physical_grid, qcap_norm))
        f_lookup[(VANILLA_LABEL, seed)] = f

    physical_grid_pred = pd.concat(physical_preds, ignore_index=True)
    physical = safety_rows_from_grid(physical_grid_pred, observed_q6_upper, f_lookup)

    opt_grid = make_grid(21)
    opt_preds = [grid_predictions_for_model(DKC_LABEL, "selected", fd, opt_grid, qcap_norm)]
    for seed_dir in get_successful_seed_dirs():
        seed = int(seed_dir.name.split("_")[-1])
        expr = (seed_dir / "best_expression_infix.txt").read_text(encoding="utf-8").strip()
        f, _, _ = compile_expr(expr)
        opt_preds.append(grid_predictions_for_model(VANILLA_LABEL, seed, f, opt_grid, qcap_norm))
    opt_grid_pred = pd.concat(opt_preds, ignore_index=True)
    opt = optimisation_rows_from_grid(opt_grid_pred, observed_q6_upper)

    audit = physical.merge(opt, on=["model", "seed"], how="outer")
    audit.to_csv(RESULTS_DIR / "vanilla_vs_dkcsr_physical_optimisation_audit.csv", index=False)

    rates = ["negative_Q_prediction_rate", "non_monotonic_curve_rate", "negative_RHS_rate", "positive_dfdQ_rate", "numerical_failure_rate", "extreme_Q6_rate"]
    plot = audit.copy()
    plot["label"] = plot.apply(lambda r: "DKC-SR" if r["model"] == DKC_LABEL else f"Vanilla {r['seed']}", axis=1)
    ax = plot.set_index("label")[rates].fillna(0.0).plot(kind="bar", figsize=(11, 5.8), width=0.78)
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.legend(fontsize=7)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vanilla_vs_dkcsr_physical_rates_barplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8.2, 4.8))
    ax = plt.gca()
    ax.bar(plot["label"], plot["best_Q6"], color="#4c78a8")
    ax.axhline(observed_q6_upper, color="black", linestyle="--", linewidth=1, label="extreme Q6 bound")
    ax.set_ylabel("Best grid Q6")
    ax.legend(fontsize=7)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vanilla_vs_dkcsr_best_q6_barplot.png", dpi=300)
    plt.close()

    lines = [
        "# Physical and Optimisation Safety: Vanilla ODE-SR vs DKC-SR",
        "",
        f"- q_scale used: `{Q_SCALE}`.",
        "- Physical rates used a 5 x 5 x 5 grid with 8 replay time points.",
        "- Optimisation safety used a 21 x 21 x 21 grid over the formulation design space.",
        f"- Extreme Q6 upper bound: `{observed_q6_upper}`.",
        "- `positive_dfdQ_rate` reports finite-difference violations of the intended nonpositive df/dQ dynamic prior.",
        "- The DKC-SR row is a replay audit of the selected equation, not a reread of training-time constraint logs.",
        "",
        "## Audit Metrics",
        "",
        *md_table(
            audit,
            [
                "model",
                "seed",
                "negative_Q_prediction_rate",
                "non_monotonic_curve_rate",
                "negative_RHS_rate",
                "positive_dfdQ_rate",
                "numerical_failure_rate",
                "extreme_Q6_rate",
                "best_C1",
                "best_C2",
                "best_C3",
                "best_Q6",
                "best_is_boundary",
                "failure_rate",
                "extreme_Q6_rate_optimisation_grid",
            ],
        ),
        "",
        "## Outputs",
        "",
        f"- `{rel(RESULTS_DIR / 'vanilla_vs_dkcsr_physical_optimisation_audit.csv')}`",
        f"- `{rel(FIGURES_DIR / 'vanilla_vs_dkcsr_physical_rates_barplot.png')}`",
        f"- `{rel(FIGURES_DIR / 'vanilla_vs_dkcsr_best_q6_barplot.png')}`",
    ]
    (REPORT_DIR / "04_vanilla_vs_dkcsr_physical_optimisation_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {rel(RESULTS_DIR / 'vanilla_vs_dkcsr_physical_optimisation_audit.csv')}")


if __name__ == "__main__":
    main()
