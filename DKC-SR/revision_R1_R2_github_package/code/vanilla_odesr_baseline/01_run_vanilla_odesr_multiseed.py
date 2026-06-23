from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _vanilla_common import (
    Q_SCALE,
    REPORT_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    curve_metrics_table,
    elapsed,
    endpoint_from_curve,
    ensure_dirs,
    expression_complexity,
    individual_to_sympy,
    load_canonical,
    load_config,
    md_table,
    predictions_for_expr,
    records_from_curve,
    rel,
    run_failed,
    train_seed,
    write_json,
)


MODEL_LABEL = "Vanilla ODE-SR"


def save_hof(hof, seed_dir: Path) -> None:
    rows = []
    for rank, ind in enumerate(hof, start=1):
        expr = str(ind)
        try:
            sympy_expr = str(individual_to_sympy(ind))
        except Exception:
            sympy_expr = ""
        rows.append({"rank": rank, "size": len(ind), "fitness_total": float(ind.fitness.values[0]), "infix": expr, "sympy_expr": sympy_expr, **expression_complexity(expr)})
    pd.DataFrame(rows).to_csv(seed_dir / "hof_top.csv", index=False)


def run_one_seed(seed: int, cfg, curve_train: pd.DataFrame, curve_test: pd.DataFrame) -> dict:
    seed_dir = RUNS_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = seed_dir / "run_failed.json"
    if failed_marker.exists():
        failed_marker.unlink()
    start = time.time()
    log_lines = [f"seed={seed}", f"mode={cfg.mode}", f"pop_size={cfg.pop_size}", f"ngen={cfg.ngen}", f"q_scale={Q_SCALE}", "softplus_absent=True"]
    try:
        train_records = records_from_curve(curve_train)
        best, hof, stats, _ = train_seed(seed, cfg, train_records)
        expr = str(best)
        try:
            sympy_expr = str(individual_to_sympy(best))
        except Exception as exc:
            sympy_expr = ""
            log_lines.append(f"sympy_export_warning={type(exc).__name__}: {exc}")
        (seed_dir / "best_expression_infix.txt").write_text(expr + "\n", encoding="utf-8")
        (seed_dir / "best_expression_sympy.txt").write_text(sympy_expr + "\n", encoding="utf-8")
        save_hof(hof, seed_dir)
        stats.to_csv(seed_dir / "generation_stats.csv", index=False)
        write_json(seed_dir / "run_config.json", {**cfg.__dict__, "seed": seed, "q_scale": Q_SCALE, "softplus_absent": True})

        train_pred, fail_train = predictions_for_expr(expr, curve_train, "train", MODEL_LABEL, seed)
        test_pred, fail_test = predictions_for_expr(expr, curve_test, "test", MODEL_LABEL, seed)
        train_pred.to_csv(seed_dir / "train_curve_predictions.csv", index=False)
        test_pred.to_csv(seed_dir / "test_curve_predictions.csv", index=False)
        train_endpoint = endpoint_from_curve(train_pred)
        test_endpoint = endpoint_from_curve(test_pred)
        train_endpoint.to_csv(seed_dir / "train_endpoint_predictions.csv", index=False)
        test_endpoint.to_csv(seed_dir / "test_endpoint_predictions.csv", index=False)

        train_metrics = curve_metrics_table(train_pred)
        test_metrics = curve_metrics_table(test_pred)
        train_primary = train_metrics[(train_metrics["dataset"] == "train") & (train_metrics["scope"] == "overall_excluding_t0")].iloc[0].to_dict()
        test_primary = test_metrics[(test_metrics["dataset"] == "test") & (test_metrics["scope"] == "overall_excluding_t0")].iloc[0].to_dict()
        write_json(seed_dir / "train_metrics.json", {k: (float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else str(v)) for k, v in train_primary.items()})
        write_json(seed_dir / "test_metrics.json", {k: (float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else str(v)) for k, v in test_primary.items()})
        log_lines.append(f"runtime_seconds={elapsed(start)}")
        log_lines.append(f"best_expression={expr}")
        log_lines.append(f"best_fitness={float(best.fitness.values[0]):.8g}")
        (seed_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {
            "seed": seed,
            "status": "success",
            "runtime_seconds": float(time.time() - start),
            "best_expression": expr,
            "complexity": int(len(best)),
            "train_RMSE_curve": float(train_primary["RMSE"]),
            "train_R2_curve": float(train_primary["R2"]),
            "test_RMSE_curve": float(test_primary["RMSE"]),
            "test_R2_curve": float(test_primary["R2"]),
            "test_MSE_normalized_by_q_scale": float(test_primary["MSE_normalized_by_q_scale"]),
            "numerical_failure_count": int(fail_train + fail_test),
            "notes": f"softplus_absent=True; sympy_export={'ok' if sympy_expr else 'failed'}",
        }
    except Exception as exc:
        run_failed(seed_dir, seed, "train_or_export", exc)
        log_lines.append(f"runtime_seconds={elapsed(start)}")
        log_lines.append(f"failed={type(exc).__name__}: {exc}")
        (seed_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {
            "seed": seed,
            "status": "failed",
            "runtime_seconds": float(time.time() - start),
            "best_expression": "",
            "complexity": np.nan,
            "train_RMSE_curve": np.nan,
            "train_R2_curve": np.nan,
            "test_RMSE_curve": np.nan,
            "test_R2_curve": np.nan,
            "test_MSE_normalized_by_q_scale": np.nan,
            "numerical_failure_count": np.nan,
            "notes": f"{type(exc).__name__}: {exc}",
        }


def write_report(summary: pd.DataFrame, cfg) -> None:
    lines = [
        "# Vanilla ODE-SR Multiseed Run Report",
        "",
        f"- q_scale used: `{Q_SCALE}`.",
        f"- Run mode: `{cfg.mode}`.",
        f"- Seeds: `{cfg.seeds}`.",
        f"- Population: `{cfg.pop_size}`; generations: `{cfg.ngen}`; Hall of Fame: `{cfg.hall_of_fame_size}`.",
        "- Training used only the 24 canonical training formulations; the 6 held-out test formulations were replayed after training.",
        "- Primitive set excludes `softplus`.",
        "",
        "## Run Summary",
        "",
        *md_table(summary, ["seed", "status", "runtime_seconds", "complexity", "train_RMSE_curve", "test_RMSE_curve", "test_R2_curve", "numerical_failure_count", "notes"]),
        "",
        "## Outputs",
        "",
        f"- `{rel(RESULTS_DIR / 'vanilla_odesr_multiseed_run_summary.csv')}`",
        f"- `{rel(RUNS_DIR)}/seed_<seed>/...`",
    ]
    (REPORT_DIR / "01_vanilla_odesr_multiseed_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dryrun", "full"], default=None)
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed subset.")
    args = parser.parse_args()
    ensure_dirs()
    cfg = load_config(args.mode)
    if args.seeds.strip():
        cfg.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    canon = load_canonical()
    rows = []
    for seed in cfg.seeds:
        print(f"[run] seed={seed}, mode={cfg.mode}, pop={cfg.pop_size}, ngen={cfg.ngen}", flush=True)
        rows.append(run_one_seed(seed, cfg, canon["curve_train"], canon["curve_test"]))
    summary_path = RESULTS_DIR / "vanilla_odesr_multiseed_run_summary.csv"
    if summary_path.exists() and args.seeds.strip():
        old = pd.read_csv(summary_path)
        old = old[~old["seed"].astype(int).isin(cfg.seeds)]
        summary = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).sort_values("seed")
    else:
        summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    write_report(summary, cfg)
    print(f"[OK] {rel(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
