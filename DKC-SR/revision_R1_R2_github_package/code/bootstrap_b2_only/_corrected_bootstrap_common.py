from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_validation_bootstrap_correction_24train6test"
CONFIG_DIR = OUT / "config"
RESULTS_DIR = OUT / "results"
REPORT_DIR = OUT / "reports"
FIGURES_DIR = OUT / "figures"
SOURCE_DATA = ROOT / "revision_validation_24train6test" / "data"

SELECTED_SOURCES = [
    ROOT / "artifacts" / "archive" / "ivrt-pair-251007" / "best_sympy.txt",
    ROOT / "artifacts" / "archive" / "ivrt-pair-251007" / "best_infix.txt",
    ROOT / "evaluation" / "used_expr.txt",
]

ROBUSTNESS_DIR = ROOT / "revision_validation_robustness_24train6test"
if str(ROBUSTNESS_DIR) not in sys.path:
    sys.path.insert(0, str(ROBUSTNESS_DIR))

from _robustness_common import (  # noqa: E402
    curve_metrics as _curve_metrics,
    curve_predictions_from_records as _curve_predictions_from_records,
    endpoint_from_curve as _endpoint_from_curve,
    endpoint_metrics as _endpoint_metrics,
    records_from_curve as _records_from_curve,
    simulate_dkcsr_record as _simulate_dkcsr_record,
)


Q_SCALE = 3008.198194823261
A_FIXED = 2.0
B2_ORIGINAL = 2.3550290604627118
B2_BOUNDS = (1e-6, 100.0)
INITIAL_VALUES = [0.5, 1.0, B2_ORIGINAL, 5.0, 10.0]
N_BOOTSTRAP = 500
RANDOM_SEED = 42
EXCLUDED_ROWS = ["S10", "Opt-2", "Opt-4", "Opt-6", "Opt-7", "Opt-10"]


def ensure_dirs() -> None:
    for directory in [OUT, CONFIG_DIR, RESULTS_DIR, REPORT_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def md_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| Item | Value |", "| --- | --- |"]
    for key, value in rows:
        if isinstance(value, (float, np.floating)):
            value_text = f"{float(value):.17g}" if np.isfinite(value) else "nan"
        else:
            value_text = str(value)
        value_text = value_text.replace("|", "\\|")
        lines.append(f"| {key} | {value_text} |")
    return lines


def load_canonical() -> dict[str, pd.DataFrame]:
    return {
        "curve_train": pd.read_csv(SOURCE_DATA / "canonical_curve_train_24.csv"),
        "curve_test": pd.read_csv(SOURCE_DATA / "canonical_curve_test_6.csv"),
        "endpoint_train": pd.read_csv(SOURCE_DATA / "canonical_endpoint_train_24.csv"),
        "endpoint_test": pd.read_csv(SOURCE_DATA / "canonical_endpoint_test_6.csv"),
    }


def records_from_curve(curve: pd.DataFrame) -> list[dict[str, Any]]:
    return _records_from_curve(curve)


def read_expression_sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in SELECTED_SOURCES:
        if path.exists():
            out[rel(path)] = path.read_text(encoding="utf-8").strip()
    return out


def expression_checks(expressions: dict[str, str]) -> dict[str, Any]:
    joined = "\n".join(expressions.values())
    compact = re.sub(r"\s+", "", joined)
    numerator_softplus_2 = any(
        pattern in compact
        for pattern in [
            "Q+Q",
            "2*Q",
            "2*ARG0",
            "exp(2*Q)",
            "exp(2*ARG0)",
            "softplus(2*Q",
            "softplus(2*ARG0",
        ]
    )
    explicit_b2 = f"{B2_ORIGINAL:.16f}" in joined or "2.3550290604627118" in joined
    squared_offsets = [float(x) ** 2 for x in re.findall(r"\((-?\d+\.\d+)\)\*\*2", compact)]
    squared_b2 = any(math.isclose(v, B2_ORIGINAL, rel_tol=1e-12, abs_tol=1e-12) for v in squared_offsets)
    return {
        "numerator_softplus_2": bool(numerator_softplus_2),
        "b2_original_confirmed": bool(explicit_b2 or squared_b2),
        "b2_original_explicit": bool(explicit_b2),
        "b2_original_from_squared_infix": bool(squared_b2),
        "squared_offsets_seen": squared_offsets,
    }


def simulate_corrected_record(rec: dict[str, Any], b2: float) -> np.ndarray:
    return _simulate_dkcsr_record(rec, a=A_FIXED, b2=float(b2))


def curve_predictions(records: list[dict[str, Any]], b2: float, dataset: str, bootstrap_id: int | str = "") -> pd.DataFrame:
    return _curve_predictions_from_records(records, a=A_FIXED, b2=float(b2), dataset=dataset, bootstrap_id=bootstrap_id)


def curve_metric_dict(records: list[dict[str, Any]], b2: float, dataset: str) -> tuple[dict[str, float], pd.DataFrame]:
    pred = curve_predictions(records, b2=b2, dataset=dataset)
    return _curve_metrics(pred), pred


def endpoint_metric_dict(pred: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    endpoint = _endpoint_from_curve(pred)
    return _endpoint_metrics(endpoint), endpoint


def full_metric_bundle(records: list[dict[str, Any]], b2: float, dataset: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    curve_m, pred = curve_metric_dict(records, b2=b2, dataset=dataset)
    endpoint_m, endpoint = endpoint_metric_dict(pred)
    out: dict[str, Any] = {
        f"{dataset}_curve_RMSE": curve_m["RMSE"],
        f"{dataset}_curve_R2": curve_m["R2"],
    }
    out.update({f"{dataset}_{key}": value for key, value in endpoint_m.items()})
    return out, pred, endpoint


def objective_b2(records: list[dict[str, Any]], b2: float) -> float:
    b2 = float(b2)
    if not (B2_BOUNDS[0] <= b2 <= B2_BOUNDS[1]):
        return 1e12
    sse = 0.0
    n = 0
    for rec in records:
        pred = simulate_corrected_record(rec, b2=b2)
        obs = rec["Qtilde_obs"]
        mask = ~np.isclose(rec["time_h"], 0.0)
        if not np.all(np.isfinite(pred[mask])):
            return 1e12
        diff = pred[mask] - obs[mask]
        sse += float(diff @ diff)
        n += int(diff.size)
    return sse / max(1, n)


def fit_b2_from_initial(records: list[dict[str, Any]], initial: float, maxiter: int = 100) -> dict[str, Any]:
    try:
        result = minimize(
            lambda x: objective_b2(records, float(np.asarray(x).ravel()[0])),
            x0=np.asarray([float(initial)], dtype=float),
            method="L-BFGS-B",
            bounds=[B2_BOUNDS],
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
        )
        fun = float(result.fun) if np.isfinite(result.fun) else float("nan")
        b2 = float(result.x[0]) if len(result.x) else float("nan")
        return {
            "initial_value": float(initial),
            "b2_refit": b2,
            "objective_normalized_mse": fun,
            "optimizer_success": bool(result.success),
            "fit_success": bool(np.isfinite(fun) and np.isfinite(b2)),
            "failure_reason": "" if result.success else str(result.message),
            "nfev": int(getattr(result, "nfev", -1)),
            "nit": int(getattr(result, "nit", -1)),
        }
    except Exception as exc:
        return {
            "initial_value": float(initial),
            "b2_refit": float("nan"),
            "objective_normalized_mse": float("nan"),
            "optimizer_success": False,
            "fit_success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "nfev": 0,
            "nit": 0,
        }


def fit_b2_best(records: list[dict[str, Any]], initial_values: list[float] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = [fit_b2_from_initial(records, initial) for initial in (initial_values or [B2_ORIGINAL])]
    table = pd.DataFrame(rows)
    ok = table[table["fit_success"] & np.isfinite(table["objective_normalized_mse"])].copy()
    if ok.empty:
        best = rows[0] if rows else fit_b2_from_initial(records, B2_ORIGINAL)
        return best, table
    idx = ok["objective_normalized_mse"].idxmin()
    best = table.loc[idx].to_dict()
    return best, table


def sanity_pass(abs_diff: float, rel_diff: float) -> bool:
    return bool(abs_diff <= 0.15 and rel_diff <= 0.075)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
