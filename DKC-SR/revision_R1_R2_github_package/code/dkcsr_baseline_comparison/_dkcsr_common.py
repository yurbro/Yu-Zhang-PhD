from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_validation_24train6test_dkcsr"
DATA_DIR = OUT / "data"
RESULTS_DIR = OUT / "results"
FIGURES_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"

SOURCE_24_DIR = ROOT / "revision_validation_24train6test"
SOURCE_24_DATA = SOURCE_24_DIR / "data"
SOURCE_24_RESULTS = SOURCE_24_DIR / "results"
SOURCE_24_CONFIG = SOURCE_24_DIR / "config" / "revision_24train6test_config.json"
SELECTED_ARTIFACT_DIR = ROOT / "artifacts" / "archive" / "ivrt-pair-251007"
BEST_SYMPY = SELECTED_ARTIFACT_DIR / "best_sympy.txt"
EXISTING_PRED_TEST_SIX = ROOT / "evaluation" / "pred_test-six.csv"

Q_SCALE = 975.1784004393666
RANDOM_STATE = 42
TIME_POINTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
FEATURES_CURVE = ["C1", "C2", "C3", "time_h"]
STATIC_PLAUSIBILITY_MODELS = [
    "PLS regression",
    "Ridge regression",
    "Polynomial RSM degree 2",
    "Random Forest Regressor",
    "Gaussian Process Regressor",
]
C_BOUNDS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def ensure_dirs() -> None:
    for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_config() -> dict[str, Any]:
    if not SOURCE_24_CONFIG.exists():
        raise FileNotFoundError(f"{rel(SOURCE_24_CONFIG)} not found.")
    return json.loads(SOURCE_24_CONFIG.read_text(encoding="utf-8"))


def load_24_canonical() -> dict[str, pd.DataFrame]:
    return {
        "curve_train": pd.read_csv(SOURCE_24_DATA / "canonical_curve_train_24.csv"),
        "curve_test": pd.read_csv(SOURCE_24_DATA / "canonical_curve_test_6.csv"),
        "endpoint_train": pd.read_csv(SOURCE_24_DATA / "canonical_endpoint_train_24.csv"),
        "endpoint_test": pd.read_csv(SOURCE_24_DATA / "canonical_endpoint_test_6.csv"),
    }


def canonical_30_train_or_none() -> pd.DataFrame | None:
    path = ROOT / "revision_validation" / "data" / "canonical_curve_train.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def normalize_c(name: str, value: float) -> float:
    lo, hi = C_BOUNDS[name]
    return (float(value) - lo) / (hi - lo)


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, q_scale: float | None = None) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "MSE": float("nan"), "MSE_normalized_by_q_scale": float("nan")}
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": r2_score_safe(y_true, y_pred),
        "MSE": mse,
        "MSE_normalized_by_q_scale": float(mse / (q_scale**2)) if q_scale and q_scale > 0 else float("nan"),
    }


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int]:
    correct = 0
    total = 0
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    for i in range(len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            st = np.sign(y_true[i] - y_true[j])
            sp = np.sign(y_pred[i] - y_pred[j])
            if st == 0 or sp == 0:
                continue
            total += 1
            correct += int(st == sp)
    return (float(correct / total) if total else float("nan"), correct, total)


def top_hit_sets(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    if not labels:
        return {"top1_hit": 0, "top2_hit": 0, "true_top1": "[]", "pred_top1": "[]", "true_top2": "[]", "pred_top2": "[]"}
    order_true = np.argsort(-np.asarray(y_true, dtype=float))
    order_pred = np.argsort(-np.asarray(y_pred, dtype=float))
    true_top1 = [labels[int(order_true[0])]]
    pred_top1 = [labels[int(order_pred[0])]]
    k2 = min(2, len(labels))
    true_top2 = [labels[int(i)] for i in order_true[:k2]]
    pred_top2 = [labels[int(i)] for i in order_pred[:k2]]
    return {
        "top1_hit": int(true_top1[0] == pred_top1[0]),
        "top2_hit": int(bool(set(true_top2) & set(pred_top2))),
        "true_top1": json.dumps(true_top1),
        "pred_top1": json.dumps(pred_top1),
        "true_top2": json.dumps(true_top2),
        "pred_top2": json.dumps(pred_top2),
    }


def rank_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(y_true, y_pred).correlation if len(y_true) >= 2 else float("nan")
        tau = kendalltau(y_true, y_pred).correlation if len(y_true) >= 2 else float("nan")
    pair_acc, pair_correct, pair_total = pairwise_accuracy(y_true, y_pred)
    return {
        "Spearman": float(rho) if rho is not None else float("nan"),
        "Kendall": float(tau) if tau is not None else float("nan"),
        "pairwise_accuracy": pair_acc,
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        **top_hit_sets(y_true, y_pred, labels),
    }


def endpoint_metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, dataset), g in pred.groupby(["model", "dataset"], sort=False):
        labels = g["run_no"].astype(str).tolist()
        row: dict[str, Any] = {"model": model, "dataset": dataset, "n_formulations": int(len(g))}
        for prefix, obs_col, pred_col in [("Q6", "Q6_obs", "Q6_pred"), ("AUC", "AUC_obs", "AUC_pred")]:
            y_true = g[obs_col].to_numpy(float)
            y_pred = g[pred_col].to_numpy(float)
            reg = regression_metrics(y_true, y_pred, q_scale=Q_SCALE if prefix == "Q6" else None)
            rank = rank_metrics(y_true, y_pred, labels)
            row[f"RMSE_{prefix}"] = reg["RMSE"]
            row[f"MAE_{prefix}"] = reg["MAE"]
            row[f"R2_{prefix}"] = reg["R2"]
            row[f"MSE_{prefix}"] = reg["MSE"]
            if prefix == "Q6":
                row["MSE_Q6_normalized_by_q_scale"] = reg["MSE_normalized_by_q_scale"]
            row[f"Spearman_{prefix}"] = rank["Spearman"]
            row[f"Kendall_{prefix}"] = rank["Kendall"]
            row[f"pairwise_accuracy_{prefix}"] = rank["pairwise_accuracy"]
            row[f"pairwise_correct_{prefix}"] = rank["pairwise_correct"]
            row[f"pairwise_total_{prefix}"] = rank["pairwise_total"]
            row[f"top1_hit_{prefix}"] = rank["top1_hit"]
            row[f"top2_hit_{prefix}"] = rank["top2_hit"]
            row[f"true_top1_{prefix}"] = rank["true_top1"]
            row[f"pred_top1_{prefix}"] = rank["pred_top1"]
            row[f"true_top2_{prefix}"] = rank["true_top2"]
            row[f"pred_top2_{prefix}"] = rank["pred_top2"]
        rows.append(row)
    return pd.DataFrame(rows)


def curve_metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, dataset), g in pred.groupby(["model", "dataset"], sort=False):
        for scope, sub in [
            ("overall_excluding_t0", g.loc[~np.isclose(g["time_h"], 0.0)]),
            ("overall_including_t0", g),
        ]:
            m = regression_metrics(sub["Q_obs"].to_numpy(float), sub["Q_pred"].to_numpy(float), q_scale=Q_SCALE)
            rows.append({"model": model, "dataset": dataset, "scope": scope, "time_h": "", "n_points": int(len(sub)), **m})
        for time_h, sub in g.groupby("time_h", sort=True):
            m = regression_metrics(sub["Q_obs"].to_numpy(float), sub["Q_pred"].to_numpy(float), q_scale=Q_SCALE)
            rows.append({"model": model, "dataset": dataset, "scope": "time_point", "time_h": float(time_h), "n_points": int(len(sub)), **m})
    return pd.DataFrame(rows)


def sat(x: float, lo: float = -1e6, hi: float = 1e6) -> float:
    try:
        if not math.isfinite(float(x)):
            return 0.0
    except Exception:
        return 0.0
    return min(max(float(x), lo), hi)


def softplus(x: float) -> float:
    x = sat(x)
    if x > 20.0:
        return x
    if x < -20.0:
        return sat(math.exp(x))
    return sat(math.log1p(math.exp(x)))


def build_dkc_callable() -> tuple[Callable[[float, float, float, float], float], str, str]:
    if not BEST_SYMPY.exists():
        raise FileNotFoundError(BEST_SYMPY)
    expr = BEST_SYMPY.read_text(encoding="utf-8").strip()
    normalized = expr
    for old, new in {"ARG0": "Q", "ARG1": "C1", "ARG2": "C2", "ARG3": "C3"}.items():
        normalized = re.sub(rf"\b{old}\b", new, normalized)
    normalized = re.sub(r"log\s*\(\s*1\s*\+\s*exp\s*\(", "softplus(", normalized)
    normalized = re.sub(r"log\s*\(\s*exp\s*\(([^)]*)\)\s*\+\s*1\s*\)", r"softplus(\1)", normalized)
    code = compile(normalized, "<dkcsr_qscale975_24train6test>", "eval")
    safe_env = {
        "__builtins__": {},
        "log": math.log,
        "exp": lambda x: math.exp(max(-50.0, min(50.0, sat(x)))),
        "sqrt": lambda x: math.sqrt(max(0.0, sat(x))),
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "softplus": softplus,
    }

    def f(qtilde: float, c1n: float, c2n: float, c3n: float) -> float:
        q = max(0.0, float(qtilde))
        c3 = float(c3n)
        eps = 1e-8
        if abs(c3) < eps:
            c3 = eps if c3 >= 0 else -eps
        return float(eval(code, safe_env, {"Q": q, "C1": float(c1n), "C2": float(c2n), "C3": c3}))

    return f, normalized, rel(BEST_SYMPY)


def step_rk4_qtilde(
    f: Callable[[float, float, float, float], float],
    qtilde: float,
    dt: float,
    c1n: float,
    c2n: float,
    c3n: float,
    qcap_tilde: float,
) -> float:
    k1 = sat(f(qtilde, c1n, c2n, c3n))
    k2 = sat(f(qtilde + 0.5 * dt * k1, c1n, c2n, c3n))
    k3 = sat(f(qtilde + 0.5 * dt * k2, c1n, c2n, c3n))
    k4 = sat(f(qtilde + dt * k3, c1n, c2n, c3n))
    out = sat(qtilde + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
    return min(max(out, 0.0), qcap_tilde)


def simulate_dkc_times(
    f: Callable[[float, float, float, float], float],
    time_points: np.ndarray,
    c1n: float,
    c2n: float,
    c3n: float,
    qcap_raw: float,
) -> np.ndarray:
    t = np.asarray(time_points, dtype=float)
    qcap_tilde = max(1.0, float(qcap_raw) / Q_SCALE)
    qtilde = 0.0
    out_raw = np.empty_like(t, dtype=float)
    out_raw[0] = 0.0
    substeps = 8
    adapt_refine_max = 8
    dt_floor = 1e-6
    for k in range(1, len(t)):
        dt_total = float(t[k] - t[k - 1])
        if dt_total <= 0:
            out_raw[k] = qtilde * Q_SCALE
            continue
        ok = False
        for refine in range(adapt_refine_max + 1):
            n_steps = substeps * (2**refine)
            dt = max(dt_total / n_steps, dt_floor)
            q_try = qtilde
            finite = True
            for _ in range(n_steps):
                q_try = step_rk4_qtilde(f, q_try, dt, c1n, c2n, c3n, qcap_tilde)
                if not math.isfinite(q_try):
                    finite = False
                    break
            if finite:
                qtilde = q_try
                ok = True
                break
        out_raw[k] = qtilde * Q_SCALE if ok else np.nan
    return out_raw


def replay_dkc_for_curve(curve: pd.DataFrame, model_label: str = "DKC-SR replayed equation, q_scale=3008.198194823261") -> pd.DataFrame:
    f, _, _ = build_dkc_callable()
    rows: list[pd.DataFrame] = []
    for _, g in curve.groupby(["dataset", "record_id"], sort=False):
        gg = g.sort_values("time_h").copy()
        qcap_raw = max(1.0, 1.5 * float(np.nanmax(gg["Q_obs"].to_numpy(float))))
        gg["model"] = model_label
        gg["Q_pred"] = simulate_dkc_times(
            f,
            gg["time_h"].to_numpy(float),
            float(gg["C1n"].iloc[0]),
            float(gg["C2n"].iloc[0]),
            float(gg["C3n"].iloc[0]),
            qcap_raw=qcap_raw,
        )
        rows.append(gg)
    keep = ["model", "dataset", "record_id", "record_index", "run_no", "time_h", "Q_obs", "Q_pred", "C1", "C2", "C3", "C1n", "C2n", "C3n"]
    return pd.concat(rows, ignore_index=True)[keep]


def canonical_groups(canonical: pd.DataFrame) -> list[tuple[int, str, pd.DataFrame]]:
    out = []
    for idx, g in canonical.groupby("record_index", sort=False):
        gg = g.sort_values("time_h").reset_index(drop=True)
        out.append((int(idx), str(gg["run_no"].iloc[0]), gg))
    return out


def map_prediction_file_to_canonical(path: Path, canonical: pd.DataFrame, model_label: str, tolerance: float = 1e-5) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    info: dict[str, Any] = {"file_path": rel(path) if path.exists() else str(path), "loaded": False}
    if not path.exists():
        info["reason"] = "file not found"
        return None, info
    df = pd.read_csv(path)
    needed = {"record_idx", "time_h", "Q_obs", "Q_pred"}
    if not needed.issubset(df.columns):
        info["reason"] = f"missing columns: {sorted(needed - set(df.columns))}"
        return None, info

    pred_groups = [(int(idx), g.sort_values("time_h").reset_index(drop=True)) for idx, g in df.groupby("record_idx", sort=True)]
    can_groups = canonical_groups(canonical)
    used_can: set[int] = set()
    rows: list[pd.DataFrame] = []
    matches: list[dict[str, Any]] = []
    all_good = True
    max_diff = float("nan")

    for pred_idx, pred_group in pred_groups:
        pred_q = pred_group["Q_obs"].to_numpy(float)
        best: tuple[int, str, pd.DataFrame, float] | None = None
        for can_idx, run_no, can_group in can_groups:
            if can_idx in used_can or len(can_group) != len(pred_group):
                continue
            diff = float(np.nanmax(np.abs(can_group["Q_obs"].to_numpy(float) - pred_q)))
            if best is None or diff < best[3]:
                best = (can_idx, run_no, can_group, diff)
        if best is None:
            all_good = False
            matches.append({"pred_record_idx": pred_idx, "canonical_record_index": "", "run_no": "", "max_abs_diff_Q_obs": float("nan"), "matched": False})
            continue
        can_idx, run_no, can_group, diff = best
        max_diff = diff if not np.isfinite(max_diff) else max(max_diff, diff)
        matched = diff <= tolerance
        all_good = all_good and matched
        used_can.add(can_idx)
        matches.append({"pred_record_idx": pred_idx, "canonical_record_index": can_idx, "run_no": run_no, "max_abs_diff_Q_obs": diff, "matched": matched})
        if matched:
            out = can_group.copy()
            out["model"] = model_label
            out["Q_pred"] = pred_group["Q_pred"].to_numpy(float)
            rows.append(out)

    all_good = all_good and len(used_can) == len(can_groups) and len(pred_groups) == len(can_groups)
    sequence_match = False
    sequence_max_diff = float("nan")
    if len(df) == len(canonical):
        can_order = canonical.sort_values(["record_index", "time_h"]).reset_index(drop=True)["Q_obs"].to_numpy(float)
        df_order = df.sort_values(["record_idx", "time_h"]).reset_index(drop=True)["Q_obs"].to_numpy(float)
        sequence_max_diff = float(np.nanmax(np.abs(can_order - df_order)))
        sequence_match = bool(np.allclose(can_order, df_order, atol=tolerance, rtol=0.0))

    info.update(
        {
            "loaded": bool(all_good and rows),
            "reason": "curve set matched canonical data" if all_good and rows else "curve set did not fully match canonical data",
            "sequence_matches_canonical_order": sequence_match,
            "sequence_max_abs_diff_Q_obs": sequence_max_diff,
            "curve_set_matches_canonical": bool(all_good and rows),
            "curve_set_max_abs_diff_Q_obs": max_diff,
            "matches": matches,
            "matched_run_no": [m["run_no"] for m in matches if m.get("matched")],
        }
    )
    if not rows or not all_good:
        return None, info
    mapped = pd.concat(rows, ignore_index=True).sort_values(["record_index", "time_h"]).reset_index(drop=True)
    keep = ["model", "dataset", "record_id", "record_index", "run_no", "time_h", "Q_obs", "Q_pred", "C1", "C2", "C3", "C1n", "C2n", "C3n"]
    return mapped[keep], info


def endpoint_from_curve(curve_pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, dataset, record_id, run_no), g in curve_pred.groupby(["model", "dataset", "record_id", "run_no"], sort=False):
        gg = g.sort_values("time_h")
        first = gg.iloc[0]
        q6_obs = gg.loc[np.isclose(gg["time_h"], 6.0), "Q_obs"].iloc[0]
        q6_pred = gg.loc[np.isclose(gg["time_h"], 6.0), "Q_pred"].iloc[0]
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "record_id": record_id,
                "record_index": int(first["record_index"]),
                "run_no": run_no,
                "C1": float(first["C1"]),
                "C2": float(first["C2"]),
                "C3": float(first["C3"]),
                "C1n": float(first["C1n"]),
                "C2n": float(first["C2n"]),
                "C3n": float(first["C3n"]),
                "Q6_obs": float(q6_obs),
                "Q6_pred": float(q6_pred),
                "AUC_obs": float(np.trapz(gg["Q_obs"].to_numpy(float), gg["time_h"].to_numpy(float))),
                "AUC_pred": float(np.trapz(gg["Q_pred"].to_numpy(float), gg["time_h"].to_numpy(float))),
            }
        )
    out = pd.DataFrame(rows)
    for target in ["Q6", "AUC"]:
        out[f"{target}_obs_rank_desc"] = out.groupby(["dataset", "model"])[f"{target}_obs"].rank(ascending=False, method="min")
        out[f"{target}_pred_rank_desc"] = out.groupby(["dataset", "model"])[f"{target}_pred"].rank(ascending=False, method="min")
    return out


def add_identity(ax: plt.Axes, y: np.ndarray, yhat: np.ndarray) -> None:
    vals = np.concatenate([y[np.isfinite(y)], yhat[np.isfinite(yhat)]])
    lo = float(np.nanmin(vals)) if vals.size else 0.0
    hi = float(np.nanmax(vals)) if vals.size else 1.0
    pad = 0.05 * max(1.0, hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def md_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.6g}" if np.isfinite(val) else "nan"
            vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def make_static_curve_model(model_name: str, n_features: int, n_samples: int) -> Any:
    if model_name == "Ridge regression":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if model_name == "PLS regression":
        n_components = max(1, min(2, n_features, n_samples - 1))
        return Pipeline([("scale", StandardScaler()), ("model", PLSRegression(n_components=n_components))])
    if model_name == "Polynomial RSM degree 2":
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            min_samples_leaf=1,
            max_features=1.0,
        )
    if model_name == "Gaussian Process Regressor":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level=1.0)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-6,
                        normalize_y=True,
                        random_state=RANDOM_STATE,
                        n_restarts_optimizer=0,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def fit_static_curve_models(curve_train: pd.DataFrame) -> dict[str, Any]:
    X = curve_train[FEATURES_CURVE].to_numpy(float)
    y = curve_train["Q_obs"].to_numpy(float)
    models: dict[str, Any] = {}
    for model_name in STATIC_PLAUSIBILITY_MODELS:
        model = make_static_curve_model(model_name, X.shape[1], X.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X, y)
        models[model_name] = model
    return models


def build_grid() -> pd.DataFrame:
    values = {
        "C1": np.linspace(C_BOUNDS["C1"][0], C_BOUNDS["C1"][1], 5),
        "C2": np.linspace(C_BOUNDS["C2"][0], C_BOUNDS["C2"][1], 5),
        "C3": np.linspace(C_BOUNDS["C3"][0], C_BOUNDS["C3"][1], 5),
    }
    rows: list[dict[str, Any]] = []
    grid_id = 0
    for c1 in values["C1"]:
        for c2 in values["C2"]:
            for c3 in values["C3"]:
                for time_h in TIME_POINTS:
                    rows.append(
                        {
                            "grid_id": grid_id,
                            "C1": float(c1),
                            "C2": float(c2),
                            "C3": float(c3),
                            "C1n": normalize_c("C1", c1),
                            "C2n": normalize_c("C2", c2),
                            "C3n": normalize_c("C3", c3),
                            "time_h": float(time_h),
                        }
                    )
                grid_id += 1
    return pd.DataFrame(rows)


def is_boundary_row(row: pd.Series) -> bool:
    return any(
        np.isclose(float(row[name]), C_BOUNDS[name][0]) or np.isclose(float(row[name]), C_BOUNDS[name][1])
        for name in ["C1", "C2", "C3"]
    )

