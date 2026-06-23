from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
OUT = ROOT / "revision_validation_24train6test"
DATA_DIR = OUT / "data"
CONFIG_PATH = OUT / "config" / "revision_24train6test_config.json"
RESULTS_DIR = OUT / "results"
FIGURES_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"

Q_SCALE = 3008.198194823261
RANDOM_STATE = 42
FEATURES = ["C1", "C2", "C3", "time_h"]
MODELS = [
    "Mean train baseline",
    "Ridge regression",
    "PLS regression",
    "Polynomial RSM degree 2",
    "Random Forest Regressor",
    "Gaussian Process Regressor",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def ensure_dirs() -> None:
    for directory in [RESULTS_DIR, FIGURES_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{rel(CONFIG_PATH)} not found. Run 00_create_24train6test_dataset.py first.")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "R2": float("nan"),
            "MSE": float("nan"),
            "MSE_normalized_by_q_scale": float("nan"),
        }
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": r2_score_safe(y_true, y_pred),
        "MSE": mse,
        "MSE_normalized_by_q_scale": float(mse / (Q_SCALE**2)),
    }


def make_model(model_name: str, n_features: int, n_samples: int) -> Any:
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


def mean_by_time_predictions(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    means = train.groupby("time_h")["Q_obs"].mean().to_dict()
    overall = float(train["Q_obs"].mean())
    return target["time_h"].map(lambda t: means.get(float(t), overall)).to_numpy(float)


def predict_single_target(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    if model_name == "Mean train baseline":
        return mean_by_time_predictions(train, target)
    model = make_model(model_name, X_train.shape[1], X_train.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    return np.asarray(model.predict(X_pred), dtype=float).reshape(-1)


def build_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    X_train = train[FEATURES].to_numpy(float)
    y_train = train["Q_obs"].to_numpy(float)
    for model_name in MODELS:
        for target in [train, test]:
            base = target.copy()
            base["model"] = model_name
            base["Q_pred"] = predict_single_target(model_name, X_train, y_train, target[FEATURES].to_numpy(float), train, target)
            rows.append(base)
    pred = pd.concat(rows, ignore_index=True)
    keep = ["model", "dataset", "record_id", "record_index", "run_no", "time_h", "Q_obs", "Q_pred", "C1", "C2", "C3", "C1n", "C2n", "C3n"]
    return pred[keep]


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, dataset), g in pred.groupby(["model", "dataset"], sort=False):
        scopes = [
            ("overall_excluding_t0", g.loc[~np.isclose(g["time_h"], 0.0)]),
            ("overall_including_t0", g),
        ]
        for scope, sub in scopes:
            m = regression_metrics(sub["Q_obs"].to_numpy(float), sub["Q_pred"].to_numpy(float))
            rows.append({"model": model, "dataset": dataset, "scope": scope, "time_h": "", "n_points": int(len(sub)), **m})
        for time_h, sub in g.groupby("time_h", sort=True):
            m = regression_metrics(sub["Q_obs"].to_numpy(float), sub["Q_pred"].to_numpy(float))
            rows.append({"model": model, "dataset": dataset, "scope": "time_point", "time_h": float(time_h), "n_points": int(len(sub)), **m})
    return pd.DataFrame(rows)


def add_identity(ax: plt.Axes, y: np.ndarray, yhat: np.ndarray) -> None:
    vals = np.concatenate([y[np.isfinite(y)], yhat[np.isfinite(yhat)]])
    lo = float(np.nanmin(vals)) if vals.size else 0.0
    hi = float(np.nanmax(vals)) if vals.size else 1.0
    pad = 0.05 * max(1.0, hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def make_parity(pred: pd.DataFrame, path: Path) -> None:
    test = pred[(pred["dataset"] == "test") & (~np.isclose(pred["time_h"], 0.0))]
    plt.figure(figsize=(7.4, 6.0))
    ax = plt.gca()
    for model, g in test.groupby("model", sort=False):
        ax.scatter(g["Q_obs"], g["Q_pred"], s=28, alpha=0.68, label=model)
    add_identity(ax, test["Q_obs"].to_numpy(float), test["Q_pred"].to_numpy(float))
    ax.set_xlabel("Observed Q")
    ax.set_ylabel("Predicted Q")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


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


def write_curve_report(config: dict[str, Any], metrics: pd.DataFrame) -> None:
    test_primary = metrics[
        (metrics["dataset"] == "test") & (metrics["scope"] == "overall_excluding_t0")
    ].sort_values("RMSE")
    text: list[str] = []
    text.append("# Curve Proxy Baselines, 24 Train + 6 Test")
    text.append("")
    text.append("## Context")
    text.append(f"- q_scale used: `{Q_SCALE}`.")
    text.append("- Models were trained on raw physical `Q_obs` values.")
    text.append("- Inputs were `C1, C2, C3, time_h`; output was `Q_obs`.")
    text.append("- Primary curve metrics exclude `time_h = 0` because `Q(0)=0` is trivial.")
    text.append("- The mean baseline predicts the mean training Q at each time point.")
    text.append(f"- Train Run No values: `{config['run_no']['train_first_24']}`.")
    text.append(f"- Test Run No values: `{config['run_no']['test_6']}`.")
    text.append("")
    text.append("## Test Curve Metrics, Excluding t=0")
    text.extend(md_table(test_primary, ["model", "RMSE", "MAE", "R2", "MSE_normalized_by_q_scale", "n_points"]))
    text.append("")
    text.append("## Outputs")
    for path in [
        RESULTS_DIR / "curve_proxy_baseline_metrics_24train6test.csv",
        RESULTS_DIR / "curve_proxy_baseline_predictions_24train6test.csv",
        FIGURES_DIR / "curve_proxy_parity_24train6test.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "02_curve_proxy_baselines_24train6test_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_summary_report(config: dict[str, Any], endpoint_metrics: pd.DataFrame, curve_metrics: pd.DataFrame) -> None:
    endpoint_test = endpoint_metrics[endpoint_metrics["dataset"] == "test"].sort_values("RMSE_Q6")
    curve_test = curve_metrics[
        (curve_metrics["dataset"] == "test") & (curve_metrics["scope"] == "overall_excluding_t0")
    ].sort_values("RMSE")
    best_endpoint_model = str(endpoint_test.iloc[0]["model"]) if len(endpoint_test) else "not available"

    text: list[str] = []
    text.append("# Summary: 24 Train + 6 Test Baseline Comparison")
    text.append("")
    text.append("## Required Statements")
    text.append(f"- q_scale used: `{Q_SCALE}`.")
    text.append("- Training set: first 24 rows from `Formulas-train` and `Release-train`.")
    text.append("- Test set: all 6 rows from `Formulas-test` and `Release-test`.")
    text.append(f"- Excluded 6 rows from original training sheet: `{config['run_no']['excluded_original_train_6']}`.")
    text.append(f"- Train Run No values: `{config['run_no']['train_first_24']}`.")
    text.append(f"- Test Run No values: `{config['run_no']['test_6']}`.")
    text.append(f"- Train/test leakage detected: `{config['leak_checks']['any_train_test_leakage']}`.")
    text.append("- No unconstrained SR, DKC-SR retraining, repeated splits, or bootstrap refitting were run.")
    text.append("")
    text.append("## Baseline Models Trained")
    for model_name in MODELS:
        text.append(f"- {model_name}")
    text.append("")
    text.append("## Endpoint Metrics Table")
    endpoint_cols = [
        "model",
        "RMSE_Q6",
        "MAE_Q6",
        "R2_Q6",
        "Spearman_Q6",
        "Kendall_Q6",
        "pairwise_accuracy_Q6",
        "top1_hit_Q6",
        "top2_hit_Q6",
        "RMSE_AUC",
        "MAE_AUC",
        "R2_AUC",
        "Spearman_AUC",
        "pairwise_accuracy_AUC",
    ]
    text.extend(md_table(endpoint_test, endpoint_cols))
    text.append("")
    text.append("## Curve Proxy Metrics Table")
    text.append("- Primary curve metrics exclude `time_h = 0`.")
    text.extend(md_table(curve_test, ["model", "RMSE", "MAE", "R2", "MSE_normalized_by_q_scale", "n_points"]))
    text.append("")
    text.append("## Recommended Manuscript Table")
    text.append("- Use the endpoint baseline table as the main Reviewer #1 comparison because it evaluates formulation-level predictions on the same 24-train/6-test split.")
    text.append("- Recommended columns: model, Q6 RMSE, Q6 MAE, Q6 R2, Q6 Spearman, Q6 pairwise accuracy, Q6 top-1/top-2 hits, AUC RMSE, AUC Spearman, and AUC pairwise accuracy.")
    text.append(f"- Best endpoint model by held-out Q6 RMSE in this run: `{best_endpoint_model}`.")
    text.append("- Use the curve proxy table as supplementary evidence only, since those models learn a direct `time_h -> Q` regression rather than an ODE/SR mechanism.")
    text.append("")
    text.append("## Review Files")
    for path in [
        REPORT_DIR / "00_dataset_24train6test_report.md",
        REPORT_DIR / "01_static_endpoint_baselines_24train6test_report.md",
        REPORT_DIR / "02_curve_proxy_baselines_24train6test_report.md",
        RESULTS_DIR / "static_endpoint_baseline_metrics_24train6test.csv",
        RESULTS_DIR / "static_endpoint_baseline_predictions_24train6test.csv",
        RESULTS_DIR / "curve_proxy_baseline_metrics_24train6test.csv",
        RESULTS_DIR / "curve_proxy_baseline_predictions_24train6test.csv",
        FIGURES_DIR / "static_q6_parity_24train6test.png",
        FIGURES_DIR / "static_auc_parity_24train6test.png",
        FIGURES_DIR / "static_q6_ranking_barplot_24train6test.png",
        FIGURES_DIR / "curve_proxy_parity_24train6test.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "summary_24train6test_baseline_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = read_config()
    train = pd.read_csv(DATA_DIR / "canonical_curve_train_24.csv")
    test = pd.read_csv(DATA_DIR / "canonical_curve_test_6.csv")

    pred = build_predictions(train, test)
    metrics = metrics_table(pred)
    pred.to_csv(RESULTS_DIR / "curve_proxy_baseline_predictions_24train6test.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "curve_proxy_baseline_metrics_24train6test.csv", index=False)
    make_parity(pred, FIGURES_DIR / "curve_proxy_parity_24train6test.png")
    write_curve_report(config, metrics)

    endpoint_metrics_path = RESULTS_DIR / "static_endpoint_baseline_metrics_24train6test.csv"
    if endpoint_metrics_path.exists():
        endpoint_metrics = pd.read_csv(endpoint_metrics_path)
        write_summary_report(config, endpoint_metrics, metrics)

    print("[OK] Curve proxy baselines complete.")
    print(f"[OK] Wrote {rel(RESULTS_DIR / 'curve_proxy_baseline_metrics_24train6test.csv')}")


if __name__ == "__main__":
    main()
