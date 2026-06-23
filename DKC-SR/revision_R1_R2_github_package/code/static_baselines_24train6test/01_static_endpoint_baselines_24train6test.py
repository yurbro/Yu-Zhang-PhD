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
OUT = ROOT / "revision_validation_24train6test"
DATA_DIR = OUT / "data"
CONFIG_PATH = OUT / "config" / "revision_24train6test_config.json"
RESULTS_DIR = OUT / "results"
FIGURES_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"

Q_SCALE = 3008.198194823261
RANDOM_STATE = 42
FEATURES = ["C1", "C2", "C3"]
TARGETS = ["Q6_obs", "AUC_obs"]
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
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "MSE": float("nan")}
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": r2_score_safe(y_true, y_pred),
        "MSE": mse,
    }


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int]:
    correct = 0
    total = 0
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    for i in range(len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            true_sign = np.sign(y_true[i] - y_true[j])
            pred_sign = np.sign(y_pred[i] - y_pred[j])
            if true_sign == 0 or pred_sign == 0:
                continue
            total += 1
            correct += int(true_sign == pred_sign)
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


def predict_single_target(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    if model_name == "Mean train baseline":
        return np.full(X_pred.shape[0], float(np.nanmean(y_train)))
    model = make_model(model_name, X_train.shape[1], X_train.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    return np.asarray(model.predict(X_pred), dtype=float).reshape(-1)


def build_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    X_train = train[FEATURES].to_numpy(float)
    for model_name in MODELS:
        q6_train = train["Q6_obs"].to_numpy(float)
        auc_train = train["AUC_obs"].to_numpy(float)
        for dataset_name, target in [("train", train), ("test", test)]:
            base = target.copy()
            X_target = target[FEATURES].to_numpy(float)
            base["model"] = model_name
            base["Q6_pred"] = predict_single_target(model_name, X_train, q6_train, X_target)
            base["AUC_pred"] = predict_single_target(model_name, X_train, auc_train, X_target)
            rows.append(base)
    pred = pd.concat(rows, ignore_index=True)
    for target in ["Q6", "AUC"]:
        pred[f"{target}_obs_rank_desc"] = pred.groupby(["dataset", "model"])[f"{target}_obs"].rank(ascending=False, method="min")
        pred[f"{target}_pred_rank_desc"] = pred.groupby(["dataset", "model"])[f"{target}_pred"].rank(ascending=False, method="min")
    keep = [
        "model",
        "dataset",
        "record_id",
        "record_index",
        "run_no",
        "C1",
        "C2",
        "C3",
        "C1n",
        "C2n",
        "C3n",
        "Q6_obs",
        "Q6_pred",
        "Q6_obs_rank_desc",
        "Q6_pred_rank_desc",
        "AUC_obs",
        "AUC_pred",
        "AUC_obs_rank_desc",
        "AUC_pred_rank_desc",
    ]
    return pred[keep]


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, dataset), g in pred.groupby(["model", "dataset"], sort=False):
        labels = g["run_no"].astype(str).tolist()
        row: dict[str, Any] = {"model": model, "dataset": dataset, "n_formulations": int(len(g))}
        for prefix, obs_col, pred_col in [("Q6", "Q6_obs", "Q6_pred"), ("AUC", "AUC_obs", "AUC_pred")]:
            y_true = g[obs_col].to_numpy(float)
            y_pred = g[pred_col].to_numpy(float)
            reg = regression_metrics(y_true, y_pred)
            rank = rank_metrics(y_true, y_pred, labels)
            row[f"RMSE_{prefix}"] = reg["RMSE"]
            row[f"MAE_{prefix}"] = reg["MAE"]
            row[f"R2_{prefix}"] = reg["R2"]
            row[f"MSE_{prefix}"] = reg["MSE"]
            if prefix == "Q6":
                row["MSE_Q6_normalized_by_q_scale"] = reg["MSE"] / (Q_SCALE**2)
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


def add_identity(ax: plt.Axes, y: np.ndarray, yhat: np.ndarray) -> None:
    vals = np.concatenate([y[np.isfinite(y)], yhat[np.isfinite(yhat)]])
    lo = float(np.nanmin(vals)) if vals.size else 0.0
    hi = float(np.nanmax(vals)) if vals.size else 1.0
    pad = 0.05 * max(1.0, hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def make_parity(pred: pd.DataFrame, target: str, path: Path) -> None:
    test = pred[pred["dataset"] == "test"]
    plt.figure(figsize=(6.8, 5.8))
    ax = plt.gca()
    for model, g in test.groupby("model", sort=False):
        ax.scatter(g[f"{target}_obs"], g[f"{target}_pred"], s=42, alpha=0.75, label=model)
    add_identity(ax, test[f"{target}_obs"].to_numpy(float), test[f"{target}_pred"].to_numpy(float))
    ax.set_xlabel(f"Observed {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_ranking_plot(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["dataset"] == "test"].copy()
    test = test.sort_values("RMSE_Q6", ascending=True)
    plt.figure(figsize=(8.2, 4.8))
    ax = plt.gca()
    ax.barh(test["model"], test["RMSE_Q6"], color="#4C78A8")
    ax.set_xlabel("Q6 RMSE on 6 held-out formulations")
    ax.invert_yaxis()
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


def write_report(config: dict[str, Any], metrics: pd.DataFrame) -> None:
    test_metrics = metrics[metrics["dataset"] == "test"].sort_values("RMSE_Q6")
    text: list[str] = []
    text.append("# Static Endpoint Baselines, 24 Train + 6 Test")
    text.append("")
    text.append("## Required Context")
    text.append(f"- q_scale used: `{Q_SCALE}`.")
    text.append("- Baseline models were trained on raw physical endpoint targets: `Q6_obs` and `AUC_obs`.")
    text.append("- The selected artifact config was not read for q_scale.")
    text.append("- `MSE_Q6_normalized_by_q_scale` was calculated as `MSE_Q6 / q_scale^2`.")
    text.append("- Training formulations: first 24 rows from `Formulas-train` and `Release-train`.")
    text.append("- Test formulations: all 6 rows from `Formulas-test` and `Release-test`.")
    text.append(f"- Train Run No values: `{config['run_no']['train_first_24']}`.")
    text.append(f"- Excluded original training Run No values: `{config['run_no']['excluded_original_train_6']}`.")
    text.append(f"- Test Run No values: `{config['run_no']['test_6']}`.")
    text.append(f"- Train/test leakage detected: `{config['leak_checks']['any_train_test_leakage']}`.")
    text.append("")
    text.append("## Models")
    for model_name in MODELS:
        text.append(f"- {model_name}")
    text.append("")
    text.append("## Test Endpoint Metrics")
    cols = [
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
        "Kendall_AUC",
        "pairwise_accuracy_AUC",
        "top1_hit_AUC",
        "top2_hit_AUC",
    ]
    text.extend(md_table(test_metrics, cols))
    text.append("")
    text.append("## Top-k Formulation IDs")
    for _, row in test_metrics.iterrows():
        text.append(
            f"- `{row['model']}`: Q6 true_top1={row['true_top1_Q6']}, pred_top1={row['pred_top1_Q6']}, "
            f"true_top2={row['true_top2_Q6']}, pred_top2={row['pred_top2_Q6']}; "
            f"AUC true_top1={row['true_top1_AUC']}, pred_top1={row['pred_top1_AUC']}."
        )
    text.append("")
    text.append("## Outputs")
    for path in [
        RESULTS_DIR / "static_endpoint_baseline_metrics_24train6test.csv",
        RESULTS_DIR / "static_endpoint_baseline_predictions_24train6test.csv",
        FIGURES_DIR / "static_q6_parity_24train6test.png",
        FIGURES_DIR / "static_auc_parity_24train6test.png",
        FIGURES_DIR / "static_q6_ranking_barplot_24train6test.png",
    ]:
        text.append(f"- `{rel(path)}`")
    (REPORT_DIR / "01_static_endpoint_baselines_24train6test_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = read_config()
    train = pd.read_csv(DATA_DIR / "canonical_endpoint_train_24.csv")
    test = pd.read_csv(DATA_DIR / "canonical_endpoint_test_6.csv")

    pred = build_predictions(train, test)
    metrics = metrics_table(pred)

    pred.to_csv(RESULTS_DIR / "static_endpoint_baseline_predictions_24train6test.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "static_endpoint_baseline_metrics_24train6test.csv", index=False)
    make_parity(pred, "Q6", FIGURES_DIR / "static_q6_parity_24train6test.png")
    make_parity(pred, "AUC", FIGURES_DIR / "static_auc_parity_24train6test.png")
    make_ranking_plot(metrics, FIGURES_DIR / "static_q6_ranking_barplot_24train6test.png")
    write_report(config, metrics)

    print("[OK] Static endpoint baselines complete.")
    print(f"[OK] Wrote {rel(RESULTS_DIR / 'static_endpoint_baseline_metrics_24train6test.csv')}")


if __name__ == "__main__":
    main()
