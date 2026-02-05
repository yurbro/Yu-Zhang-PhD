# -*- coding: utf-8 -*-
"""
replay_sr_artifact.py
---------------------
用训练好的 SR 表达式（best_infix.txt / best_sympy.txt）在【训练集 / 测试集】上做：
- 预测（与训练同构：读取同一 Excel、相同变量缩放、相同积分器设置）
- 可视化（散点图 / 序列对比）
- 指标导出（RMSE、R2、MSE、训练q_scale归一化后的MSE）
- 明细CSV导出（pred_train.csv / pred_test.csv）

依赖：
- main_ivrt_pair.py 的数据构造/默认cfg（保证与训练一致）  # 引用: build_dataset_from_pair/default_cfg/load_pair
- sr_ode_mod.py 的预测与画图工具：predict_dataset/plot_scatter/plot_series/export_predictions_csv/simulate_series

作者：你的 AI 小助手
"""

import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import re  # 新增

# ======== 路径配置（按需改） ========
ART_DIR = Path("Symbolic Differential/SymODE/artifacts/archive/ivrt-pair-251007")
OUT_DIR = Path("Symbolic Differential/SymODE/evaluation")
BEST_INFIX = ART_DIR / "best_infix.txt"   # 训练导出的 infix 表达式
BEST_SYMPY = ART_DIR / "best_sympy.txt"   # 可选：若存在优先用它解析
XLSX_PATH  = Path("Symbolic Differential/SymODE/data/IVRT-Pure.xlsx")  # 训练读的同一 Excel
# ====================================

# ---------- 导入你训练代码里的逻辑（用于保证一致） ----------
# 数据构造/默认配置/加载器：与训练一致
from main_ivrt_pair import load_pair, default_cfg  # 数据读法 & 默认Cfg（含积分参数/硬约束等）  :contentReference[oaicite:2]{index=2}
# 预测/作图/导出：严格沿用 sr_ode_mod 的工具函数
from sr_ode_mod import predict_dataset, plot_scatter, plot_series, export_predictions_csv, simulate_series

# ---------- 数学安全函数：稳定 softplus/log1p/exp/sqrt ----------
import math as _m
def _sat(x, lo=-1e6, hi=1e6):
    try:
        if not _m.isfinite(x): return 0.0
    except Exception:
        return 0.0
    if x > hi: return hi
    if x < lo: return lo
    return x

def softplus(x):
    x = _sat(x)
    if x > 20:  # 与训练时 p_softplus 的数值策略一致
        return _sat(x)
    if x < -20:
        return _sat(_m.exp(x))
    return _sat(_m.log1p(_m.exp(x)))

def log1p(x):
    x = _sat(x)
    if x < -0.999999:  # 与训练的 p_log1p 钳制一致
        x = -0.999999
    return _sat(_m.log1p(x))

def exp(x):
    x = max(-50.0, min(50.0, _sat(x)))
    return _sat(_m.exp(x))

def sqrt(x):
    return _sat(_m.sqrt(max(0.0, _sat(x))))

# ---------- 将 best 表达式转为可调用 f(Q,C1,C2,C3) ----------
def _load_expression_text():
    if BEST_SYMPY.exists():
        return BEST_SYMPY.read_text(encoding="utf-8").strip()
    if BEST_INFIX.exists():
        return BEST_INFIX.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("best_sympy.txt / best_infix.txt 都不存在，请确认训练工件目录。")

def _normalize_expr(expr: str) -> str:
    """
    - 把 to_infix_str 的 log(1+exp(x)) 还原为 softplus(x)
    - 把 log(1+X) 改为 log1p(X)
    - 把 DEAP 占位变量 ARGi 改为 Q/C1/C2/C3
    """
    s = expr.strip()

    # 1) softplus/log1p 规范化（正则更鲁棒）
    s = re.sub(r"log\s*\(\s*1\s*\+\s*exp\s*\(", "softplus(", s)
    s = re.sub(r"log\s*\(\s*1\s*\+\s*", "log1p(", s)

    # 2) 变量名映射：ARG0..3 -> Q/C1/C2/C3（使用词边界避免误替换）
    mapping = {"ARG0":"Q", "ARG1":"C1", "ARG2":"C2", "ARG3":"C3"}
    for k, v in mapping.items():
        s = re.sub(rf"\b{k}\b", v, s)

    return s

EPS = 1e-8  # 放到文件顶部或 build_callable_from_text 上面

def build_callable_from_text(expr_text: str):
    expr = _normalize_expr(expr_text)

    safe_env = {
        "__builtins__": {},
        "softplus": softplus, "log1p": log1p, "exp": exp, "sqrt": sqrt,
        "log": _m.log, "abs": abs, "min": min, "max": max, "pow": pow,
        "EPS": EPS,  # 以防表达式里我们替换成 (C3+EPS) 这种写法
    }
    code = compile(expr, "<best_expr>", "eval")

    def f(Q, C1, C2, C3):
        # 数值护栏：Q≥0；对可能出现在分母的变量加最小模长
        Qs  = float(Q)
        if Qs < 0.0: Qs = 0.0

        C1s = float(C1)
        C2s = float(C2)
        C3s = float(C3)
        # 给 C3 加下界，避免 C1/C3 类项 0 除；如果你有 1/(C1+Q) 一类，也可在这里给 C1s 加个极小值
        if abs(C3s) < EPS:
            C3s = EPS if C3s >= 0 else -EPS

        loc = {
            "Q": Qs, "C1": C1s, "C2": C2s, "C3": C3s,
            # 兜底：表达式里万一还残留 ARGi，也能对上
            "ARG0": Qs, "ARG1": C1s, "ARG2": C2s, "ARG3": C3s,
            "EPS": EPS,
        }
        return float(eval(code, safe_env, loc))

    return f, expr

# ---------- 评价指标（与训练同构口径） ----------
def compute_metrics(ds, preds, q_scale_train: float):
    # 拼接所有点
    y = np.concatenate([rec.Q for rec in ds.records]) if ds.records else np.array([])
    yhat = np.concatenate(preds) if preds else np.array([])
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return {"RMSE": float("nan"), "R2": float("nan"), "MSE": float("nan"), "MSE_normalized_by_train_q_scale": float("nan")}
    y = y[mask]; yhat = yhat[mask]
    mse = float(np.mean((y - yhat)**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res/ss_tot

    # 与 main_ivrt_pair.evaluate 一致的“按训练 q_scale 归一”的 MSE 口径
    qn = max(1.0, float(q_scale_train))
    mse_n = float(np.mean(((y - yhat)/qn)**2))
    return {"RMSE": rmse, "R2": r2, "MSE": mse, "MSE_normalized_by_train_q_scale": mse_n}

def main():
    # 0) 基础检查
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"{XLSX_PATH} 不存在，请把 IVRT-Pure.xlsx 放到该路径，或修改脚本顶部 XLSX_PATH。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 加载表达式并构建可调用 f
    expr_text = _load_expression_text()
    f_best, expr_normed = build_callable_from_text(expr_text)
    (OUT_DIR/"used_expr.txt").write_text(expr_normed + "\n", encoding="utf-8")
    print("[expr] using:", expr_normed[:120] + ("..." if len(expr_normed)>120 else ""))

    # 2) 加载数据（与训练一致：同一 Excel、同一列映射/时间点/预置 t=0, Q=0、同一配方缩放）
    ds_train = load_pair(XLSX_PATH, "Formulas-train", "Release-train")   # 训练构造法  :contentReference[oaicite:4]{index=4}
    ds_test  = load_pair(XLSX_PATH, "Formulas-test",  "Release-test")    # 测试构造法  :contentReference[oaicite:5]{index=5}

    # 3) cfg：默认配置（里面包括积分器、步长、qcap、硬约束等 —— 与训练同构）
    cfg = default_cfg()  # 不训练，仅用于 simulate/predict/plots 的一致参数  :contentReference[oaicite:6]{index=6}

    # 统一“训练集 q_scale”（用于指标归一）
    q_scale_train = getattr(ds_train, "q_scale", 1.0)

    # 4) 预测 & 作图 & 导出（训练集）
    preds_tr = predict_dataset(f_best, ds_train, cfg)   # 同构前向  :contentReference[oaicite:7]{index=7}
    plot_scatter(ds_train, preds_tr, OUT_DIR / "scatter_train.png")
    plot_series(ds_train, preds_tr, OUT_DIR / "series_train.png", max_examples=6)
    export_predictions_csv(ds_train, preds_tr, OUT_DIR / "pred_train.csv")  # 与训练一致的 CSV 列名  :contentReference[oaicite:8]{index=8}
    metrics_tr = compute_metrics(ds_train, preds_tr, q_scale_train)
    (OUT_DIR/"metrics_train.json").write_text(json.dumps(metrics_tr, indent=2), encoding="utf-8")
    print("[train]", metrics_tr)

    # 5) 预测 & 作图 & 导出（测试集）
    preds_te = predict_dataset(f_best, ds_test, cfg)
    plot_scatter(ds_test, preds_te, OUT_DIR / "scatter_test.png")
    plot_series(ds_test, preds_te, OUT_DIR / "series_test.png", max_examples=6)
    export_predictions_csv(ds_test, preds_te, OUT_DIR / "pred_test.csv")
    metrics_te = compute_metrics(ds_test, preds_te, q_scale_train)
    (OUT_DIR/"metrics_test.json").write_text(json.dumps(metrics_te, indent=2), encoding="utf-8")
    print("[test ]", metrics_te)

    # 6) 元数据记录：便于追溯
    meta = {
        "artifact_dir": str(ART_DIR.resolve()),
        "expr_source": "best_sympy.txt" if BEST_SYMPY.exists() else "best_infix.txt",
        "xlsx_path": str(XLSX_PATH.resolve()),
        "q_scale_train": float(q_scale_train),
        "cfg_like_training": True,
        "outputs": {
            "scatter_train.png": str((OUT_DIR/"scatter_train.png").resolve()),
            "series_train.png": str((OUT_DIR/"series_train.png").resolve()),
            "pred_train.csv": str((OUT_DIR/"pred_train.csv").resolve()),
            "metrics_train.json": str((OUT_DIR/"metrics_train.json").resolve()),
            "scatter_test.png": str((OUT_DIR/"scatter_test.png").resolve()),
            "series_test.png": str((OUT_DIR/"series_test.png").resolve()),
            "pred_test.csv": str((OUT_DIR/"pred_test.csv").resolve()),
            "metrics_test.json": str((OUT_DIR/"metrics_test.json").resolve()),
            "used_expr.txt": str((OUT_DIR/"used_expr.txt").resolve()),
        }
    }
    (OUT_DIR/"replay_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] 导出到 {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
