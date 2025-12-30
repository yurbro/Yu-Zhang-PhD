# -*- coding: utf-8 -*-
"""
expr_pred.py
------------
将已训练得到的最佳表达式 dQ/dt 在测试集上做预测、评估并可视化。

使用方法：
1) 修改 EXCEL_PATH 为你的 Excel 路径（含 Formulas-test / Release-test 两表）。
2) 在 BEST_EXPR_STR 里粘贴 main 打印出的最佳表达式（例如：add(mul(C1, log1p(C3)), add(C2, Q))）
   - 或者把它存到 best_expr.txt，设置 LOAD_EXPR_FROM_FILE=True。
3) 运行：python expr_pred.py
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 从你的模块导入（需在同目录或 PYTHONPATH 中）
from sr_ode import (
    Dataset, Record, Config,
    simulate_with_model, PRIMITIVE_REGISTRY
)

import json
from dataclasses import fields


# ========== 0) 路径 & 表达式配置 ==========
EXCEL_PATH = Path("Symbolic Differential") / "SymODE" / "data" / "IVRT-Pure.xlsx"
SHEET_F_TRAIN = "Formulas-train"   # 本脚本不使用训练表，只读测试表；保留变量仅为对齐
SHEET_R_TRAIN = "Release-train"
SHEET_F_TEST  = "Formulas-test-opt2"
SHEET_R_TEST  = "Release-test-opt2"

# 采样时刻（强烈建议显式给出，避免误推断）
TIMES_H: List[float] = [0.0, 0.5, 1, 2, 3, 4, 5, 6]

# 是否从文件读表达式（否则用 BEST_EXPR_STR）
LOAD_EXPR_FROM_FILE = True
BEST_EXPR_FILE = Path("Symbolic Differential/SymODE/artifacts/best_expr.txt")

# 直接把 main 打印出的表达式粘到这：
BEST_EXPR_STR = r"""add(mul(C1, log1p(C3)), add(C2, Q))"""

# 是否画残差图（推荐 True）
PLOT_RESIDUALS = True
SAVE_FIG = True
FIG_DIR = Path("pred_plots")

# ========== 1) 与训练对齐的 cfg（可按需微调，但保持一致性） ==========
def load_cfg_from_json(path: Path, fallback_cfg: Config | None = None) -> tuple[Config, dict]:
    """
    从 artifacts/cfg.json 读取配置并合并到一个 Config 实例里。
    - 只应用 Config dataclass 中存在的字段，忽略未知键（比如 q_scale）。
    - 返回 (cfg, meta)；meta 保留 json 里所有键，便于读取 q_scale 等。
    """
    if fallback_cfg is None:
        cfg = Config()  # 用默认
    else:
        # 拷贝一份以免修改外部对象
        cfg = Config(**{f.name: getattr(fallback_cfg, f.name) for f in fields(Config)})

    meta: dict = {}
    if not path.exists():
        print(f"[WARN] cfg.json not found at {path}. Using default cfg.")
        return cfg, meta

    meta = json.loads(path.read_text(encoding="utf-8"))
    valid_keys = {f.name for f in fields(Config)}
    for k, v in meta.items():
        if k in valid_keys:
            setattr(cfg, k, v)
    return cfg, meta

def make_cfg() -> Config:
    return Config(
        var_names=("Q","C1","C2","C3"),
        must_have=("Q","C1","C2","C3"),
        # 这里列出你训练时允许的原子名（要覆盖表达式中用到的那些）
        primitive_names=("add","sub","mul","div","log1p","exp","sqrt","softplus","tanh","relu","abs","pow2","pow3","min","max"),
        # 即便表达式没用到所有原子，列着也没关系；关键是把用到的都包含进来
        ephemeral_range=(-1.0, 1.0),

        # 数值积分器建议与训练一致
        integrator="rk4", substeps=8, adapt_refine_max=8, dt_floor=1e-6,
        qcap_factor=1.5, clamp_nonneg=True,

        # 其它参数对预测无影响，这里随便保留默认
        alpha_complexity=1e-2, enable_nest_penalty=True, nest_op="exp", nest_weight=1e-3,
        lambda_phys=0.0, lambda_cov=0.0,
        n_jobs=1, seed=13
    )

# ========== 2) 读取测试数据 ==========
def _normalized_map(cols): return {c: str(c).strip().lower() for c in cols}
def _find_col(norm_map, wanted):
    for orig, low in norm_map.items():
        if low == wanted:
            return orig
    raise KeyError(f"Column '{wanted}' not found. Available: {list(norm_map.keys())}")

def _find_like(norm_map, wanted_exact: str, aliases: list[str]) -> str:
    for orig, low in norm_map.items():
        if low == wanted_exact.lower():
            return orig
    for orig, low in norm_map.items():
        if any(alias.lower() in low for alias in aliases):
            return orig
    raise KeyError(f"Column for '{wanted_exact}' not found. Tried aliases: {aliases}")

def _to_float(x):
    # 支持 "1,146.64" 这类带千分位的字符串
    if isinstance(x, str):
        x = x.replace(",", "")
    return float(x)

# === 新的带 std 的测试集读取 ===
def load_test_dataset_with_std(
    excel_path: Path,
    sheet_formulas: str = "Formulas-test-",
    sheet_release_opt: str = "Release-test-opt2",
    times_h: list[float] = [0.5, 1, 2, 3, 4, 5, 6],
    normalize: bool = False,
):
    import pandas as pd
    xls = pd.ExcelFile(excel_path)

    df_f = xls.parse(sheet_formulas)
    df_r = xls.parse(sheet_release_opt)

    nf = _normalized_map(df_f.columns)
    nr = _normalized_map(df_r.columns)

    col_run_f = _find_col(nf, "run no")
    col_run_r = _find_col(nr, "run no")

    # R_t* 列
    r_cols = [c for c in df_r.columns if str(c).strip().lower().startswith("r_t")]
    if not r_cols:
        raise KeyError(f"No 'R_t*' columns found in '{sheet_release_opt}'.")
    # 按编号排序 R_t1, R_t2, ...
    r_cols = sorted(r_cols, key=lambda c: int("".join(ch for ch in str(c) if ch.isdigit()) or "0"))

    times = np.array(times_h, dtype=float)
    if len(times) not in (len(r_cols), len(r_cols)+1):
        raise ValueError(f"times_h length {len(times)} incompatible with {len(r_cols)} R_t columns")

    # 三个配方变量列
    col_c1 = _find_like(nf, "Poloxamer 407", ["poloxamer","p407","poloxamer 407"])
    col_c2 = _find_like(nf, "Ethanol",        ["ethanol","etoh"])
    col_c3 = _find_like(nf, "Propylene glycol", ["propylene glycol","pg"])

    # 将 Formulas-test 按 Run No 索引
    f_indexed = df_f.set_index(col_run_f)

    # 把 Release-test-opt1 变成 {run: (mean, std)}
    runs = {}
    for _, r in df_r.iterrows():
        name = str(r[col_run_r]).strip()
        vals = np.array([_to_float(r[c]) for c in r_cols], dtype=float)
        if name.endswith("-Std"):
            base = name[:-4]  # 去掉 -Std
            runs.setdefault(base, {})["std"] = vals
        else:
            runs.setdefault(name, {})["mean"] = vals

    # 组装 Dataset + 把 std 挂到 record.sigma
    from sr_ode import Dataset, Record  # 避免顶层循环导入
    recs = []
    for base, pack in runs.items():
        if "mean" not in pack: 
            continue
        mean = pack["mean"]
        std  = pack.get("std", None)
        if base not in f_indexed.index:
            # 没有对应配方，跳过
            continue
        frow = f_indexed.loc[base]

        if len(times) == len(r_cols) + 1:
            Q = np.concatenate([[0.0], mean]); Q0 = 0.0
            sigma = np.concatenate([[0.0], std]) if std is not None else None
        else:
            Q = mean; Q0 = 0.0
            sigma = std

        rec = Record(
            t=times,
            Q=Q,
            Q0=Q0,
            vars={
                "C1": float(_to_float(frow[col_c1])),
                "C2": float(_to_float(frow[col_c2])),
                "C3": float(_to_float(frow[col_c3])),
            }
        )
        if sigma is not None:
            # 动态挂一个可选属性，供画图与加权指标使用
            setattr(rec, "sigma", sigma.astype(float))
        recs.append(rec)

    if not recs:
        raise RuntimeError(f"No test records built from '{sheet_formulas}' + '{sheet_release_opt}'.")
    return Dataset(recs, normalize=normalize)

# ========== 3) 将表达式字符串变成可调用 f(Q,C1,C2,C3) ==========
def make_callable_from_expr(expr_str: str, var_names: Tuple[str,...]) -> callable:
    # 从注册表拿到 {原子名: 函数}
    prim_ctx = {name: func for name, (func, arity) in PRIMITIVE_REGISTRY.items()}
    # 构造 lambda 源码
    args_sig = ",".join(var_names)
    code = f"lambda {args_sig}: {expr_str}"
    # 关键：把原子放进 globals（而不是 locals）
    safe_globals = {"__builtins__": {}}
    safe_globals.update(prim_ctx)
    try:
        f = eval(code, safe_globals, {})
    except NameError as e:
        raise NameError(
            f"Unknown name in expression: {e}. "
            f"Make sure the expr only uses primitives in PRIMITIVE_REGISTRY and var_names={var_names}."
        )
    return f

# ========== 4) 评估指标 ==========
def r2_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return r2, rmse

# ========== 5) 可视化：观察 vs 预测 + 残差 ==========
def plot_record(rec, Q_pred: np.ndarray, rec_idx: int,
                plot_residuals: bool = True, save: bool = False):
    t = rec.t
    y_true = rec.Q
    y_pred = Q_pred
    res = y_pred - y_true
    yerr = getattr(rec, "sigma", None)

    if plot_residuals:
        fig, axes = plt.subplots(2, 1, figsize=(6,6), sharex=True,
                                 gridspec_kw={"height_ratios":[3,1]})
        ax1, ax2 = axes

        # 观测：带误差棒（若有）
        if yerr is not None:
            ax1.errorbar(t, y_true, yerr=yerr, fmt="o-", capsize=3, label="Observed Q(t) ±1σ")
        else:
            ax1.plot(t, y_true, "o-", label="Observed Q(t)")
        # 预测
        ax1.plot(t, y_pred, "s-", label="Predicted Q(t)")
        ax1.set_ylabel("Q (μg/cm²)")
        ax1.set_title(f"Test #{rec_idx} — Observed vs Predicted")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.axhline(0, linewidth=1, color="k", alpha=0.6)
        markerline, stemlines, baseline = ax2.stem(t, res)
        plt.setp(stemlines, linewidth=1)
        plt.setp(markerline, markersize=5)
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Residual")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        plt.figure()
        if yerr is not None:
            plt.errorbar(t, y_true, yerr=yerr, fmt="o-", capsize=3, label="Observed Q(t) ±1σ")
        else:
            plt.plot(t, y_true, "o-", label="Observed Q(t)")
        plt.plot(t, y_pred, "s-", label="Predicted Q(t)")
        plt.xlabel("Time (h)"); plt.ylabel("Q (μg/cm²)")
        plt.title(f"Test #{rec_idx} — Observed vs Predicted")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

    # 可固定刻度
    # ax2.set_xticks([0,0.5,1,2,3,4,5,6])

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fn = FIG_DIR / f"test_{rec_idx:03d}.png"
        plt.savefig(fn, dpi=150)
    plt.show()

# ========== 6) 主流程 ==========
def main():
    # 1) 加载测试集
    ds_test = load_test_dataset_with_std(EXCEL_PATH, 
                                         sheet_formulas=SHEET_F_TEST,
                                         sheet_release_opt=SHEET_R_TEST,
                                         times_h=TIMES_H,
                                         normalize=False)

    # 2) cfg（尽量与训练一致，从 artifacts/cfg.json 读取）
    cfg_path = Path("Symbolic Differential/SymODE/artifacts") / "cfg.json"
    default_cfg = make_cfg()  # 若没有 make_cfg()，也可以用 Config() 当默认
    cfg, meta = load_cfg_from_json(cfg_path, fallback_cfg=default_cfg)

    # 可选：打印关键配置以核对
    print("[CFG] var_names:", cfg.var_names)
    print("[CFG] integrator:", cfg.integrator, "| substeps:", cfg.substeps)

    # 3) 拿到表达式字符串
    if LOAD_EXPR_FROM_FILE and BEST_EXPR_FILE.exists():
        expr_str = BEST_EXPR_FILE.read_text(encoding="utf-8").strip()
    else:
        expr_str = BEST_EXPR_STR.strip()
    print("Using expression:\n", expr_str)

    # 4) 构造可调用 f(Q,C1,C2,C3)
    f = make_callable_from_expr(expr_str, cfg.var_names)

    # 5) 在测试集上逐条预测与评估
    r2_list_o, rmse_list_o = [], []
    r2_list_n, rmse_list_n = [], []
    all_true_o, all_pred_o = [], []
    all_true_n, all_pred_n = [], []

    train_q_scale = meta.get("q_scale", None)
    def scale_by_train(x: np.ndarray) -> np.ndarray:
        if train_q_scale is None or float(train_q_scale) <= 0:
            # 回退：用测试集自己的缩放（如果 Dataset 提供了）；否则不缩放
            return getattr(ds_test, "scaled", lambda y: y)(x)
        return x / float(train_q_scale)

    for i, rec in enumerate(ds_test.records):
        Q_pred = simulate_with_model(f, rec, cfg)
        if not np.isfinite(Q_pred).all():
            print(f"[Test #{i}] simulation failed (NaN/Inf)")
            continue

        # 原始单位
        y_true_n = scale_by_train(rec.Q)
        y_pred_n = scale_by_train(Q_pred)

        r2_o, rmse_o = r2_rmse(y_true_n, y_pred_n)
        r2_list_o.append(r2_o); rmse_list_o.append(rmse_o)
        all_true_o.append(y_true_n); all_pred_o.append(y_pred_n)

        # 归一化（用本测试集的最大值缩放，仅做参考）
        y_true_n = y_true_n / max(1.0, ds_test.q_scale)
        y_pred_n = y_pred_n / max(1.0, ds_test.q_scale)
        r2_n, rmse_n = r2_rmse(y_true_n, y_pred_n)
        r2_list_n.append(r2_n); rmse_list_n.append(rmse_n)
        all_true_n.append(y_true_n); all_pred_n.append(y_pred_n)

        # 画图
        plot_record(rec, Q_pred, rec_idx=i, plot_residuals=PLOT_RESIDUALS, save=SAVE_FIG)

        print(f"[Test #{i}] R2(orig)={r2_o:.4f} | RMSE(orig)={rmse_o:.4f} | "
              f"R2(norm)={r2_n:.4f} | RMSE(norm)={rmse_n:.4f}")

    # 6) 汇总
    if r2_list_o:
        all_true_o = np.concatenate(all_true_o); all_pred_o = np.concatenate(all_pred_o)
        R2_all_o, RMSE_all_o = r2_rmse(all_true_o, all_pred_o)

        all_true_n = np.concatenate(all_true_n); all_pred_n = np.concatenate(all_pred_n)
        R2_all_n, RMSE_all_n = r2_rmse(all_true_n, all_pred_n)

        print("\n===== Test set summary =====")
        print(f"Per-record (orig): R2 mean={np.nanmean(r2_list_o):.4f}, RMSE mean={np.nanmean(rmse_list_o):.4f}")
        print(f"Overall   (orig): R2={R2_all_o:.4f}, RMSE={RMSE_all_o:.4f}")
        print(f"Per-record (norm): R2 mean={np.nanmean(r2_list_n):.4f}, RMSE mean={np.nanmean(rmse_list_n):.4f}")
        print(f"Overall   (norm): R2={R2_all_n:.4f}, RMSE={RMSE_all_n:.4f}")
    else:
        print("No valid predictions to summarize.")

if __name__ == "__main__":
    main()
