# -*- coding: utf-8 -*-
"""
main_ivrt.py  —  最简主程序（直接设 cfg，直接读 Excel）

数据文件：Symbolic Differential/SymODE/data/IVRT-Pure.xlsx
工作表：
  - Formulas-train / Release-train
  - Formulas-test  / Release-test
含义：
  - Formulas-*:   Run No, Poloxamer 407, Ethanol, Propylene glycol
  - Release-*:    Run No, R_t1..R_tK（累计 Q，单位 μg/cm²）

变量映射：
  C1 = Poloxamer 407
  C2 = Ethanol
  C3 = Propylene glycol

训练：在训练集上做 SR-ODE
评估：在测试集上预测，画出 Q(t)：真实 vs 预测
"""

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sr_ode import (
    Dataset, Record, Config,
    train_symbolic_ode, compile_individual, simulate_with_model
)

# ===== 保存最优表达式 + 元数据 =====
import json
from deap import gp

# ============== 1) Excel -> Dataset（支持 train/test） ==============
def _normalized_map(cols) -> Dict[str, str]:
    return {c: str(c).strip().lower() for c in cols}

def _find_col(norm_map: Dict[str, str], wanted: str) -> str:
    for orig, low in norm_map.items():
        if low == wanted:
            return orig
    raise KeyError(f"Column '{wanted}' not found. Available: {list(norm_map.keys())}")

def _find_like(norm_map: Dict[str, str], wanted_exact: str, aliases: List[str]) -> str:
    # 优先精确匹配
    for orig, low in norm_map.items():
        if low == wanted_exact.lower():
            return orig
    # 模糊匹配
    for orig, low in norm_map.items():
        if any(alias.lower() in low for alias in aliases):
            return orig
    raise KeyError(f"Column for '{wanted_exact}' not found. Tried aliases: {aliases}")

def _build_dataset_from_two_sheets(
    xls: pd.ExcelFile,
    sheet_formulas: str,
    sheet_release: str,
    map_c1: str = "Poloxamer 407",
    map_c2: str = "Ethanol",
    map_c3: str = "Propylene glycol",
    times_h: Optional[List[float]] = None,
    normalize: bool = True,
) -> Dataset:
    df_f = xls.parse(sheet_formulas)
    df_r = xls.parse(sheet_release)

    nf = _normalized_map(df_f.columns)
    nr = _normalized_map(df_r.columns)

    col_run_f = _find_col(nf, "run no")
    col_run_r = _find_col(nr, "run no")

    r_cols = [c for c in df_r.columns if str(c).strip().lower().startswith("r_t")]
    if not r_cols:
        raise KeyError(f"No 'R_t*' columns found in sheet '{sheet_release}'.")
    # 按编号排序 R_t1, R_t2, ...
    r_cols = sorted(r_cols, key=lambda c: int("".join(ch for ch in str(c) if ch.isdigit()) or "0"))

    # 时间轴：若未指定，默认 0 + 等步 0.5 的 K 个点（与列数匹配）
    if times_h is None:
        t_rest = [0.5 * (i + 1) for i in range(len(r_cols))]
        times = np.array([0.0] + t_rest, dtype=float)
    else:
        times = np.array(times_h, dtype=float)
        if len(times) not in (len(r_cols), len(r_cols) + 1):
            raise ValueError(
                f"times_h length {len(times)} incompatible with {len(r_cols)} R_t columns "
                f"(or +1 if including t=0)."
            )

    col_c1 = _find_like(nf, map_c1, aliases=["poloxamer", "p407", "poloxamer 407"])
    col_c2 = _find_like(nf, map_c2, aliases=["ethanol", "etoh"])
    col_c3 = _find_like(nf, map_c3, aliases=["propylene glycol", "pg"])

    f_indexed = df_f.set_index(col_run_f)

    recs: List[Record] = []
    for _, row in df_r.iterrows():
        run = row[col_run_r]
        if run not in f_indexed.index:
            continue
        frow = f_indexed.loc[run]

        q_vals = np.array([row[c] for c in r_cols], dtype=float)
        if len(times) == len(r_cols) + 1:
            Q = np.concatenate([[0.0], q_vals])
            Q0 = 0.0
        else:
            Q = q_vals
            Q0 = 0.0

        recs.append(Record(
            t=times,
            Q=Q,
            Q0=Q0,
            vars={
                "C1": float(frow[col_c1]),  # Poloxamer 407
                "C2": float(frow[col_c2]),  # Ethanol
                "C3": float(frow[col_c3]),  # Propylene glycol
            }
        ))

    if not recs:
        raise RuntimeError(f"No records built from '{sheet_formulas}' + '{sheet_release}'.")
    return Dataset(recs, normalize=normalize)

def load_train_test_dataset(
    excel_path: str | Path,
    normalize: bool = True,
    times_h: Optional[List[float]] = None
) -> tuple[Dataset, Dataset]:
    """
    从 IVRT Excel 加载 train/test 两套 Dataset。
    期望工作表：
      - Formulas-train / Release-train
      - Formulas-test  / Release-test
    """
    p = Path(excel_path)
    xls = pd.ExcelFile(p)

    train_ds = _build_dataset_from_two_sheets(
        xls, "Formulas-train", "Release-train", times_h=times_h, normalize=normalize
    )
    test_ds  = _build_dataset_from_two_sheets(
        xls, "Formulas-test",  "Release-test",  times_h=times_h, normalize=normalize
    )
    return train_ds, test_ds

# ============== 2) 配置 cfg（直接在这里修改） ==============
def make_cfg() -> Config:
    return Config(
        # 变量：Q + 三个配方变量
        var_names=("Q", "C1", "C2", "C3"),
        must_have=("Q", "C1", "C2", "C3"),

        # 原子集合：稳妥的一组；如需更稳可把 "exp" 换成 "softplus"
        primitive_names=("add","sub","mul","div",
                         "log1p","exp","sqrt", "pow2"),
        ephemeral_range=(-2.0, 2.0),

        # 遗传参数
        pop_size=1000,
        ngen=200,
        cxpb=0.6,
        mutpb=0.5,
        tournsize=4,
        tree_len_max=25,
        init_depth_min=1,
        init_depth_max=4,

        # 积分器与数值稳定
        integrator="rk4",        # 可选: "euler"/"rk2"/"rk4"；若 sr_ode 已接入 solve_ivp，也可 "dopri5"
        substeps=8,
        adapt_refine_max=8,
        dt_floor=1e-6,
        qcap_factor=1.5,
        clamp_nonneg=True,

        # 正则/惩罚
        alpha_complexity=5e-4,  # 复杂度惩罚
        enable_nest_penalty=True, nest_op="exp", nest_weight=1e-3,
        lambda_phys=5e-3,   # 物理约束：Q(t) 单调非减 df/dQ <= 0
        lambda_cov=0.0,    # 允许 cov 参与正则化, 鼓励多用变量

        # 并行 / 随机
        n_jobs=None,   # None -> 自动用 CPU-1（Windows 下注意 __main__ 保护）
        seed=13,
    )

# ============== 3) 训练、测试、画图 ==============
def main():
    # 1) 路径与时间轴（如你的 Release_t 列对应时刻不是默认的 0.5..6，可在 times_h 里明确给出）
    excel_path = Path("Symbolic Differential") / "SymODE" / "data" / "IVRT-Pure.xlsx"
    times_h = [0.0, 0.5, 1, 2, 3, 4, 5, 6]     # 例如明确指定: [0.0, 0.5, 1, 2, 3, 4, 5, 6]
    normalize = True

    # 2) 加载训练/测试数据
    ds_train, ds_test = load_train_test_dataset(excel_path, normalize=normalize, times_h=times_h)
    print(f"Loaded: train {len(ds_train.records)} records, test {len(ds_test.records)} records.")

    # 3) 配置
    cfg = make_cfg()

    # 4) 训练
    hof, log, best, pset = train_symbolic_ode(ds_train, cfg)
    print("\n===== Training finished =====")
    print("Best individual:\n", best)

    # ===== 保存最优表达式 + 元数据 =====
    out_dir = Path("Symbolic Differential/SymODE/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 最优表达式（expr_pred.py 就读这个）
    best_expr_str = str(best)            # 例如：add(mul(C1, log1p(C3)), add(C2, Q))
    (out_dir / "best_expr.txt").write_text(best_expr_str + "\n", encoding="utf-8")
    print(f"[Saved] {out_dir/'best_expr.txt'}")

    # 2) 配置元数据（方便预测阶段保持一致）
    meta = {
        "var_names": list(cfg.var_names),
        "primitive_names": list(cfg.primitive_names),
        "integrator": cfg.integrator,
        "substeps": cfg.substeps,
        "adapt_refine_max": cfg.adapt_refine_max,
        "dt_floor": cfg.dt_floor,
        "qcap_factor": cfg.qcap_factor,
        "clamp_nonneg": cfg.clamp_nonneg,
        "alpha_complexity": cfg.alpha_complexity,
        "nest_op": cfg.nest_op,
        "nest_weight": cfg.nest_weight,
        "lambda_phys": cfg.lambda_phys,
        "lambda_cov": cfg.lambda_cov,
        # 如果 Dataset 暴露缩放参数，顺手存一下（expr_pred 可用得到相同归一化）
        "q_scale": getattr(ds_train, "q_scale", None),
        "normalized": getattr(ds_train, "normalized", True),
    }
    (out_dir / "cfg.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Saved] {out_dir/'cfg.json'}")

    # 3) 导出 HOF 前若干个表达式（便于回看/对比）
    import csv
    top_k = min(10, len(hof))
    with open(out_dir / "hof_top.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "expr", "fitness", "size"])
        for i in range(top_k):
            ind = hof[i]
            w.writerow([i+1, str(ind), float(ind.fitness.values[0]), len(ind)])
    print(f"[Saved] {out_dir/'hof_top.csv'}")

    # 5) 编译最优模型
    f = compile_individual(best, pset)

    # --- 评估辅助：R2 / RMSE ---
    def r2_rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return r2, rmse

    # 6) 在测试集上预测并计算指标；同时画图
    test_mse_list = []

    # 逐样本的度量（normalized 与 original）
    r2_list_n, rmse_list_n = [], []
    r2_list_o, rmse_list_o = [], []

    # 汇总（把所有测试样本的点拼在一起计算“整体”）
    all_true_n, all_pred_n = [], []
    all_true_o, all_pred_o = [], []

    for i, rec in enumerate(ds_test.records):
        Q_pred = simulate_with_model(f, rec, cfg)
        if not np.isfinite(Q_pred).all():
            print(f"[Test #{i}] simulation failed (NaN/Inf)")
            continue

        # ---- 归一化尺度（与训练目标一致）----
        y_true_n = ds_test.scaled(rec.Q)
        y_pred_n = ds_test.scaled(Q_pred)
        mse_i = float(np.mean((y_pred_n - y_true_n) ** 2))
        r2_i_n, rmse_i_n = r2_rmse(y_true_n, y_pred_n)
        test_mse_list.append(mse_i)
        r2_list_n.append(r2_i_n)
        rmse_list_n.append(rmse_i_n)
        all_true_n.append(y_true_n)
        all_pred_n.append(y_pred_n)

        # ---- 原始单位（μg/cm²）----
        y_true_o = rec.Q
        y_pred_o = Q_pred
        r2_i_o, rmse_i_o = r2_rmse(y_true_o, y_pred_o)
        r2_list_o.append(r2_i_o)
        rmse_list_o.append(rmse_i_o)

        print(f"[Test #{i}] MSE(n)={mse_i:.6f} | R2(n)={r2_i_n:.4f} | RMSE(n)={rmse_i_n:.4f} | "
            f"R2(orig)={r2_i_o:.4f} | RMSE(orig)={rmse_i_o:.4f}")

        # --- 画图：真实 vs 预测（原始单位） ---
        plt.figure()
        plt.plot(rec.t, y_true_o, marker="o", label="Observed Q(t)")
        plt.plot(rec.t, y_pred_o, marker="s", label="Predicted Q(t)")
        plt.xlabel("Time (h)")
        plt.ylabel("Cumulative Q (μg/cm²)")
        plt.title(f"Test #{i} — Observed vs Predicted")
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Scatter pred vs true
        plt.figure()
        plt.scatter(y_true_o, y_pred_o, alpha=0.7)
        min_val = min(y_true_o.min(), y_pred_o.min())
        max_val = max(y_true_o.max(), y_pred_o.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label=f"y=x (perfect) \n R2={r2_i_o:.3f} RMSE={rmse_i_o:.3f}")
        plt.xlabel("Observed Q (μg/cm²)")
        plt.ylabel("Predicted Q (μg/cm²)")
        plt.title(f"Test #{i} — Predicted vs Observed")
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        # plt.show()

    # ------ 汇总输出 ------
    if len(all_true_n) > 0:
        all_true_n = np.concatenate(all_true_n)
        all_pred_n = np.concatenate(all_pred_n)
        R2_all_n, RMSE_all_n = r2_rmse(all_true_n, all_pred_n)

        # 原始单位的“整体”指标
        all_true_o = np.concatenate([ds_test.descale(x) for x in [all_true_n]])  # 这里只是演示，直接用 all_true_o 更简洁
        all_pred_o = np.concatenate([ds_test.descale(x) for x in [all_pred_n]])
        # 更直接的做法（避免上面两行的绕路）：
        # all_true_o = np.concatenate([rec.Q for rec in ds_test.records])
        # all_pred_o = np.concatenate([simulate_with_model(f, rec, cfg) for rec in ds_test.records])
        R2_all_o, RMSE_all_o = r2_rmse(all_true_o, all_pred_o)

        print("\n===== Test set summary =====")
        print(f"Per-record (normalized):   R2 mean={np.nanmean(r2_list_n):.4f}, "
            f"RMSE mean={np.nanmean(rmse_list_n):.4f},  MSE mean={np.mean(test_mse_list):.6f}")
        print(f"Per-record (original):     R2 mean={np.nanmean(r2_list_o):.4f}, "
            f"RMSE mean={np.nanmean(rmse_list_o):.4f}")
        print(f"Overall   (normalized):    R2={R2_all_n:.4f}, RMSE={RMSE_all_n:.4f}")
        print(f"Overall   (original unit): R2={R2_all_o:.4f}, RMSE={RMSE_all_o:.4f}")
    else:
        print("No valid test predictions to summarize.")

if __name__ == "__main__":
    main()
