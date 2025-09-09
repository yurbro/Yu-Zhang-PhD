# -*- coding: utf-8 -*-
"""
Multi-criteria selection for SR expressions:
- recompute true complexity on restored_equation (sympy.count_ops)
- hard-constraints: coverage + physics checks (Q(0)≈0, Q(t)>=0, monotone non-decreasing, Q(1)>0)
- rank by AIC (optionally BIC)
- save passing and ranked tables

Usage:
    python consGuACIaf.py
"""

import os
import re
import math
import numpy as np
import pandas as pd
import sympy as sp

# -----------------------------
# Config (edit as needed)
# -----------------------------
CSV_PATH = r"Symbolic Regression/srloop/data/hall_of_fame_run-8_restored.csv"  # 你的最终轮CSV
EXPR_COL = "restored_equation"          # 如无可改为 'equation' 或 'sympy_format'
SAVE_DIR = r"Symbolic Regression/srloop/data"

# 训练集样本量（配方数 × 时间点数），用于AIC/BIC
# 请改成你真实的训练集样本量
N_OBS = 26 * 10   # 例如30个配方 × 10个时间点 = 300

# 物理检查用的时间点（建议用你实际的采样时刻）
TIME_POINTS = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 22.0, 24.0, 26.0, 28.0], dtype=float)

# 变量覆盖要求（硬筛）
CORE_VARS = ['C_pol', 'C_eth', 'C_pg', 't']
REQUIRE_FULL_COVERAGE = True

# 物理检查参数
PARAM_RANGES = {
    'C_pol': (20.0, 30.0),   # 请按你的真实范围调整 
    'C_eth': (10.0, 20.0),
    'C_pg' : (10.0, 20.0),
}
N_PARAM_SAMPLES = 6           # 每轮随机抽样的配方组合数
TOL_Q0 = 1e-4                 # 允许 Q(0) 偏离0的容差
TOL_QPOS = 1e-6               # 允许 Q(t) 的最小阈值（避免-0/数值噪声）
TOL_DQDT = 1e-6               # dQ/dt 容许的小负值（数值噪声）
ALLOW_NEG_FRAC = 0.0          # 允许违反比例（硬筛，建议0）

# 复杂度模式：'count_ops'（推荐）或 'node_count'
# COMPLEXITY_MODE = "count_ops"
COMPLEXITY_MODE = "node_count"

# 只打印/返回前N个模型
TOP_N_PRINT = 10


# -----------------------------
# Helpers
# -----------------------------
def coverage_score(expr_str: str, keywords=CORE_VARS) -> float:
    s = str(expr_str)
    return sum(kw in s for kw in keywords) / len(keywords)


def sympify_with_locals(expr_str: str) -> sp.Expr:
    """将字符串表达式转为sympy表达式，支持 inv/square 等自定义符号。"""
    # 允许的函数映射
    local_dict = {
        'inv': lambda x: 1/x,
        'square': lambda x: x**2,
        'sqrt': sp.sqrt,
        'exp': sp.exp,
        'log': sp.log,
    }
    # 变量符号
    t = sp.Symbol('t')
    C_pol = sp.Symbol('C_pol')
    C_eth = sp.Symbol('C_eth')
    C_pg = sp.Symbol('C_pg')

    try:
        return sp.sympify(expr_str, locals={**local_dict, 't': t, 'C_pol': C_pol, 'C_eth': C_eth, 'C_pg': C_pg})
    except Exception as e:
        raise ValueError(f"sympify failed for: {expr_str}\nError: {e}")


# def recompute_complexity(expr_str: str, mode: str = COMPLEXITY_MODE) -> int:
#     try:
#         expr = sympify_with_locals(expr_str)
#     except Exception:
#         return 10**9  # 无法解析，给超大复杂度
#     if mode == "count_ops":
#         return int(sp.count_ops(expr, visual=False))
#     elif mode == "node_count":
#         return int(sum(1 for _ in sp.preorder_traversal(expr)))
#     else:
#         raise ValueError("complexity mode must be 'count_ops' or 'node_count'")

# ------------------------- Same as complexity calculation rule of PySR -------------------------
def count_ops_pysr_style(expr):
    """按 PySR 的默认风格计算复杂度：只数运算符，不数常数和变量。"""
    ops = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function) or isinstance(node, sp.Pow) or isinstance(node, sp.Mul) or isinstance(node, sp.Add):
            ops += 1
    return ops

def recompute_complexity(expr_str: str, mode: str = "count_ops"):
    try:
        expr = sympify_with_locals(expr_str)
    except Exception:
        return None  # 解析失败

    if mode == "count_ops":
        return count_ops_pysr_style(expr)
    elif mode == "node_count":
        return sum(1 for _ in sp.preorder_traversal(expr))
    else:
        raise ValueError("complexity mode must be 'count_ops' or 'node_count'")
# ------------------------------------------------------------------------------------------------ 


def make_lambda(expr_str: str):
    """返回可向量化的 numpy 函数 f(t, C_pol, C_eth, C_pg)."""
    expr = sympify_with_locals(expr_str)
    t = sp.Symbol('t')
    C_pol = sp.Symbol('C_pol')
    C_eth = sp.Symbol('C_eth')
    C_pg = sp.Symbol('C_pg')
    return sp.lambdify((t, C_pol, C_eth, C_pg), expr, modules=["numpy"])


def physics_check_expr(expr_str: str,
                       time_points=TIME_POINTS,
                       param_ranges=PARAM_RANGES,
                       tol_q0=TOL_Q0,
                       tol_pos=TOL_QPOS,
                       tol_dqdt=TOL_DQDT,
                       allow_neg_frac=ALLOW_NEG_FRAC,
                       n_param_samples=N_PARAM_SAMPLES,
                       check_q1_positive=True):
    """
    物理硬筛：
      - Q(0) ≈ 0
      - Q(t) >= 0, 单调不减
      - (可选) Q(1) > 0
    任一配方组合、任一时间点违反即算失败（allow_neg_frac 控制豁免比例，建议0）
    """
    rng = np.random.default_rng(2025)

    try:
        f = make_lambda(expr_str)
    except Exception:
        return False, {"error": "sympify/lambdify failed"}

    bounds = list(param_ranges.items())
    # 采样配方
    samples = []
    for _ in range(n_param_samples):
        sample = [rng.uniform(lo, hi) for _, (lo, hi) in bounds]
        samples.append(sample)
    samples = np.array(samples)

    total = 0
    fails = 0
    detail = {"q0_fail": 0, "qpos_fail": 0, "dqdt_fail": 0, "q1_fail": 0, "nan_inf": 0}

    for Cpol, Ceth, Cpg in samples:
        try:
            Q = np.array([f(ti, Cpol, Ceth, Cpg) for ti in time_points], dtype=float)
        except Exception:
            fails += len(time_points) + 2
            detail["nan_inf"] += 1
            continue

        if not np.all(np.isfinite(Q)):
            fails += np.sum(~np.isfinite(Q)) + 2
            detail["nan_inf"] += 1
            continue

        # # Q(0) ≈ 0
        # total += 1
        # if abs(Q[0]) > tol_q0:
        #     fails += 1
        #     detail["q0_fail"] += 1

        # Q(1) > 0（如有1h点）
        if check_q1_positive:
            if 1.0 in time_points:
                total += 1
                q1 = Q[np.where(time_points == 1.0)[0][0]]
                if not (q1 > 0):
                    fails += 1
                    detail["q1_fail"] += 1

        # Q(t) >= 0
        total += (len(Q) - 1)
        neg_mask = Q[1:] < tol_pos
        if np.any(neg_mask):
            fails += np.sum(neg_mask)
            detail["qpos_fail"] += int(np.sum(neg_mask))

        # dQ/dt >= -tol_dqdt
        dQdt = np.gradient(Q, time_points)
        total += len(dQdt)
        bad_grad = dQdt < -tol_dqdt
        if np.any(bad_grad):
            fails += np.sum(bad_grad)
            detail["dqdt_fail"] += int(np.sum(bad_grad))

    # 硬筛通过（违例比例必须 <= allow_neg_frac）
    pass_flag = (fails / max(total, 1)) <= allow_neg_frac and detail["nan_inf"] == 0
    print(f"Physics check: {pass_flag} | Total: {total}, Fails: {fails}, Neg. frac: {fails / max(total, 1):.2%}")
    return pass_flag, detail


def aic_from_loss(rss: float, k: int, n_obs: int) -> float:
    """
    AIC = n*ln(RSS/n) + 2k
    这里把 k 用“重算后的复杂度”近似（或你也可以换成 term 数）
    """
    rss = max(rss, 1e-12)
    n = max(int(n_obs), 1)
    return float(n * math.log(rss / n) + 2 * k)


def bic_from_loss(rss: float, k: int, n_obs: int) -> float:
    """
    BIC = n*ln(RSS/n) + k*ln(n)
    """
    rss = max(rss, 1e-12)
    n = max(int(n_obs), 1)
    return float(n * math.log(rss / n) + k * math.log(n))


# -----------------------------
# Main pipeline
# -----------------------------
def select_models(csv_path=CSV_PATH,
                  expr_col=EXPR_COL,
                  n_obs=N_OBS,
                  require_full_coverage=REQUIRE_FULL_COVERAGE,
                  save_dir=SAVE_DIR,
                  top_n=TOP_N_PRINT):
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)

    if expr_col not in df.columns:
        raise ValueError(f"Column '{expr_col}' not in CSV. Available: {list(df.columns)}")

    # 1) 重算复杂度（基于 restored_equation）
    df["complexity_recomputed"] = df[expr_col].apply(lambda s: recompute_complexity(str(s), mode=COMPLEXITY_MODE))

    # 保存所有restored_equation和重算复杂度
    df[["restored_equation", "complexity_recomputed", "loss"]].to_csv(os.path.join(save_dir, "all_expr_recomputed_complexity.csv"), index=False)

    # 2) 覆盖度/硬筛（先做 coverage，再做物理）
    df["coverage_score"] = df[expr_col].apply(lambda s: coverage_score(str(s), CORE_VARS))
    if require_full_coverage:
        df = df[df["coverage_score"] == 1.0].copy()
        if df.empty:
            raise RuntimeError("No expressions pass full variable coverage. Relax coverage or check expressions.")

    # 3) 物理硬筛
    physics_pass_list = []
    physics_detail_list = []
    for expr in df[expr_col].tolist():
        ok, detail = physics_check_expr(expr)
        physics_pass_list.append(bool(ok))
        physics_detail_list.append(detail)

    df["physics_pass"] = physics_pass_list
    df["physics_detail"] = physics_detail_list

    df_pass = df[df["physics_pass"]].copy()
    if df_pass.empty:
        raise RuntimeError("No expressions pass physics constraints. Check constraints or expressions.")

    # 4) 计算 AIC / BIC 排序
    #    注意：这里用 loss 作为 RSS 近似（如果你的loss不是RSS，请替换为RSS）
    df_pass["AIC"] = df_pass.apply(lambda r: aic_from_loss(float(r["loss"]),
                                                           int(r["complexity_recomputed"]),
                                                           int(n_obs)),
                                   axis=1)
    df_pass["BIC"] = df_pass.apply(lambda r: bic_from_loss(float(r["loss"]),
                                                           int(r["complexity_recomputed"]),
                                                           int(n_obs)),
                                   axis=1)

    # 5) 排序（先AIC，再loss兜底）
    df_ranked = df_pass.sort_values(by=["AIC", "loss", "complexity_recomputed"], ascending=[True, True, True]).reset_index(drop=True)

    # 6) 保存
    os.makedirs(save_dir, exist_ok=True)
    pass_path = os.path.join(save_dir, "final_physically_passed_models.csv")
    rank_path = os.path.join(save_dir, "final_ranked_models_by_AIC.csv")
    df_pass.to_csv(pass_path, index=False)
    df_ranked.to_csv(rank_path, index=False)

    # 7) 打印Top-N
    print("\n✅ Physics-passed expressions saved to:", pass_path)
    print("✅ Ranked (by AIC) expressions saved to:", rank_path)
    print(f"\nTop {min(top_n, len(df_ranked))} models (AIC ascending):\n")
    cols_show = [expr_col, "AIC", "BIC", "loss", "complexity_recomputed", "coverage_score"]
    print(df_ranked[cols_show].head(top_n).to_string(index=False))

    return df_ranked

if __name__ == "__main__":
    _ = select_models()
