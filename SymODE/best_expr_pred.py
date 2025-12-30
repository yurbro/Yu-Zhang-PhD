# -*- coding: utf-8 -*-
"""
评估 SR 最佳模型（dQ/dt = f(Q, C1, C2, C3)）在 IVRT 训练/测试上的预测性能
- 读取: best_expr.txt（中缀表达式）
- 读取: IVRT-Pure.xlsx 里的 Release-train/Release-test + Formulas-train/Formulas-test
- 合并: 若 Release-* 内含三列(C1/C2/C3)则直接用，否则从 Formulas-* 合并
- 积分: RK4 仿照主训练 simulate_series（带限幅、qcap、非负、细分）
- 指标: RMSE、R²
- 图: 观测 vs 预测散点、抽样时间序列对比（各保存一张图）

作者：for Yu
"""

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import ast
import warnings
import builtins as _builtins  # 用于安全地给 eval 提供最小 builtins

# ========= 路径与配置 =========
BEST_EXPR_FILE = r"Symbolic Differential/SymODE/artifacts/ivrt_pair/best_infix.txt"
XLSX_PATH      = r"Symbolic Differential/SymODE/data/IVRT-Pure.xlsx"
PLOT_SAVE_PATH = r"Symbolic Differential/SymODE/visualisation"

SHEET_REL_TRAIN = "Release-train"
SHEET_REL_TEST  = "Release-test"
SHEET_FOR_TRAIN = "Formulas-train"   # 若不存在，程序会自动忽略
SHEET_FOR_TEST  = "Formulas-test"

JOIN_KEY = None

# ========= 固定时间轴与配方缩放 =========
R_COLS    = ["R_t1","R_t2","R_t3","R_t4","R_t5","R_t6","R_t7"]
TIMES_HRS = np.array([0.5, 1, 2, 3, 4, 5, 6], dtype=float)

C_BOUNDS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}
def scale_fixed(x, lo, hi): return (x - lo) / (hi - lo)

# ========= 表达式读取与编译 =========
def read_best_expr(path: str) -> str:
    assert os.path.exists(path), f"best_expr 文件不存在: {path}"
    s = open(path, "r", encoding="utf-8").read().strip()
    m = re.search(r"dQ/dt\s*=\s*(.*)$", s)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1]

def _clip(x, lo=-50.0, hi=50.0):
    return np.clip(x, lo, hi)

def softplus(x):
    return np.log1p(np.exp(_clip(x)))

def log1p_stable(x):  # 保留名字 log1p 给表达式用
    return np.log1p(x)

def exp_stable(x):
    return np.exp(_clip(x))

def log_stable(x):
    return np.log(np.maximum(x, 1e-16))

def pow2(x):
    return np.square(x)

def psqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

DIV_MIN  = 1e-8
def pdiv(a, b):
    return np.where(np.abs(b) < DIV_MIN, 0.0, a / b)

SAT_MAX  = 1e6
def sat(x):
    x = np.nan_to_num(x, neginf=-SAT_MAX, posinf=SAT_MAX)
    return np.clip(x, -SAT_MAX, SAT_MAX)

class _DivToPDiv(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id="pdiv", ctx=ast.Load()),
                            args=[node.left, node.right],
                            keywords=[])
        return node

def _compile_with_safe_div(expr: str):
    tree = ast.parse(expr, mode="eval")
    tree = _DivToPDiv().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<best_expr_ast>", "eval")

SAFE_BUILTINS = {
    "__import__": __import__,
    "abs": abs, "min": min, "max": max,
    "float": float, "int": int, "pow": pow, "round": round,
}

def compile_expr(expr: str):
    allowed = {
        "Q": None, "C1": None, "C2": None, "C3": None,
        "softplus": softplus, "log1p": log1p_stable, "exp": exp_stable, "log": log_stable,
        "pow2": pow2, "psqrt": psqrt, "pdiv": pdiv,
        "np": np, "pi": math.pi, "e": math.e,
    }
    code = _compile_with_safe_div(expr)
    def f(Q, C1, C2, C3):
        env = dict(allowed); env.update({"Q": Q, "C1": C1, "C2": C2, "C3": C3})
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            val = eval(code, {"__builtins__": SAFE_BUILTINS}, env)
        return sat(val)
    return f

# ========= 数据工具 =========
COLUMNS_CANDIDATES = {
    "C1": ["Poloxamer 407", "Poloxmaer 407", "Cpol", "C_pol", "C1"],
    "C2": ["Ethanol", "Ceth", "C_eth", "C2"],
    "C3": ["Propylene glycol", "Cpg", "C_pg", "C3"],
}
JOIN_KEY_CANDIDATES = ["Formulation", "Formulation ID", "FID", "ID", "Sample", "Name"]

def _lower_map(cols: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in cols}

def _pick_first(df: pd.DataFrame, cands: List[str]) -> str | None:
    lm = _lower_map(list(df.columns))
    for name in cands:
        if name.lower() in lm:
            return lm[name.lower()]
    return None

def _find_formula_cols(df: pd.DataFrame) -> Dict[str, str] | None:
    c1 = _pick_first(df, COLUMNS_CANDIDATES["C1"])
    c2 = _pick_first(df, COLUMNS_CANDIDATES["C2"])
    c3 = _pick_first(df, COLUMNS_CANDIDATES["C3"])
    if c1 and c2 and c3:
        return {"C1": c1, "C2": c2, "C3": c3}
    return None

def _find_join_key(dfA: pd.DataFrame, dfB: pd.DataFrame) -> str | None:
    Al = _lower_map(list(dfA.columns)); Bl = _lower_map(list(dfB.columns))
    for k in JOIN_KEY_CANDIDATES:
        kl = k.lower()
        if kl in Al and kl in Bl:
            return Al[kl]
    inter = set(Al.keys()).intersection(Bl.keys())
    for k in inter:
        if any(x in k for x in ["id", "formu", "sample", "name"]):
            return Al[k]
    return None

def load_release_and_formulas(
    xlsx_path: str,
    sheet_release: str,
    sheet_formulas: str | None,
    join_key: str | None
) -> tuple[pd.DataFrame, Dict[str, str], str]:
    df_rel = pd.read_excel(xlsx_path, sheet_name=sheet_release)
    cmap = _find_formula_cols(df_rel)
    if cmap:
        return df_rel, cmap, "(in-release)"
    if sheet_formulas:
        try:
            df_for = pd.read_excel(xlsx_path, sheet_name=sheet_formulas)
        except Exception as e:
            raise KeyError(f"未在文件中找到工作表: {sheet_formulas}") from e
        cmap_for = _find_formula_cols(df_for)
        if cmap_for is None:
            raise KeyError(f"[{sheet_formulas}] 中未找到 C1/C2/C3 三列。")
        use_key = None
        if join_key is not None:
            if join_key not in df_rel.columns or join_key not in df_for.columns:
                raise KeyError(f"指定 join_key={join_key} 不在两个表中。")
            use_key = join_key
        else:
            use_key = _find_join_key(df_rel, df_for)
        if use_key is not None:
            merged = pd.merge(df_rel, df_for, on=use_key, how="inner", suffixes=("", "_for"))
            return merged, {"C1": cmap_for["C1"], "C2": cmap_for["C2"], "C3": cmap_for["C3"]}, use_key
        if len(df_rel) != len(df_for):
            raise ValueError(f"[{sheet_release}] 与 [{sheet_formulas}] 行数不同，且无公共键可合并。")
        df_for = df_for.reset_index(drop=True)
        df_rel = df_rel.reset_index(drop=True)
        merged = pd.concat([df_rel, df_for], axis=1)
        return merged, {"C1": cmap_for["C1"], "C2": cmap_for["C2"], "C3": cmap_for["C3"]}, "__index__"
    raise KeyError(f"[{sheet_release}] 未包含配方列，且未提供对应的 Formulas 表。")

# ========= Q归一化与积分器 =========
def get_q_scale(Y_tr):
    q_scale_file = os.path.join(os.path.dirname(BEST_EXPR_FILE), "q_scale.txt")
    if os.path.exists(q_scale_file):
        return float(open(q_scale_file, "r", encoding="utf-8").read().strip())
    else:
        return float(np.nanmax(Y_tr))

# ========= Q域积分器（与主训练同构） =========
def simulate_one_like_train(f, c1, c2, c3, y_obs_row, times):
    # 每条记录的上限（与主程序一致）：原始 Q 域
    qcap = 1.5 * float(np.nanmax(y_obs_row))
    Q = 0.0
    out_Q = np.empty_like(times, dtype=float)

    # 自适应细分 + 安全饱和（等价于主代码 simulate_series）
    substeps, adapt_max, dt_floor = 8, 8, 1e-6
    def _sat(v, lo=-1e6, hi=1e6): 
        if not np.isfinite(v): return 0.0
        return min(max(v, lo), hi)
    def _step(Q, dt):
        k1 = _sat(f(Q, c1, c2, c3))
        k2 = _sat(f(_sat(Q + 0.5*dt*k1), c1, c2, c3))
        k3 = _sat(f(_sat(Q + 0.5*dt*k2), c1, c2, c3))
        k4 = _sat(f(_sat(Q + dt*k3),  c1, c2, c3))
        Q_new = _sat(Q + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))
        if not np.isfinite(Q_new): return np.nan
        if Q_new < 0.0: Q_new = 0.0
        if Q_new > qcap: Q_new = qcap
        return Q_new

    t_prev = 0.0
    for i, t_target in enumerate(times):
        dt_total = float(t_target - t_prev)
        n0 = max(1, substeps)
        ok = False
        for ref in range(adapt_max + 1):
            n = n0 * (2 ** ref)
            dt = max(dt_total / n, dt_floor)
            Q_try, finite = Q, True
            for _ in range(n):
                Q_try = _step(Q_try, dt)
                if not np.isfinite(Q_try): finite = False; break
            if finite:
                Q = Q_try; ok = True; break
        if not ok: return np.full_like(times, np.nan)
        out_Q[i] = Q
        t_prev = t_target

    return out_Q  # ⚠️ 不要再乘 q_scale

def simulate_mat_like_train(f, C1v, C2v, C3v, Y_obs, times):
    N = Y_obs.shape[0]
    Y_hat = np.zeros_like(Y_obs)
    for i in range(N):
        Y_hat[i,:] = simulate_one_like_train(f, C1v[i], C2v[i], C3v[i], Y_obs[i], times)
    return Y_hat

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

# ========= 主流程 =========
def main():
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    expr = read_best_expr(BEST_EXPR_FILE)
    print("[best_expr] ", expr)
    f = compile_expr(expr)

    # 读取并合并 train/test
    df_tr, cmap_tr, key_tr = load_release_and_formulas(
        XLSX_PATH, SHEET_REL_TRAIN, SHEET_FOR_TRAIN, JOIN_KEY)
    df_te, cmap_te, key_te = load_release_and_formulas(
        XLSX_PATH, SHEET_REL_TEST,  SHEET_FOR_TEST,  JOIN_KEY)
    print(f"[info] Train 使用键: {key_tr} | Test 使用键: {key_te}")

    # 固定时间轴
    Y_tr = df_tr[R_COLS].to_numpy(float)
    Y_te = df_te[R_COLS].to_numpy(float)
    times = TIMES_HRS
    print(f"[info] 时间列: {R_COLS}")

    # 固定范围缩放到 [0,1]
    C1_tr = scale_fixed(df_tr[cmap_tr["C1"]].to_numpy(float), *C_BOUNDS["C1"])
    C2_tr = scale_fixed(df_tr[cmap_tr["C2"]].to_numpy(float), *C_BOUNDS["C2"])
    C3_tr = scale_fixed(df_tr[cmap_tr["C3"]].to_numpy(float), *C_BOUNDS["C3"])
    C1_te = scale_fixed(df_te[cmap_te["C1"]].to_numpy(float), *C_BOUNDS["C1"])
    C2_te = scale_fixed(df_te[cmap_te["C2"]].to_numpy(float), *C_BOUNDS["C2"])
    C3_te = scale_fixed(df_te[cmap_te["C3"]].to_numpy(float), *C_BOUNDS["C3"])

    # Q归一化
    q_scale = get_q_scale(Y_tr)
    print(f"[info] q_scale = {q_scale:.6g}")

    def eval_and_plot(tag: str, C1v, C2v, C3v, Y_obs):
        Y_hat = simulate_mat_like_train(f, C1v, C2v, C3v, Y_obs, times)
        E = rmse(Y_obs.flatten(), Y_hat.flatten())
        R = r2(Y_obs.flatten(), Y_hat.flatten())
        print(f"[{tag}] RMSE={E:.3f}  R^2={R:.4f}")

        plt.figure(figsize=(5,5))
        plt.scatter(Y_obs.flatten(), Y_hat.flatten(), s=8, alpha=0.6)
        lim = [0, max(float(np.nanmax(Y_obs)), float(np.nanmax(Y_hat)))]
        plt.plot(lim, lim, linestyle="--")
        plt.xlabel("Observed Q (ug/cm^2)")
        plt.ylabel("Predicted Q (ug/cm^2)")
        plt.title(f"{tag} | RMSE={E:.1f}, R^2={R:.3f}")
        plt.tight_layout()
        plt.savefig(f"{PLOT_SAVE_PATH}/ivrt_pred_{tag}.png", dpi=160)
        plt.close()

        nplot = min(6, Y_obs.shape[0])
        idx = np.linspace(0, Y_obs.shape[0]-1, nplot, dtype=int)
        plt.figure(figsize=(9,6))
        for i in idx:
            plt.plot(times, Y_obs[i], label=f"obs#{i}", alpha=0.9)
            plt.plot(times, Y_hat[i], linestyle="--", label=f"pred#{i}", alpha=0.9)
        plt.xlabel("Time (h)"); plt.ylabel("Q (ug/cm^2)")
        plt.title(f"{tag} time-series")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{PLOT_SAVE_PATH}/ivrt_series_{tag}.png", dpi=160)
        plt.close()

    print("===== TRAIN =====")
    eval_and_plot("train", C1_tr, C2_tr, C3_tr, Y_tr)
    print("===== TEST =====")
    eval_and_plot("test",  C1_te, C2_te, C3_te, Y_te)

if __name__ == "__main__":
    main()
