# -*- coding: utf-8 -*-
"""
eval_best_infix.py
------------------
读取 main_ivrt_pair.py 训练导出的 best_infix.txt + cfg.json，
严格复用 sr_ode_mod 的数值环境（安全原子/积分器/qcap/钳制）在
train/test 上做预测，评估 MSE 与按训练 q_scale 归一的 MSE，并画图。

依赖：与你训练同一工程环境（sr_ode_mod.py）
用法：python eval_best_infix.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import json, math, ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 你工程里的核心模块 ===
import sr_ode_mod as sm  # Dataset, Record, Config, simulate_series, PRIMITIVE_REGISTRY 等

ART_DIR   = Path("Symbolic Differential") / "SymODE" / "artifacts" / "ivrt_pair"
XLSX_PATH = Path("Symbolic Differential") / "SymODE" / "data" / "IVRT-Pure.xlsx"
BEST_FILE = ART_DIR / "best_infix.txt"
CFG_FILE  = ART_DIR / "cfg.json"

# ---------- 1) 读取 cfg.json（完全对齐训练时设置） ----------
def load_cfg() -> Tuple[sm.Config, dict]:
    meta = json.loads(CFG_FILE.read_text(encoding="utf-8"))
    cfg_dict = meta.get("cfg", {})
    cfg = sm.Config()  # 用默认，再覆盖
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg, meta

# ---------- 2) 用 cfg.json 的 var_map/time_points/build dataset ----------
def build_dataset_from_excel(xlsx: Path, f_sheet: str, r_sheet: str, meta: dict) -> sm.Dataset:
    df_f = pd.read_excel(xlsx, sheet_name=f_sheet)
    df_r = pd.read_excel(xlsx, sheet_name=r_sheet)

    # 这些信息训练时已写到 cfg.json，复用即可
    var_map: Dict[str,str] = meta["var_map"]               # {"Poloxamer 407":"C1",...}
    r_cols:  List[str]     = meta["release_columns"]       # ["R_t1",...,"R_t7"]
    times:   List[float]   = meta["time_points"]           # [0.0,0.5,1,2,3,4,5,6]
    bounds:  Dict[str,List[float]] = meta["var_bounds"]    # {"C1":[20,30],...}

    # 按 Run No 合并（与 main 完全一致）
    if "Run No" not in df_f.columns or "Run No" not in df_r.columns:
        raise ValueError("Both sheets must contain 'Run No' column.")
    df = pd.merge(df_f, df_r, on="Run No", how="inner").sort_values("Run No")

    # 固定范围缩放到 [0,1]
    def scale_vars(raw: dict) -> dict:
        out = {}
        for k,(lo,hi) in bounds.items():
            v = float(raw[k]); out[k] = (v - float(lo)) / (float(hi) - float(lo))
        return out

    recs: List[sm.Record] = []
    for _, row in df.iterrows():
        # 原始配方 -> 归一
        raw = { var_map["Poloxamer 407"]: float(row["Poloxamer 407"]),
                var_map["Ethanol"]       : float(row["Ethanol"]),
                var_map["Propylene glycol"]: float(row["Propylene glycol"]) }
        vars_scaled = scale_vars(raw)

        # Q 与 t（注意：与 main 相同，前置 t=0/Q=0）
        Q_nonzero = [float(row[c]) for c in r_cols]
        t = np.array(times, dtype=float)
        if abs(t[0]) > 1e-12:
            # 保守起见，如果 meta 中 time_points 没有 0.0，则补上
            t = np.concatenate([[0.0], t])
            Q = np.array([0.0] + Q_nonzero, dtype=float)
        else:
            Q = np.array([0.0] + Q_nonzero, dtype=float)  # 训练就是这样写入的
        recs.append(sm.Record(t=t, Q=Q, Q0=Q[0], vars=vars_scaled))
    return sm.Dataset(recs)

# ---------- 3) 编译 infix 表达式为可调用 f(Q,C1,C2,C3) ----------
# 把所有 a/b => p_div(a,b)，并将 log(1+exp(x)) 识别为 softplus(x) 以复用 sm.p_softplus
class _DivToPDiv(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id="p_div", ctx=ast.Load()),
                            args=[node.left, node.right], keywords=[])
        return node

class _Log1pExpToSoftplus(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        # 匹配 log( 1 + exp(x) )
        if isinstance(node.func, ast.Name) and node.func.id == "log" and len(node.args) == 1:
            a = node.args[0]
            if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add):
                # 左边 1, 右边 exp(x)（两种排列都兼容）
                def _is_one(n): return isinstance(n, ast.Constant) and (n.value == 1 or n.value == 1.0)
                def _is_exp(n): return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "exp" and len(n.args) == 1
                if (_is_one(a.left) and _is_exp(a.right)) or (_is_one(a.right) and _is_exp(a.left)):
                    x = a.right.args[0] if _is_exp(a.right) else a.left.args[0]
                    return ast.Call(func=ast.Name(id="softplus", ctx=ast.Load()),
                                    args=[x], keywords=[])
        return node

def compile_infix(expr: str):
    # 先把字符串解析为 AST，并做两次变换
    tree = ast.parse(expr, mode="eval")
    tree = _DivToPDiv().visit(tree)
    tree = _Log1pExpToSoftplus().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<best_infix_ast>", "eval")

    # 允许的上下文（完全用训练时的安全原子）
    ctx = {
        "p_div": sm.p_div,
        "softplus": sm.p_softplus,
        "exp": sm.p_exp,
        "log": lambda x: sm._sat(math.log(max(1e-16, sm._sat(float(x))))),  # 仅兜底；softplus 已替换
        "sqrt": sm.p_sqrt,
        "abs": sm.p_abs,
        "tanh": sm.p_tanh,
        "relu": sm.p_relu,
        "min": lambda a,b: sm.p_min(a,b),
        "max": lambda a,b: sm.p_max(a,b),
        "pow2": sm.p_pow2,
        "pow3": sm.p_pow3,
        # 变量占位（eval 时会覆盖）
        "Q": None, "C1": None, "C2": None, "C3": None,
        "np": np, "pi": math.pi, "e": math.e,
    }

    def f(Q, C1, C2, C3):
        env = dict(ctx)
        env.update({"Q": Q, "C1": C1, "C2": C2, "C3": C3})
        # 静默数值告警（与训练的 p_* 饱和一起兜底）
        with np.errstate(all="ignore"):
            val = eval(code, {"__builtins__": {}}, env)
        return sm._sat(float(val))
    return f

# ---------- 4) 评估（与 main 的 evaluate 完全一致） ----------
def evaluate_on(ds: sm.Dataset, f, cfg: sm.Config, q_scale_train: float, tag: str, out_dir: Path):
    mse_sum = 0.0; n_pts = 0; mse_sum_n = 0.0
    for rec in ds.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        pred = sm.simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        diff = pred - rec.Q
        mse_sum += float(np.dot(diff, diff)); n_pts += diff.size
        diff_n = (pred / q_scale_train) - (rec.Q / q_scale_train)
        mse_sum_n += float(np.dot(diff_n, diff_n))
    mse = mse_sum / max(1, n_pts)
    mse_n = mse_sum_n / max(1, n_pts)
    (out_dir/f"metrics_recheck_{tag}.json").write_text(
        json.dumps({"MSE": mse, "MSE_normalized_by_train_q_scale": mse_n}, indent=2))
    print(f"[{tag}] MSE={mse:.6g}, MSE(norm by train q_scale)={mse_n:.6g}")
    return mse, mse_n

# ---------- 5) 绘图 ----------
def plots(ds: sm.Dataset, f, cfg: sm.Config, tag: str, save_dir: Path, max_examples: int = 6):
    save_dir.mkdir(parents=True, exist_ok=True)
    # 1) 散点：Observed vs Predicted（所有点拼一起）
    all_obs, all_pred = [], []
    for rec in ds.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        yhat = sm.simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        all_obs.append(rec.Q); all_pred.append(yhat)
    y = np.concatenate(all_obs); yhat = np.concatenate(all_pred)
    lim = [0, float(max(1.0, np.nanmax(y), np.nanmax(yhat)))]
    plt.figure(figsize=(5,5))
    plt.scatter(y, yhat, s=8, alpha=0.6)
    plt.plot(lim, lim, linestyle="--")
    plt.xlabel("Observed Q (μg/cm²)"); plt.ylabel("Predicted Q (μg/cm²)")
    plt.title(f"{tag}: Observed vs Predicted")
    plt.tight_layout(); plt.savefig(save_dir / f"scatter_{tag}.png", dpi=160); plt.close()

    # 2) 时间序列（抽样若干条）
    idx = np.linspace(0, len(ds.records)-1, min(max_examples, len(ds.records)), dtype=int)
    plt.figure(figsize=(9,6))
    for i in idx:
        rec = ds.records[i]
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        yhat = sm.simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        plt.plot(rec.t, rec.Q,  "o-", label=f"obs#{i}", alpha=0.9)
        plt.plot(rec.t, yhat, "--",  label=f"pred#{i}", alpha=0.9)
    plt.xlabel("Time (h)"); plt.ylabel("Q (μg/cm²)")
    plt.title(f"{tag}: Time series")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(save_dir / f"series_{tag}.png", dpi=160); plt.close()

# ---------- 6) 主流程 ----------
def main():
    assert BEST_FILE.exists(), f"{BEST_FILE} not found"
    assert CFG_FILE.exists(),  f"{CFG_FILE} not found"

    cfg, meta = load_cfg()
    expr = BEST_FILE.read_text(encoding="utf-8").strip()
    print("[best_infix]\n", expr)

    # 编译表达式：/ -> p_div；log(1+exp(.)) -> softplus(.)
    f = compile_infix(expr)

    # 数据：严格遵循 cfg.json
    ds_train = build_dataset_from_excel(XLSX_PATH, "Formulas-train", "Release-train", meta)
    ds_test  = build_dataset_from_excel(XLSX_PATH, "Formulas-test",  "Release-test",  meta)

    q_scale_train = float(meta.get("q_scale", 1.0))
    print(f"[info] q_scale (train) = {q_scale_train}")

    # 评估
    out = ART_DIR
    evaluate_on(ds_train, f, cfg, q_scale_train, "train", out)
    evaluate_on(ds_test,  f, cfg, q_scale_train, "test",  out)

    # 画图
    plot_dir = ART_DIR / "pred_plots"
    plots(ds_train, f, cfg, "train", plot_dir)
    plots(ds_test,  f, cfg, "test",  plot_dir)

if __name__ == "__main__":
    main()
