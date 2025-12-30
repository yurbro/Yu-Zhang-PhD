# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, ast, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import sr_ode_mod as sm  # 直接复用你的训练模块

ART_DIR   = Path("Symbolic Differential")/"SymODE"/"artifacts"/"ivrt_pair"
XLSX_PATH = Path("Symbolic Differential")/"SymODE"/"data"/"IVRT-Pure.xlsx"
BEST_FILE = ART_DIR/"best_infix.txt"
CFG_FILE  = ART_DIR/"cfg.json"
HOF_FILE  = ART_DIR/"hof_top.csv"

# --- AST 变换：/ -> p_div； log(1+exp(x)) -> softplus(x)
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
        if isinstance(node.func, ast.Name) and node.func.id=="log" and len(node.args)==1:
            a = node.args[0]
            if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add):
                def _is_one(n): return isinstance(n, ast.Constant) and n.value in (1,1.0)
                def _is_exp(n): return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id=="exp" and len(n.args)==1
                if (_is_one(a.left) and _is_exp(a.right)) or (_is_one(a.right) and _is_exp(a.left)):
                    x = a.right.args[0] if _is_exp(a.right) else a.left.args[0]
                    return ast.Call(func=ast.Name(id="softplus", ctx=ast.Load()), args=[x], keywords=[])
        return node

def compile_infix(expr: str):
    tree = ast.parse(expr, mode="eval")
    tree = _DivToPDiv().visit(tree)
    tree = _Log1pExpToSoftplus().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<infix_ast>", "eval")
    ctx = {
        # 训练时的安全原子
        "p_div": sm.p_div, "softplus": sm.p_softplus, "exp": sm.p_exp,
        "sqrt": sm.p_sqrt, "abs": sm.p_abs, "tanh": sm.p_tanh, "relu": sm.p_relu,
        "min": sm.p_min, "max": sm.p_max, "pow2": sm.p_pow2, "pow3": sm.p_pow3,
        # 变量
        "Q": None, "C1": None, "C2": None, "C3": None,
        "pi": math.pi, "e": math.e, "np": np,
    }
    def f(Q, C1, C2, C3):
        env = dict(ctx); env.update({"Q":Q,"C1":C1,"C2":C2,"C3":C3})
        with np.errstate(all="ignore"):
            v = eval(code, {"__builtins__": {}}, env)
        return sm._sat(float(v))
    return f

def load_cfg():
    meta = json.loads(CFG_FILE.read_text(encoding="utf-8"))
    cfg = sm.Config(); 
    for k,v in meta["cfg"].items():
        if hasattr(cfg,k): setattr(cfg,k,v)
    return cfg, meta

def build_dataset(xlsx: Path, f_sheet: str, r_sheet: str, meta: dict) -> sm.Dataset:
    df_f = pd.read_excel(xlsx, sheet_name=f_sheet)
    df_r = pd.read_excel(xlsx, sheet_name=r_sheet)
    df = pd.merge(df_f, df_r, on="Run No", how="inner").sort_values("Run No")

    var_map = meta["var_map"]           # {"Poloxamer 407":"C1",...}
    bounds  = meta["var_bounds"]        # {"C1":[20,30],...}
    r_cols  = meta["release_columns"]   # ["R_t1",...,"R_t7"]
    times   = meta["time_points"]       # [0.0,0.5,...,6]

    def scale_fixed(raw):
        out = {}
        for name,(lo,hi) in bounds.items():
            out[name] = (float(raw[name]) - float(lo)) / (float(hi)-float(lo))
        return out

    recs=[]
    for _, row in df.iterrows():
        raw = {"C1": float(row["Poloxamer 407"]),
               "C2": float(row["Ethanol"]),
               "C3": float(row["Propylene glycol"])}
        Qnz = [float(row[c]) for c in r_cols]
        t   = np.array(times, dtype=float)
        # 训练就是 t 含 0.0，Q 在首位补 0
        if abs(t[0]) > 1e-12:
            t = np.concatenate([[0.0], t])
        Q = np.array([0.0] + Qnz, dtype=float)
        recs.append(sm.Record(t=t, Q=Q, Q0=0.0, vars=scale_fixed(raw)))
    return sm.Dataset(recs)

def eval_dataset(ds: sm.Dataset, f, cfg: sm.Config, q_scale: float):
    mse_sum=n_pts=0.0
    mse_sum_n=0.0
    for rec in ds.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        pred = sm.simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        d = pred - rec.Q
        mse_sum += float(d @ d); n_pts += d.size
        d_n = (pred/q_scale) - (rec.Q/q_scale)
        mse_sum_n += float(d_n @ d_n)
    return mse_sum/max(1,n_pts), mse_sum_n/max(1,n_pts)

def main():
    assert BEST_FILE.exists() and CFG_FILE.exists() and HOF_FILE.exists(), "缺少 artifacts 文件"
    cfg, meta = load_cfg()
    q_scale = float(meta["q_scale"])
    print("[info] q_scale(train) =", q_scale)

    ds_tr = build_dataset(XLSX_PATH, "Formulas-train", "Release-train", meta)
    ds_te = build_dataset(XLSX_PATH, "Formulas-test",  "Release-test",  meta)

    # 1) 先评 best_infix.txt
    expr_best = BEST_FILE.read_text(encoding="utf-8").strip()
    print("[best_infix]\n", expr_best)
    f_best = compile_infix(expr_best)
    mse_tr, mse_tr_n = eval_dataset(ds_tr, f_best, cfg, q_scale)
    mse_te, mse_te_n = eval_dataset(ds_te, f_best, cfg, q_scale)
    print(f"[best] train MSE={mse_tr:.6g}, norm={mse_tr_n:.6g}")
    print(f"[best] test  MSE={mse_te:.6g}, norm={mse_te_n:.6g}")

    # 2) 批量评估 HOF，找“真·贴指标”的式子
    df_hof = pd.read_csv(HOF_FILE)
    scores=[]
    for i, row in df_hof.iterrows():
        infix = str(row["infix"])
        try:
            f = compile_infix(infix)
            tr, trn = eval_dataset(ds_tr, f, cfg, q_scale)
            te, ten = eval_dataset(ds_te, f, cfg, q_scale)
            scores.append((i+1, len(str(infix)), trn, ten, tr, te, infix))
        except Exception as e:
            scores.append((i+1, len(str(infix)), np.inf, np.inf, np.inf, np.inf, f"[COMPILE_FAIL]{infix}"))
    scores.sort(key=lambda x: x[2])  # 先按 train_norm 排
    print("\n[top 10 by train_norm]")
    for k in range(min(10, len(scores))):
        rk, ln, trn, ten, tr, te, inf = scores[k]
        print(f"#{rk:02d}  train_norm={trn:.6g}  test_norm={ten:.6g}  |  train={tr:.3g}  test={te:.3g}")

    # 3) 如果 HOF 里有人能接近 0.0048/0.0240，画图以便肉眼核对
    out = ART_DIR/"pred_plots_recheck"; out.mkdir(exist_ok=True, parents=True)
    if scores and np.isfinite(scores[0][2]):
        _, _, trn, ten, tr, te, inf = scores[0]
        print(f"\n[best-by-eval] train_norm={trn:.6g}, test_norm={ten:.6g}\n{inf}\n")
        f = compile_infix(inf)
        # 散点
        all_obs, all_pred = [], []
        for rec in ds_tr.records:
            qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
            yhat = sm.simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
            all_obs.append(rec.Q); all_pred.append(yhat)
        y = np.concatenate(all_obs); yhat = np.concatenate(all_pred)
        lim = [0, float(max(1.0, np.nanmax(y), np.nanmax(yhat)))]
        plt.figure(figsize=(5,5))
        plt.scatter(y, yhat, s=8, alpha=0.6); plt.plot(lim, lim, "--")
        plt.xlabel("Observed Q"); plt.ylabel("Predicted Q"); plt.title("Train scatter")
        plt.tight_layout(); plt.savefig(out/"scatter_train.png", dpi=160); plt.close()
        # 时间序列（抽样）
        idx = np.linspace(0, len(ds_tr.records)-1, 6, dtype=int)
        plt.figure(figsize=(9,6))
        for i in idx:
            rec = ds_tr.records[i]
            qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
            yhat = sm.simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
            plt.plot(rec.t, rec.Q, "o-", label=f"obs#{i}")
            plt.plot(rec.t, yhat, "--", label=f"pred#{i}")
        plt.legend(ncol=2, fontsize=8); plt.xlabel("h"); plt.ylabel("Q")
        plt.tight_layout(); plt.savefig(out/"series_train.png", dpi=160); plt.close()

if __name__=="__main__":
    main()
