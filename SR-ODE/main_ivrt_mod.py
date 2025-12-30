# -*- coding: utf-8 -*-
"""
main_ivrt_mod.py
----------------
Driver script for SR-ODE on IVRT/IVPT-like datasets.
- Reads Excel (two-sheet layout or long-format), builds Dataset
- Scales C1/C2/C3 to [0,1] based on known bounds
- Uses training-set q_scale for normalized reporting on test split
- Saves best expression in infix, SymPy and LaTeX
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json, math, numpy as np, pandas as pd, sympy as sp
from sr_ode_mod import Dataset, Record, Config, train_symbolic_ode, to_infix_str, to_sympy

# ===== Known bounds (percent w/w) =====
C_BOUNDS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}

def _scale_vars(raw: dict) -> dict:
    out = {}
    for k,(lo,hi) in C_BOUNDS.items():
        v = float(raw[k])
        out[k] = (v - lo) / (hi - lo)  # map to [0,1]
    return out

def build_dataset_from_long_format(df: pd.DataFrame,
                                  time_col: str = "t_h",
                                  q_col: str = "Q",
                                  c_cols: Tuple[str,str,str] = ("C1","C2","C3"),
                                  group_cols: Tuple[str,...] = ("FormID",),
                                  include_t0: bool = True) -> Dataset:
    """
    Expects long table with columns: group_cols + [time_col, q_col] + c_cols (percent values)
    One record per group.
    """
    records: List[Record] = []
    for gid, g in df.groupby(list(group_cols)):
        g = g.sort_values(time_col)
        t = g[time_col].to_numpy(dtype=float)
        Q = g[q_col].to_numpy(dtype=float)
        # enforce t0
        if include_t0 and (len(t)==0 or t[0] > 1e-12):
            t = np.concatenate([[0.0], t])
            Q = np.concatenate([[0.0], Q])
        Q0 = float(Q[0])
        vars_raw = {"C1": float(g[c_cols[0]].iloc[0]),
                    "C2": float(g[c_cols[1]].iloc[0]),
                    "C3": float(g[c_cols[2]].iloc[0])}
        rec = Record(t=t, Q=Q, Q0=Q0, vars=_scale_vars(vars_raw))
        records.append(rec)
    return Dataset(records)

def default_cfg() -> Config:
    return Config(
        var_names=("Q","C1","C2","C3"),
        must_have=("Q","C1","C2","C3"),
        primitive_names=("add","sub","mul","div","log1p","softplus","exp","sqrt","pow2"),
        ephemeral_range=(-2.0, 2.0),
        pop_size=600, ngen=120, cxpb=0.6, mutpb=0.5, tournsize=4,
        tree_len_max=25, init_depth_min=0, init_depth_max=4,
        integrator="rk4", substeps=8, adapt_refine_max=8, dt_floor=1e-6,
        qcap_factor=1.5, clamp_nonneg=True,
        alpha_complexity=2e-4,
        nest_penalties={"exp":5e-3, "softplus":2e-3, "log1p":1e-3, "pow2":5e-4},
        hard_nonneg=True, hard_dfdQ_nonpos=True, hard_tol=1e-9,
        grad_check_points=11, grad_eps_rel=1e-3,
        lambda_gradC=0.0, gradC_signs={"C1":-1,"C2":+1,"C3":+1},
        n_jobs=None, seed=13,
    )

def main():
    # === 1) Load your data ===
    # Example: a long-format CSV/Excel assembled by you.
    # Replace the path below with your real file.
    data_path = Path("IVRT_or_IVPT_long.xlsx")  # <-- TODO: set your file
    if not data_path.exists():
        print(f"[Warn] {data_path} not found. Please set data_path to your file.")
        print("Expected columns: FormID, t_h, Q, C1, C2, C3 (C in %w/w).")
        return

    if data_path.suffix.lower() in [".xls",".xlsx"]:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Sanity enforce times include t=0 implicitly
    include_t0 = True
    ds_all = build_dataset_from_long_format(df, include_t0=include_t0)

    # === 2) Train-test split (simple) ===
    # Here do a simple 80/20 by record order; replace with your split logic if needed.
    n = len(ds_all.records)
    idx = np.arange(n)
    cut = max(1, int(0.8*n))
    tr_idx, te_idx = idx[:cut], idx[cut:] if cut < n else np.array([], dtype=int)

    ds_train = Dataset([ds_all.records[i] for i in tr_idx]) if cut>0 else Dataset([])
    ds_test  = Dataset([ds_all.records[i] for i in te_idx]) if cut<n else Dataset([])

    # === 3) Configure & train ===
    cfg = default_cfg()
    best, hof, stats = train_symbolic_ode(ds_train, cfg)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # === 4) Save best expression (human-readable) ===
    infix = to_infix_str(best)
    (out_dir/"best_infix.txt").write_text(infix+"\n", encoding="utf-8")

    try:
        expr_sym, syms = to_sympy(best)
        (out_dir/"best_sympy.txt").write_text(str(expr_sym)+"\n", encoding="utf-8")
        (out_dir/"best_latex.txt").write_text(sp.latex(expr_sym)+"\n", encoding="utf-8")
    except Exception as e:
        print("[Warn] SymPy export failed:", e)

    # === 5) Evaluation using training-set q_scale ===
    q_scale_train = getattr(ds_train, "q_scale", 1.0)
    def scale_train(x): return x / q_scale_train

    def eval_dataset(ds: Dataset, tag: str):
        import numpy as np
        from sr_ode_mod import simulate_series, compile_individual, build_pset_from_config, make_arg_getter
        pset = build_pset_from_config(cfg)
        f = compile_individual(best, pset)
        mse_sum = 0.0; n_pts = 0
        mse_sum_n = 0.0
        for rec in ds.records:
            qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
            pred = simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
            diff = pred - rec.Q
            mse_sum += float(np.dot(diff, diff))
            n_pts += diff.size
            diff_n = scale_train(pred) - scale_train(rec.Q)
            mse_sum_n += float(np.dot(diff_n, diff_n))
        mse = mse_sum / max(1, n_pts)
        mse_n = mse_sum_n / max(1, n_pts)
        (out_dir/f"metrics_{tag}.json").write_text(json.dumps({"MSE": mse, "MSE_normalized_by_train_q_scale": mse_n}, indent=2))
        print(f"[{tag}] MSE={mse:.6g}, MSE(norm by train q_scale)={mse_n:.6g}")

    eval_dataset(ds_train, "train")
    if ds_test.records:
        eval_dataset(ds_test, "test")

    # === 6) Save cfg & bounds ===
    meta = {
        "q_scale": q_scale_train,
        "normalized": True,
        "var_bounds": C_BOUNDS,
        "cfg": cfg.__dict__,
    }
    (out_dir/"cfg.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
