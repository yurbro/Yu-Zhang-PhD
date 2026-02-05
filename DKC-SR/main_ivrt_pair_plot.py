# -*- coding: utf-8 -*-
"""
main_ivrt_pair.py
-----------------
Reads your specific Excel format:
- Train: sheets "Formulas-train" + "Release-train"
- Test : sheets "Formulas-test"  + "Release-test"
Joins by "Run No".
Maps columns to variables: C1=Poloxamer 407, C2=Ethanol, C3=Propylene glycol.
Assumes release columns R_t1..R_t7 correspond to times [0.5,1,2,3,4,5,6] hours.
Automatically prepends t=0, Q=0 to each record.
Scales C1,C2,C3 to [0,1] using bounds: C1:[20,30], C2:[10,20], C3:[10,20] (% w/w).
Trains SR-ODE (strong form, RK4) and saves best expressions + metrics.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import json, numpy as np, pandas as pd, sympy as sp

from sr_ode_mod import Dataset, Record, Config, train_symbolic_ode, to_infix_str, to_sympy, build_pset_from_config, compile_individual, simulate_series, HARD_KO
import time

# ===== Known bounds (percent w/w) =====
C_BOUNDS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}

def _scale_vars(raw: dict) -> dict:
    out = {}
    for k,(lo,hi) in C_BOUNDS.items():
        v = float(raw[k])
        out[k] = (v - lo) / (hi - lo)  # map to [0,1]
    return out

T_NONZERO = [0.5, 1, 2, 3, 4, 5, 6]
R_COLS = ["R_t1","R_t2","R_t3","R_t4","R_t5","R_t6","R_t7"]
VAR_MAP = {"Poloxamer 407":"C1","Ethanol":"C2","Propylene glycol":"C3"}

def build_dataset_from_pair(df_form: pd.DataFrame, df_rel: pd.DataFrame) -> Dataset:
    # minimal sanitation
    if "Run No" not in df_form.columns or "Run No" not in df_rel.columns:
        raise ValueError("Both sheets must contain 'Run No' column.")
    missing = [c for c in R_COLS if c not in df_rel.columns]
    if missing:
        raise ValueError(f"Release sheet missing columns: {missing}")
    miss_f = [c for c in VAR_MAP if c not in df_form.columns]
    if miss_f:
        raise ValueError(f"Formulas sheet missing columns: {miss_f}")

    # Join
    df = pd.merge(df_form, df_rel, on="Run No", how="inner")
    df = df.sort_values("Run No")

    recs: List[Record] = []
    for _, row in df.iterrows():
        vars_raw = {"C1": float(row["Poloxamer 407"]),
                    "C2": float(row["Ethanol"]),
                    "C3": float(row["Propylene glycol"])}
        Q_nonzero = [float(row[c]) for c in R_COLS]
        t = np.array([0.0] + T_NONZERO, dtype=float)
        Q = np.array([0.0] + Q_nonzero, dtype=float)
        Q0 = float(Q[0])
        recs.append(Record(t=t, Q=Q, Q0=Q0, vars=_scale_vars(vars_raw)))
    return Dataset(recs)

def default_cfg() -> Config:
    return Config(
        var_names=("Q","C1","C2","C3"),
        must_have=("Q","C1","C2","C3"),
        primitive_names=("add","sub","mul","div","softplus","pow2"),
        ephemeral_range=(-2.0, 2.0),
        pop_size=3000, ngen=300, cxpb=0.6, mutpb=0.4, tournsize=5,
        tree_len_max=25, init_depth_min=1, init_depth_max=4,
        integrator="rk4", substeps=8, adapt_refine_max=8, dt_floor=1e-6,
        qcap_factor=1.5, clamp_nonneg=True,
        alpha_complexity=5e-4,
        nest_penalties={"exp":1e-3, "softplus":1e-3, "log1p":1e-3, "pow2":1e-3},
        hard_nonneg=True, hard_dfdQ_nonpos=True, hard_tol=1e-9,
        grad_check_points=31, grad_eps_rel=1e-3,
        lambda_gradC=2e-3, lambda_gradC_mag=5e-3, gradC_mag_min=5e-3,lambda_divQ=5e-3,
        gradC_signs={"C1":-1,"C2":+1,"C3":+1},
        n_jobs=None, seed=12,
        verbosity=2, show_progress=False, log_best_every=50,
        hard_check_nrecords=0, # all records
        hard_dfdQ_margin=7e-4, hard_check_all_points=True,
        debug_serial_eval=True,
        hof_snapshot_every=50, hof_snapshot_dir="Symbolic Differential/SymODE/artifacts/ivrt_pair", 
        hof_snapshot_topk=20,
    )

def load_pair(path: Path, f_sheet: str, r_sheet: str) -> Dataset:
    df_form = pd.read_excel(path, sheet_name=f_sheet)
    df_rel  = pd.read_excel(path, sheet_name=r_sheet)
    return build_dataset_from_pair(df_form, df_rel)

def evaluate(best, ds: Dataset, cfg: Config, out_dir: Path, tag: str, q_scale_train: float):
    pset = build_pset_from_config(cfg)
    f = compile_individual(best, pset)
    mse_sum = 0.0; n_pts = 0; mse_sum_n = 0.0
    for rec in ds.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        pred = simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        diff = pred - rec.Q
        mse_sum += float(np.dot(diff, diff))
        n_pts += diff.size
        diff_n = (pred / q_scale_train) - (rec.Q / q_scale_train)
        mse_sum_n += float(np.dot(diff_n, diff_n))
    mse = mse_sum / max(1, n_pts)
    mse_n = mse_sum_n / max(1, n_pts)
    (out_dir/f"metrics_{tag}.json").write_text(json.dumps({"MSE": mse, "MSE_normalized_by_train_q_scale": mse_n}, indent=2))
    print(f"[{tag}] MSE={mse:.6g}, MSE(norm by train q_scale)={mse_n:.6g}")

def main():
    start_time = time.time()
    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    xlsx = Path("Symbolic Differential/SymODE/data/IVRT-Pure.xlsx")  # <- your uploaded file name
    if not xlsx.exists():
        raise FileNotFoundError(f"{xlsx} not found. Place it next to this script or update the path.")

    ds_train = load_pair(xlsx, "Formulas-train", "Release-train")
    ds_test  = load_pair(xlsx, "Formulas-test",  "Release-test")

    cfg = default_cfg()
    print(f"[data] train_records={len(ds_train.records)}, test_records={len(ds_test.records)}")
    print(f"[data] time_points={[0.0]+T_NONZERO}")
    print(f"[cfg] pop={cfg.pop_size}, ngen={cfg.ngen}, cxpb={cfg.cxpb}, mutpb={cfg.mutpb}, n_jobs={cfg.n_jobs}, verbosity={cfg.verbosity}")

    best, hof, stats = train_symbolic_ode(ds_train, cfg)
    try:
        from sr_ode_mod import to_infix_str
        print("[final best expr]", to_infix_str(best)[:300])
    except Exception:
        pass

    out_dir = Path("Symbolic Differential/SymODE/artifacts/ivrt_pair")
    out_dir.mkdir(exist_ok=True)

    # Export expressions
    (out_dir/"best_infix.txt").write_text(to_infix_str(best)+"\n", encoding="utf-8")
    try:
        expr, _ = to_sympy(best)
        (out_dir/"best_sympy.txt").write_text(str(expr)+"\n", encoding="utf-8")
        (out_dir/"best_latex.txt").write_text(sp.latex(expr)+"\n", encoding="utf-8")
    except Exception as e:
        print("[Warn] SymPy export failed:", e)

    # Unified scaling with training q_scale
    q_scale_train = getattr(ds_train, "q_scale", 1.0)
    evaluate(best, ds_train, cfg, out_dir, "train", q_scale_train)
    evaluate(best, ds_test,  cfg, out_dir, "test",  q_scale_train)

    # === 导出 HOF 前N ===
    import csv
    from sr_ode_mod import to_infix_str, build_pset_from_config, compile_individual, simulate_series

    hof_csv = out_dir / "hof_top.csv"
    with hof_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank","size","fitness_total","mse_train_norm","infix"])
        pset = build_pset_from_config(cfg)
        for k, ind in enumerate(hof, start=1):
            # 适应度（含罚项）的总值
            fit = ind.fitness.values[0]
            # 计算“按训练集 q_scale 归一”的 MSE（只对 top N，成本可接受）
            f_comp = compile_individual(ind, pset)
            sse = 0.0; npts = 0
            for rec in ds_train.records:
                qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
                pred = simulate_series(f_comp, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
                d = (pred - rec.Q) / max(1.0, q_scale_train)  # 归一化
                sse += float(d @ d); npts += d.size
            mse_train_norm = sse / max(1, npts)
            # 写一行
            writer.writerow([k, len(ind), f"{fit:.6g}", f"{mse_train_norm:.6g}", to_infix_str(ind)[:500]])
    print(f"[export] HOF written to {hof_csv}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[end] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, elapsed {elapsed/60:.1f} min")

    # Save cfg + bounds
    meta = {
        "q_scale": q_scale_train,
        "normalized": True,
        "var_bounds": {"C1":[20.0,30.0], "C2":[10.0,20.0], "C3":[10.0,20.0]},
        "cfg": cfg.__dict__,
        "time_points": [0.0] + T_NONZERO,
        "release_columns": R_COLS,
        "var_map": VAR_MAP,
        "run_start": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        "run_end": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed/60,
        "elapsed_hours": elapsed/3600,
    }
    (out_dir/"cfg.json").write_text(json.dumps(meta, indent=2))

    # === 画图 & 导出预测（和训练同构） ===
    from sr_ode_mod import build_pset_from_config, compile_individual, predict_dataset, plot_scatter, plot_series, export_predictions_csv

    pset = build_pset_from_config(cfg)
    f_best = compile_individual(best, pset)

    # 训练集
    preds_tr = predict_dataset(f_best, ds_train, cfg)
    plot_scatter(ds_train, preds_tr, out_dir / "scatter_train.png")
    plot_series(ds_train, preds_tr, out_dir / "series_train.png", max_examples=6)
    export_predictions_csv(ds_train, preds_tr, out_dir / "pred_train.csv")

    # 测试集
    preds_te = predict_dataset(f_best, ds_test, cfg)
    plot_scatter(ds_test, preds_te, out_dir / "scatter_test.png")
    plot_series(ds_test, preds_te, out_dir / "series_test.png", max_examples=6)
    export_predictions_csv(ds_test, preds_te, out_dir / "pred_test.csv")
    print(f"[plots] PNGs and CSVs exported to {out_dir}")
    
if __name__ == "__main__":
    main()
