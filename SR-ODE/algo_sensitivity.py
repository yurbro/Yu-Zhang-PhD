# -*- coding: utf-8 -*-
"""
algo_sensitivity.py
对 SR-ODE 训练做“算法级”超参数敏感度分析（OAT：单因子-多水平-多种子）。

依赖你的现有代码：
- main_ivrt_pair.py: default_cfg(), build_dataset_from_pair(), load_pair()
- sr_ode_mod.py: train_symbolic_ode(), build_pset_from_config(), compile_individual(), simulate_series(), Record, Dataset

输出：
- artifacts/ivrt_pair/algo_sensitivity.csv  （每次训练一行，逐行追加）
"""

from __future__ import annotations
import time, csv, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

import sr_ode_mod as sm
import main_ivrt_pair as app
from sr_ode_mod import build_pset_from_config, compile_individual, simulate_series, Record

ART_DIR   = Path("Symbolic Differential")/"SymODE"/"artifacts"/"ivrt_pair"
XLSX_PATH = Path("Symbolic Differential")/"SymODE"/"data"/"IVRT-Pure.xlsx"

# ====== OAT：每次只改一个超参，其余用 baseline ======
BASELINE = {
    "pop_size": 4000, "ngen": 250, "cxpb": 0.6, "mutpb": 0.4, "tournsize": 5,
    "tree_len_max": 25, "init_depth_max": 4, "alpha_complexity": 5e-4,
}
GRID = {
    "pop_size":         [3000, 4000, 5000],
    "ngen":             [200, 250, 300],
    "cxpb":             [0.5, 0.6, 0.7],   # mutpb 保持你 cfg 里的值；如需联动可在 run_once 里改
    "tournsize":        [3, 5, 7],
    "tree_len_max":     [20, 25, 30],
    "init_depth_max":   [3, 4, 5],
    "alpha_complexity": [2e-4, 5e-4, 1e-3],
}
REPEATS = 3  # 每组不同随机种子次数

def build_datasets():
    """与主程序一致地读 Excel -> (ds_train, ds_test, meta)"""
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"{XLSX_PATH} not found")
    ds_train = app.load_pair(XLSX_PATH, "Formulas-train", "Release-train")
    ds_test  = app.load_pair(XLSX_PATH, "Formulas-test",  "Release-test")
    meta = {"q_scale": float(getattr(ds_train, "q_scale", 1.0))}
    return ds_train, ds_test, meta

def eval_mse_pair(best_ind, ds_train, ds_test, cfg, q_scale_train: float):
    """返回 (train_MSE, train_MSE_norm, test_MSE, test_MSE_norm) —— 与主程序 evaluate 同口径"""
    pset = build_pset_from_config(cfg)
    f = compile_individual(best_ind, pset)
    def _mse_on(ds):
        sse = sse_n = 0.0
        npts = 0
        for rec in ds.records:
            qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
            pred = simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
            d  = (pred - rec.Q)
            dn = d / max(1.0, q_scale_train)
            sse  += float(np.dot(d,  d))
            sse_n+= float(np.dot(dn, dn))
            npts += d.size
        mse   = sse  / max(1, npts)
        mse_n = sse_n/ max(1, npts)
        return mse, mse_n
    tr_mse, tr_mse_n = _mse_on(ds_train)
    te_mse, te_mse_n = _mse_on(ds_test)
    return tr_mse, tr_mse_n, te_mse, te_mse_n

def cfg_from_default():
    """从 main_ivrt_pair.default_cfg() 拿一份 cfg，并应用 BASELINE。"""
    cfg = app.default_cfg()
    # baseline 覆盖
    for k, v in BASELINE.items():
        setattr(cfg, k, v)
    return cfg

def run_once(override: dict, seed: int):
    """训练一次 + 评估一次。返回结果 dict，可直接写 CSV。"""
    # 1) 配置
    cfg = cfg_from_default()
    for k, v in override.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field '{k}'")
        setattr(cfg, k, v)
    cfg.seed = int(seed)

    if "cxpb" in override: cfg.mutpb = max(0.0, 1.0 - cfg.cxpb)  # 保持互补

    # 2) 数据
    ds_train, ds_test, meta = build_datasets()
    q_scale_train = float(meta["q_scale"])

    # 3) 训练
    t0 = time.time()
    best, hof, stats = sm.train_symbolic_ode(ds_train, cfg)
    sec = time.time() - t0

    # 4) 评估（同口径）
    tr_mse, tr_mse_n, te_mse, te_mse_n = eval_mse_pair(best, ds_train, ds_test, cfg, q_scale_train)

    # 5) 汇总
    from sr_ode_mod import to_infix_str
    rec = {
        **{k: override.get(k, BASELINE.get(k)) for k in GRID.keys()},
        "seed": seed,
        "size": len(best),
        "sec": round(sec, 2),
        "train_MSE": tr_mse,
        "train_MSE_norm": tr_mse_n,
        "test_MSE": te_mse,
        "test_MSE_norm": te_mse_n,
        "best_infix": to_infix_str(best)[:500],
    }
    return rec

def generate_oat_combos():
    """OAT 单因子组合列表（包含 baseline 本身，方便对照）"""
    combos = [dict()]  # baseline
    for key, vals in GRID.items():
        for v in vals:
            combos.append({key: v})
    return combos

def append_row_csv(path: Path, row: dict):
    existed = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not existed:
            w.writeheader()
        w.writerow(row)

def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ART_DIR / "algo_sensitivity.csv"
    combos = generate_oat_combos()
        # —— 已完成集合（超参+seed 唯一键）——
    done = set()
    if out_csv.exists():
        df_done = pd.read_csv(out_csv)
        if "error" in df_done.columns:
            df_done = df_done[df_done["error"].isna()]  # 仅跳过成功的；失败的可重试
        key_cols = list(GRID.keys()) + ["seed"]
        for _, row in df_done.iterrows():
            try:
                key = tuple([row.get(k, BASELINE.get(k)) for k in GRID.keys()] + [int(row["seed"])])
                done.add(key)
            except Exception:
                continue
    print(f"已有记录 {len(done)} 条，准备新增记录...")
    rng = np.random.RandomState(2025)

    for override in combos:
        for r in range(REPEATS):
            seed = int(rng.randint(0, 10_000_000))
            key = tuple([override.get(k, BASELINE.get(k)) for k in GRID.keys()] + [seed])
            if key in done:
                print("[SKIP]", override if override else {"baseline": True}, "seed=", seed)
                continue
            try:
                rec = run_once(override, seed)
                append_row_csv(out_csv, rec)
                print("[OK]", override if override else {"baseline": True},
                      "seed=", seed, " test_MSE_n=", f"{rec['test_MSE_norm']:.6g}",
                      " time=", rec["sec"], "s")
            except Exception as e:
                # 遇错也记录，便于排查
                bad = {**{k: override.get(k, BASELINE.get(k)) for k in GRID.keys()},
                       "seed": seed, "error": str(e)}
                append_row_csv(out_csv, bad)
                print("[ERR]", override if override else {"baseline": True}, "seed=", seed, " ->", e)

    print("Wrote", out_csv)

if __name__ == "__main__":
    main()
