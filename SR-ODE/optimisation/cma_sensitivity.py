# -*- coding: utf-8 -*-
"""
CMA-ES 参数敏感度分析（pop × iters × sigma0 × seeds）
- 依赖你的 opt_expr_api.OptimizeConfig / optimize
- 生成结果 CSV + 一组 Matplotlib 图
- 默认 objective='softcap'，q_scale≈975.1784（与数据标度一致）
"""

import time
import math
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opt_expr_api import OptimizeConfig, optimize

# ================= 用户可改区 =================
OUTDIR = Path("Symbolic Differential/SymODE/Optimisation/cma_sensitivity_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 网格（你提的 pop/iters + 我补充的 sigma0）
POPS   = [50, 100, 200]
ITERS  = [50, 100, 200]   # 你所说的 gen，这里对应 CMA-ES 的迭代数 iters
SIGMA0 = [0.1, 0.2, 0.4]  # 常见且关键的 CMA 步长初值

SEEDS  = [20251001, 20251002, 20251003]  # 多个种子更稳；可再加

# 目标与尺度（保持与你数据一致）
OBJECTIVE   = "softcap"                 # 或 "raw"
Q_SCALE     = 975.1784004393666         # 你的数据标度
CAP         = 3500.0                    # softcap 的阈值
LAMBDA_PEN  = 1.0                       # softcap 的二次罚系数

TOPK_KEEP   = 10                        # 不重要，仅便于 debug/留档
# ============================================


def run_one(pop, iters, sigma0, seed):
    """跑一个组合，返回汇总 dict。"""
    cfg = OptimizeConfig(
        method="cma",
        objective=OBJECTIVE,
        q_scale=Q_SCALE,
        cap=CAP,
        lambda_pen=LAMBDA_PEN,
        cma_pop=pop,
        cma_iters=iters,
        cma_sigma0=sigma0,
        seed=seed,
        topk=TOPK_KEEP,
    )
    t0 = time.time()
    res = optimize(cfg)
    t1 = time.time()

    best = res["best"]
    best_Q6     = float(best.get("Q6", np.nan))
    best_Q6_pen = float(best.get("Q6_pen", np.nan))
    budget      = int(pop * iters)

    # 收敛历史（迭代 vs 最优值）
    hist = res.get("history", None)
    if hist is not None:
        hist_df = pd.DataFrame(hist, columns=["iter", "best_raw", "best_pen"])
    else:
        hist_df = None

    return {
        "pop": pop,
        "iters": iters,
        "sigma0": sigma0,
        "seed": seed,
        "budget": budget,
        "best_Q6": best_Q6,
        "best_Q6_pen": best_Q6_pen,
        "elapsed_sec": (t1 - t0),
        "history": hist_df,            # DataFrame or None
        "best": best,                  # dict: C1,C2,C3,...
        "config": cfg.__dict__,        # 仅为留档
    }


def main():
    rows = []
    histories = []  # 存 (pop,iters,sigma0,seed,hist_df)
    combos = list(itertools.product(POPS, ITERS, SIGMA0, SEEDS))
    print(f"Total runs = {len(combos)}")

    for pop, iters, sigma0, seed in combos:
        print(f"Run pop={pop}, iters={iters}, sigma0={sigma0}, seed={seed} ...")
        out = run_one(pop, iters, sigma0, seed)
        rows.append({
            k: out[k] for k in [
                "pop","iters","sigma0","seed","budget",
                "best_Q6","best_Q6_pen","elapsed_sec"
            ]
        })
        if out["history"] is not None:
            h = out["history"].copy()
            h["pop"] = pop; h["iters"] = iters; h["sigma0"] = sigma0; h["seed"] = seed
            histories.append(h)

    df = pd.DataFrame(rows)
    csv_path = OUTDIR / "cma_sensitivity_results.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # ---------------- 可视化（不指定颜色，每图一个画布） ----------------
    target_col = "best_Q6_pen" if OBJECTIVE == "softcap" else "best_Q6"

    # (1) Heatmap：每个 sigma0 画一张 pop×iters 的中位数热图
    for s0 in SIGMA0:
        sub = df[df["sigma0"] == s0].copy()
        # 以 median(跨 seed) 作为统计
        med = sub.groupby(["iters","pop"], as_index=False)[target_col].median()
        pivot = med.pivot(index="iters", columns="pop", values=target_col).sort_index()
        plt.figure()
        plt.imshow(pivot.values, origin="lower", aspect="auto",
                   extent=[min(POPS), max(POPS), min(ITERS), max(ITERS)])
        plt.colorbar()
        plt.xlabel("pop")
        plt.ylabel("iters")
        plt.title(f"Median {target_col} across seeds (sigma0={s0})")
        plt.savefig(OUTDIR / f"heatmap_{target_col}_sigma0_{s0}.png", bbox_inches="tight")
        plt.close()

    # (2) Boxplot：对每个 pop×iters×sigma0 组合，跨 seed 的分布
    # 注意：组合多，图会比较长；按 sigma0 分图，避免过挤
    for s0 in SIGMA0:
        sub = df[df["sigma0"] == s0].copy()
        sub = sub.sort_values(["pop","iters"])
        # 组标签
        groups = [f"p{p}-i{i}" for p,i in zip(sub["pop"], sub["iters"])]
        plt.figure()
        # 每个组合会重复多次（不同 seed），我们用分组 boxplot
        data = []
        labels = []
        for (p,i), g in sub.groupby(["pop","iters"]):
            data.append(g[target_col].values)
            labels.append(f"p{p}-i{i}")
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel(target_col)
        plt.title(f"Seed distributions per (pop,iters), sigma0={s0}")
        plt.xticks(rotation=45)
        plt.savefig(OUTDIR / f"box_{target_col}_sigma0_{s0}.png", bbox_inches="tight")
        plt.close()

    # (3) Scatter：最佳值 vs 预算（跨所有组合），看“预算效益”
    plt.figure()
    plt.scatter(df["budget"].values, df[target_col].values)
    plt.xlabel("budget (= pop * iters)")
    plt.ylabel(target_col)
    plt.title(f"{target_col} vs budget (all combos)")
    plt.savefig(OUTDIR / f"scatter_{target_col}_vs_budget.png", bbox_inches="tight")
    plt.close()

    # (4) Top 组合的收敛曲线（选 median 表现最好的 (pop,iters,sigma0)）
    # 先按 (pop,iters,sigma0) 聚合 seeds 的 median
    agg = df.groupby(["pop","iters","sigma0"], as_index=False)[target_col].median()
    top_row = agg.sort_values(target_col, ascending=False).iloc[0]
    top_p, top_i, top_s0 = int(top_row["pop"]), int(top_row["iters"]), float(top_row["sigma0"])
    print("Top config by median:", dict(pop=top_p, iters=top_i, sigma0=top_s0))

    # 收敛曲线：把 histories 里对应组合的多条曲线画在一张图
    sel_hist = [h for h in histories if
                int(h["pop"].iloc[0])==top_p and int(h["iters"].iloc[0])==top_i and float(h["sigma0"].iloc[0])==top_s0]
    if len(sel_hist) > 0:
        plt.figure()
        for h in sel_hist:
            if OBJECTIVE == "softcap":
                plt.plot(h["iter"].values, h["best_pen"].values)
            else:
                plt.plot(h["iter"].values, h["best_raw"].values)
        plt.xlabel("Iteration")
        plt.ylabel("Best-so-far")
        plt.title(f"Convergence (pop={top_p}, iters={top_i}, sigma0={top_s0}) across seeds")
        plt.savefig(OUTDIR / f"convergence_top_combo.png", bbox_inches="tight")
        plt.close()

    # 小结再存成一个人类可读的 md
    with open(OUTDIR / "README.md", "w", encoding="utf-8") as f:
        f.write(f"# CMA-ES 参数敏感度分析\n")
        f.write(f"- objective: {OBJECTIVE}\n")
        f.write(f"- q_scale: {Q_SCALE}\n")
        f.write(f"- cap/lambda_pen: {CAP}/{LAMBDA_PEN}\n")
        f.write(f"- pops: {POPS}\n")
        f.write(f"- iters: {ITERS}\n")
        f.write(f"- sigma0: {SIGMA0}\n")
        f.write(f"- seeds: {SEEDS}\n")
        f.write(f"- results csv: {OUTDIR/'cma_sensitivity_results.csv'}\n")
        f.write(f"- heatmaps/boxplots/scatter & top convergence saved in this folder.\n")

if __name__ == "__main__":
    main()
