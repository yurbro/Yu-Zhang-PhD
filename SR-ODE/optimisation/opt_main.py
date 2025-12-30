from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opt_expr_api import optimize, OptimizeConfig, eval_Q6, curve_Q

# ========= 基本设置 =========
OUTDIR = Path("Symbolic Differential/SymODE/Optimisation/out_opt")
OUTDIR.mkdir(parents=True, exist_ok=True)

# —— 关键：按你附件对齐的 q_scale（不要再用 3008;975.1784004393666）——
# Q_SCALE_ALIGNED = 3008.198194823261
Q_SCALE_ALIGNED = 975.1784004393666

# ========= 工具函数 =========
def save_topk_xlsx(res: dict, cfg: OptimizeConfig, path: Path):
    """
    写出 Top-K/Best/History/Config 到一个 XLSX。
    res['topk'] 的元素形如 [C1,C2,C3,Q6,Q6_pen,fitness]
    res['history'] 是 [(iter, best_raw, best_pen), ...]
    """
    topk_cols = ["C1","C2","C3","Q6","Q6_pen","fitness"]
    topk_df = pd.DataFrame(res["topk"], columns=topk_cols)

    best_df = pd.DataFrame([res["best"]])
    hist_df = pd.DataFrame(res["history"], columns=["iter","best_raw","best_pen"])
    cfg_df = pd.DataFrame({"param": list(res["config"].keys()),
                           "value": list(res["config"].values())})

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        topk_df.to_excel(xl, index=False, sheet_name="TopK")
        best_df.to_excel(xl, index=False, sheet_name="Best")
        hist_df.to_excel(xl, index=False, sheet_name="History")
        cfg_df.to_excel(xl, index=False, sheet_name="Config")

def plot_best_curve(res: dict, cfg: OptimizeConfig, outdir: Path):
    t, Q = res["best_curve"]
    df = pd.DataFrame({"t_h": t, "Q": Q})
    df.to_csv(outdir / "best_curve.csv", index=False)

    plt.figure()
    plt.plot(t, Q)
    plt.xlabel("Time (h)"); plt.ylabel("Q")
    plt.title("Best Q(t)")
    plt.savefig(outdir / "best_curve.png", bbox_inches="tight")
    plt.close()

def plot_convergence(res: dict, outdir: Path, objective: str):
    hist = pd.DataFrame(res["history"], columns=["iter","best_raw","best_pen"])
    plt.figure()
    if objective == "raw":
        plt.plot(hist["iter"], hist["best_raw"])
        plt.ylabel("Best Q6 so far")
    else:
        plt.plot(hist["iter"], hist["best_pen"])
        plt.ylabel("Best penalized Q6 so far")
    plt.xlabel("Iteration"); plt.title("CMA-ES Convergence")
    plt.savefig(outdir / "convergence.png", bbox_inches="tight")
    plt.close()
    hist.to_csv(outdir / "convergence_history.csv", index=False)

def plot_sensitivity_1d(best: dict, cfg: OptimizeConfig, outdir: Path, n: int = 100):
    """
    围绕最优点，分别扫 C1/C2/C3 在物理域全范围的 Q6（按当前 objective 返回）
    """
    bounds = cfg.bounds_phys
    for var in ["C1","C2","C3"]:
        lo, hi = bounds[var]
        xs = np.linspace(lo, hi, n)
        ys = []
        for v in xs:
            C1, C2, C3 = best["C1"], best["C2"], best["C3"]
            if var == "C1": C1 = v
            elif var == "C2": C2 = v
            else: C3 = v
            Q6, Q6_pen = eval_Q6(C1, C2, C3, cfg, return_penalized=True)
            ys.append(Q6 if cfg.objective == "raw" else Q6_pen)
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(var); plt.ylabel("Objective (Q6 raw or penalized)")
        plt.title(f"1D sensitivity at best: vary {var}")
        plt.savefig(outdir / f"sensitivity_{var}.png", bbox_inches="tight")
        plt.close()

def plot_surface_C1C2(best: dict, cfg: OptimizeConfig, outdir: Path, n1: int = 80, n2: int = 80):
    """
    固定 C3=best，画 C1×C2 的 2D 响应面（imshow）
    """
    lo1, hi1 = cfg.bounds_phys["C1"]; lo2, hi2 = cfg.bounds_phys["C2"]
    xs = np.linspace(lo1, hi1, n1)
    ys = np.linspace(lo2, hi2, n2)
    Z = np.empty((n2, n1), dtype=float)
    for j, c2 in enumerate(ys):
        for i, c1 in enumerate(xs):
            Q6, Q6_pen = eval_Q6(c1, c2, best["C3"], cfg, return_penalized=True)
            Z[j, i] = Q6 if cfg.objective == "raw" else Q6_pen

    # 存 CSV（长表）
    xs_rep = np.repeat(xs, len(ys))
    ys_tile = np.tile(ys, len(xs))
    pd.DataFrame({"C1": xs_rep, "C2": ys_tile, "objective": Z.T.flatten()}).to_csv(
        outdir / "surface_C1_C2_at_bestC3.csv", index=False
    )

    plt.figure()
    plt.imshow(Z, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect="auto")
    plt.colorbar()
    plt.xlabel("C1"); plt.ylabel("C2")
    plt.title(f"Response surface ({'Q6' if cfg.objective=='raw' else 'penalized Q6'}) @ C3={best['C3']:.3f}")
    plt.savefig(outdir / "surface_C1_C2_at_bestC3.png", bbox_inches="tight")
    plt.close()

# ========= 主流程（你现有逻辑基础上增强）=========
if __name__ == "__main__":
    # 1) CMA-ES + 软上限（推荐用于工程可用解）
    cfg = OptimizeConfig(
        method="cma",
        objective="softcap",
        q_scale=Q_SCALE_ALIGNED,   # 关键：和你的数据一致
        cap=3500, lambda_pen=1.0,
        cma_iters=60, cma_pop=13, cma_sigma0=0.2,
        topk=50,                   # 多存一些 Top-K
    )
    res = optimize(cfg)

    # 保存 Top-K 到 XLSX
    save_topk_xlsx(res, cfg, OUTDIR / "TopK_CMAES.xlsx")
    # 最优 Q(t) 曲线
    plot_best_curve(res, cfg, OUTDIR)
    # 收敛曲线
    plot_convergence(res, OUTDIR, cfg.objective)
    # 灵敏度 + 2D 响应面
    plot_sensitivity_1d(res["best"], cfg, OUTDIR, n=120)
    plot_surface_C1C2(res["best"], cfg, OUTDIR, n1=90, n2=90)

    print("Best (CMA, softcap):", res["best"])
    print("Top-5:")
    for row in res["topk"][:5]:
        print(row)

    # 2) 可选：DE 原始目标（看“天花板”）
    cfg2 = OptimizeConfig(
        method="de",
        objective="raw",
        q_scale=Q_SCALE_ALIGNED,   # 同样对齐
        de_pop=60, de_gens=80, de_F=0.7, de_CR=0.9,
        topk=50,
    )
    res2 = optimize(cfg2)
    save_topk_xlsx(res2, cfg2, OUTDIR / "TopK_DE.xlsx")
    print("Best (DE, raw):", res2["best"])
