# opt_expr_api.py
# -*- coding: utf-8 -*-
"""
Programmatic API for optimizing Yu's dQ/dt expression
- Cs min-max to [0,1] before entering model
- Q integrated in tilde scale with RK45; mapped back via q_scale
- Optimizers: CMA-ES (diagonal) and Differential Evolution (DE)
- Objectives: 'raw' (maximize Q6) or 'softcap' (Q6 - lambda * max(0, Q6-cap)^2)

Usage:
    from opt_expr_api import optimize, OptimizeConfig
    cfg = OptimizeConfig(method="cma", objective="softcap", q_scale=3008.198194823261, cap=3500)
    res = optimize(cfg)
    print(res["best"])
    t, Q = res["best_curve"]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Literal, Optional
import numpy as np

# --------- Expression constant (from your model) ---------
B_CONST = 1.5346103937034676  # denominator width

# --------- Config dataclass ---------
ObjectiveT = Literal["raw", "softcap"]
MethodT = Literal["cma", "de"]

@dataclass
class OptimizeConfig:
    # Design space (physical units)
    bounds_phys: Dict[str, Tuple[float, float]] = None
    # Time horizon
    T: float = 6.0
    # q_scale for mapping Q = q_scale * Qtilde
    q_scale: float = 3008.198194823261
    # Objective choice
    objective: ObjectiveT = "softcap"
    # Soft cap settings (used when objective == "softcap")
    cap: float = 3500.0
    lambda_pen: float = 1.0
    # Optimizer selection
    method: MethodT = "cma"
    # CMA-ES params
    cma_iters: int = 60
    cma_pop: int = 13
    cma_sigma0: float = 0.2  # in normalized [0,1]^3
    # DE params
    de_gens: int = 60
    de_pop: int = 50
    de_F: float = 0.7
    de_CR: float = 0.9
    # Integrator tolerances
    rtol: float = 1e-6
    atol: float = 1e-9
    h0: float = 0.05
    h_min: float = 1e-4
    h_max: float = 0.25
    # Random seed
    seed: int = 20251008
    # How many top candidates to return
    topk: int = 20

    def __post_init__(self):
        if self.bounds_phys is None:
            self.bounds_phys = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}

# --------- Utilities ---------
def _minmax_norm(x: float, lo: float, hi: float) -> float:
    return (x - lo) / (hi - lo)

def _minmax_denorm(xn: float, lo: float, hi: float) -> float:
    return xn * (hi - lo) + lo

def _softplus(x: float) -> float:
    if x > 20.0: return x
    if x < -20.0: return math.exp(x)
    return math.log1p(math.exp(x))

# dQtilde/dt using normalized Cs (all in training scale)
def _dQtilde_dt(Qtilde: float, C1n: float, C2n: float, C3n: float) -> float:
    eps = 1e-9  # guard for C3n ~ 0
    denom = (C2n - ((C1n / (C3n + eps)) - (Qtilde**2)))**2 + (B_CONST**2)
    return _softplus(2.0 * Qtilde) / denom

# RK45 (Dormand–Prince 5(4)) for scalar ODE
def _rk45_Qtilde(C1n: float, C2n: float, C3n: float, T: float,
                 rtol: float, atol: float, h0: float, h_min: float, h_max: float) -> float:
    t, q, h = 0.0, 0.0, h0
    safety = 0.9
    max_steps = 200000
    steps = 0
    while t < T and steps < max_steps:
        if t + h > T: h = T - t
        k1 = _dQtilde_dt(q, C1n, C2n, C3n)
        k2 = _dQtilde_dt(q + h*0.25*k1, C1n, C2n, C3n)
        k3 = _dQtilde_dt(q + h*(3/32*k1 + 9/32*k2), C1n, C2n, C3n)
        k4 = _dQtilde_dt(q + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3), C1n, C2n, C3n)
        k5 = _dQtilde_dt(q + h*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4), C1n, C2n, C3n)
        k6 = _dQtilde_dt(q + h*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5), C1n, C2n, C3n)
        q5 = q + h*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)  # 5th
        q4 = q + h*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)                # 4th
        err = abs(q5 - q4)
        tol = atol + rtol*max(abs(q), abs(q5))
        if err <= tol or h <= h_min*1.01:
            q = q5; t += h
        s = 2.0 if err == 0.0 else safety * (tol / err)**0.2  # 1/(order+1), order=4
        h = min(max(h*s, h_min), h_max)
        steps += 1
    return q

def _eval_Q6_and_pen(C1: float, C2: float, C3: float, cfg: OptimizeConfig) -> Tuple[float, float]:
    C1n = _minmax_norm(C1, *cfg.bounds_phys["C1"])
    C2n = _minmax_norm(C2, *cfg.bounds_phys["C2"])
    C3n = _minmax_norm(C3, *cfg.bounds_phys["C3"])
    qtilde_T = _rk45_Qtilde(C1n, C2n, C3n, T=cfg.T,
                            rtol=cfg.rtol, atol=cfg.atol, h0=cfg.h0, h_min=cfg.h_min, h_max=cfg.h_max)
    Q6 = cfg.q_scale * qtilde_T
    if cfg.objective == "raw":
        return Q6, Q6
    # softcap
    exceed = max(0.0, Q6 - cfg.cap)
    Q6_pen = Q6 - cfg.lambda_pen * (exceed**2)
    return Q6, Q6_pen

# --------- Differential Evolution ---------
def _de_optimize(cfg: OptimizeConfig):
    rng = np.random.default_rng(cfg.seed)
    keys = ["C1","C2","C3"]

    def to_norm(C1, C2, C3):
        return np.array([
            _minmax_norm(C1, *cfg.bounds_phys["C1"]),
            _minmax_norm(C2, *cfg.bounds_phys["C2"]),
            _minmax_norm(C3, *cfg.bounds_phys["C3"]),
        ], dtype=float)

    def from_norm(z):
        C1 = _minmax_denorm(z[0], *cfg.bounds_phys["C1"])
        C2 = _minmax_denorm(z[1], *cfg.bounds_phys["C2"])
        C3 = _minmax_denorm(z[2], *cfg.bounds_phys["C3"])
        return C1, C2, C3

    # 初始化：在归一化空间均匀采样
    pop_z = rng.random(size=(cfg.de_pop, 3))
    pop = []
    for i in range(cfg.de_pop):
        C1, C2, C3 = from_norm(pop_z[i])
        Q6, Q6pen = _eval_Q6_and_pen(C1, C2, C3, cfg)
        fit = Q6 if cfg.objective=="raw" else Q6pen
        pop.append([pop_z[i].copy(), Q6, Q6pen, fit])  # 保存 z 与指标

    # 当前最优
    best = max(pop, key=lambda x: x[3]).copy()
    hist = [(0, best[1], best[2])]

    for g in range(1, cfg.de_gens+1):
        for i in range(cfg.de_pop):
            idxs = list(range(cfg.de_pop)); idxs.remove(i)
            a, b, c = rng.choice(idxs, size=3, replace=False)
            Az, Bz, Cz = pop[a][0], pop[b][0], pop[c][0]

            # DE/rand/1/bin in normalized space
            v = Az + cfg.de_F*(Bz - Cz)
            jrand = rng.integers(3)
            trial = np.empty(3, dtype=float)
            for j in range(3):
                if rng.random() < cfg.de_CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][0][j]
            # 镜像到 [0,1]^3
            trial = _mirror_unit(trial)

            # 评估
            C1, C2, C3 = from_norm(trial)
            Q6, Q6pen = _eval_Q6_and_pen(C1, C2, C3, cfg)
            fit = Q6 if cfg.objective=="raw" else Q6pen

            # 贪婪选择
            if fit >= pop[i][3]:
                pop[i] = [trial, Q6, Q6pen, fit]
                if fit >= best[3]:
                    best = pop[i].copy()

        hist.append((g, best[1], best[2]))

    # Top-K（反归一化导出）
    pop.sort(key=lambda x: x[3], reverse=True)
    topk = []
    for z, Q6, Q6pen, f in pop[:cfg.topk]:
        C1, C2, C3 = from_norm(z)
        topk.append([C1, C2, C3, Q6, Q6pen, f])

    C1b, C2b, C3b = from_norm(best[0])
    best_dict = {"C1":C1b, "C2":C2b, "C3":C3b, "Q6":best[1], "Q6_pen":best[2], "fitness":best[3]}
    return best_dict, topk, hist

# --------- Diagonal CMA-ES ---------
def _mirror_unit(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = np.where(x < 0, -x, x)
    x = np.where(x > 1, 2 - x, x)
    x = np.mod(x, 2.0)
    x = np.where(x > 1, 2 - x, x)
    return x

def _cma_es_diagonal(cfg: OptimizeConfig):
    rng = np.random.default_rng(cfg.seed)
    dim = 3
    pop_size = cfg.cma_pop
    w = np.array([math.log(pop_size + 0.5) - math.log(i + 1) for i in range(pop_size)])
    w = w / w.sum()
    mu_eff = 1.0 / np.sum(w**2)
    cc = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2*max(0, math.sqrt((mu_eff - 1)/(dim + 1)) - 1) + cs

    m = np.array([0.5, 0.5, 0.5])  # normalized center
    sigma = cfg.cma_sigma0
    pc = np.zeros(dim); ps = np.zeros(dim)
    Cdiag = np.ones(dim); D = np.sqrt(Cdiag)

    def ask():
        Z = rng.standard_normal(size=(pop_size, dim))
        X = m + sigma * Z * D
        X = _mirror_unit(X)
        return X, Z

    def tell(X, Z, fit):
        nonlocal m, sigma, pc, ps, Cdiag, D
        idx = np.argsort(-fit)  # maximize
        Xsel, Zsel = X[idx], Z[idx]
        w_col = w.reshape(-1,1)
        m_old = m.copy()
        m = (w_col * Xsel).sum(axis=0)

        y = (m - m_old) / sigma
        c_sigma = math.sqrt(cs*(2 - cs)*mu_eff)
        ps = (1 - cs)*ps + c_sigma * (y / D)
        hsig = 1.0 if (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**2)) < (1.4 + 2/(dim + 1)) else 0.0
        pc = (1 - cc)*pc + hsig * math.sqrt(cc*(2 - cc)*mu_eff) * y

        artmp = (Zsel * w_col).sum(axis=0)
        Cdiag = (1 - c1 - cmu)*Cdiag + c1*(pc**2) + cmu*(artmp**2)
        sigma *= math.exp((cs/damps)*(np.linalg.norm(ps)/math.sqrt(dim) - 1))
        D = np.sqrt(np.maximum(Cdiag, 1e-16))

    def z_to_phys(z):
        C1 = _minmax_denorm(z[0], *cfg.bounds_phys["C1"])
        C2 = _minmax_denorm(z[1], *cfg.bounds_phys["C2"])
        C3 = _minmax_denorm(z[2], *cfg.bounds_phys["C3"])
        return C1, C2, C3

    best = {"z": None, "Q6": -1e99, "Q6_pen": -1e99}
    hist: List[Tuple[int,float,float]] = []
    archive: List[List[float]] = []  # [C1,C2,C3,Q6,Q6_pen,fitness]

    for it in range(cfg.cma_iters):
        Xz, Z = ask()
        fit = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            C1, C2, C3 = z_to_phys(Xz[i])
            Q6, Q6pen = _eval_Q6_and_pen(C1, C2, C3, cfg)
            f = Q6 if cfg.objective=="raw" else Q6pen
            archive.append([C1, C2, C3, Q6, Q6pen, f])

            if Q6 > best["Q6"]:
                best.update({"z": Xz[i].copy(), "Q6": Q6, "Q6_pen": Q6pen})
            fit[i] = f
        tell(Xz, Z, fit)
        hist.append((it+1, best["Q6"], best["Q6_pen"]))

    # 去重 + 取 Top-K
    # 以配方四舍五入到 6 位作为 key，避免重复
    seen = set()
    uniq = []
    for row in sorted(archive, key=lambda r: r[5], reverse=True):
        key = (round(row[0],6), round(row[1],6), round(row[2],6))
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(row)
        if len(uniq) >= cfg.topk:
            break

    z = best["z"]
    C1b, C2b, C3b = _minmax_denorm(z[0], *cfg.bounds_phys["C1"]), _minmax_denorm(z[1], *cfg.bounds_phys["C2"]), _minmax_denorm(z[2], *cfg.bounds_phys["C3"])
    best_dict = {"C1":C1b, "C2":C2b, "C3":C3b, "Q6":best["Q6"], "Q6_pen":best["Q6_pen"]}
    return best_dict, hist, uniq

# --------- Utility: best curve with fine RK4 for plotting ---------
def _curve_Q(C1: float, C2: float, C3: float, cfg: OptimizeConfig, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    C1n = _minmax_norm(C1, *cfg.bounds_phys["C1"])
    C2n = _minmax_norm(C2, *cfg.bounds_phys["C2"])
    C3n = _minmax_norm(C3, *cfg.bounds_phys["C3"])
    n = int(math.ceil(cfg.T / dt))
    ts = np.linspace(0.0, cfg.T, n+1)
    qtilde = 0.0
    Q = np.empty(n+1, dtype=float); Q[0] = 0.0
    for k in range(n):
        k1 = _dQtilde_dt(qtilde, C1n, C2n, C3n)
        k2 = _dQtilde_dt(qtilde + 0.5*dt*k1, C1n, C2n, C3n)
        k3 = _dQtilde_dt(qtilde + 0.5*dt*k2, C1n, C2n, C3n)
        k4 = _dQtilde_dt(qtilde + dt*k3, C1n, C2n, C3n)
        qtilde = qtilde + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if qtilde < 0: qtilde = 0.0
        Q[k+1] = cfg.q_scale * qtilde
    return ts, Q

# --------- Public API ---------
def optimize(cfg: OptimizeConfig) -> Dict:
    """
    Return:
      'best': dict, 'history': list, 'best_curve': (t,Q), 'topk': list[list], 'config': dict
    """
    if cfg.method == "de":
        best, topk, hist = _de_optimize(cfg)
    else:
        best, hist, topk = _cma_es_diagonal(cfg)  # ← 现在 CMA 也有 topk

    t, Q = _curve_Q(best["C1"], best["C2"], best["C3"], cfg, dt=0.01)
    best["fitness"] = best["Q6"] if cfg.objective=="raw" else best["Q6_pen"]
    return {
        "best": best,
        "history": hist,
        "best_curve": (t, Q),
        "topk": topk,  # 现在两种方法都给前 cfg.topk 条
        "config": asdict(cfg),
    }

# === 在文件末尾 optimize(...) 之后，__main__ 之前，新增这两段 ===

def eval_Q6(C1: float, C2: float, C3: float, cfg: OptimizeConfig, return_penalized: bool = True):
    """
    公开的评估函数：给定配方，返回 (Q6, Q6_pen) 或仅 Q6。
    用于灵敏度分析/响应面绘图。
    """
    Q6, Q6_pen = _eval_Q6_and_pen(C1, C2, C3, cfg)
    return (Q6, Q6_pen) if return_penalized else Q6

def curve_Q(C1: float, C2: float, C3: float, cfg: OptimizeConfig, dt: float = 0.01):
    """
    公开的曲线函数：返回 (t, Q(t))。内部沿用 _curve_Q（RK4 细采样，仅用于可视化）。
    如需 RK45 单程积分也可以再加一版。
    """
    return _curve_Q(C1, C2, C3, cfg, dt=dt)

# # Optional: a quick demo if you just run this file
# if __name__ == "__main__":
#     cfg = OptimizeConfig(method="cma", objective="softcap", cap=3500.0)
#     res = optimize(cfg)
#     print("[Demo] Best:", res["best"])
