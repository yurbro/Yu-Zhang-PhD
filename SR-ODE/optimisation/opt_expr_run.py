# -*- coding: utf-8 -*-
"""
Optimization for dQ/dt expression under training scale
------------------------------------------------------
- Cs (C1,C2,C3) are min-max normalized to [0,1] before entering the model
- Q is integrated in training scale (Qtilde) then mapped back with q_scale
- Integrator: RK45 (Dormand–Prince 5(4)) adaptive step
- Optimizers: CMA-ES (diagonal) or Differential Evolution (DE)
- Objectives:
    - raw:      maximize Q(6h)
    - softcap:  maximize Q(6h) - max(0, Q(6h)-cap)^2

Usage examples
--------------
# CMA-ES, soft cap at 3500
python opt_expr_run.py --method cma --objective softcap --cap 3500

# Differential Evolution, raw objective, more iters
python opt_expr_run.py --method de --objective raw --pop 60 --gens 80

Author: for Yu
"""
import math, json, argparse, os, csv, sys, time
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

# ---------- Config (edit if needed) ----------
# Physical design-space bounds
BOUNDS_PHYS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}

# Expression constant
B_CONST = 1.5346103937034676  # denominator width

# Default q_scale (will be overridden if --cfg has q_scale)
Q_SCALE_DEFAULT = 3008.198194823261

# ---------- Helpers ----------
def minmax_norm(x: float, lo: float, hi: float) -> float:
    return (x - lo) / (hi - lo)

def minmax_denorm(xn: float, lo: float, hi: float) -> float:
    return xn * (hi - lo) + lo

def softplus_scalar(x: float) -> float:
    # numerically stable softplus
    if x > 20.0: return x
    if x < -20.0: return math.exp(x)
    return math.log1p(math.exp(x))

# ---------- Model: dQtilde/dt with normalized Cs ----------
def dQtilde_dt(Qtilde: float, C1n: float, C2n: float, C3n: float) -> float:
    # guard for division by zero via epsilon in C3n
    eps = 1e-9
    denom = (C2n - ((C1n / (C3n + eps)) - (Qtilde**2)))**2 + (B_CONST**2)
    return softplus_scalar(2.0 * Qtilde) / denom

# ---------- RK45 Integrator (Dormand–Prince 5(4)) ----------
def integrate_Qtilde_RK45(
    C1n: float, C2n: float, C3n: float,
    T: float = 6.0,
    rtol: float = 1e-6, atol: float = 1e-9,
    h0: float = 0.05, h_min: float = 1e-4, h_max: float = 0.25,
    max_steps: int = 200000
) -> float:
    t, q, h = 0.0, 0.0, h0
    safety = 0.9
    steps = 0
    while t < T and steps < max_steps:
        if t + h > T:
            h = T - t
        k1 = dQtilde_dt(q, C1n, C2n, C3n)
        k2 = dQtilde_dt(q + h*0.25*k1, C1n, C2n, C3n)
        k3 = dQtilde_dt(q + h*(3/32*k1 + 9/32*k2), C1n, C2n, C3n)
        k4 = dQtilde_dt(q + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3), C1n, C2n, C3n)
        k5 = dQtilde_dt(q + h*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4), C1n, C2n, C3n)
        k6 = dQtilde_dt(q + h*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5), C1n, C2n, C3n)
        # 5th-order
        q5 = q + h*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
        # 4th-order
        q4 = q + h*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
        err = abs(q5 - q4)
        tol = atol + rtol*max(abs(q), abs(q5))
        if err <= tol or h <= h_min*1.01:
            q = q5
            t += h
        if err == 0.0:
            s = 2.0
        else:
            s = safety * (tol / err)**0.2  # order=4
        h = min(max(h*s, h_min), h_max)
        steps += 1
    return q

# ---------- Objectives ----------
def eval_Q6_and_penalized(C1: float, C2: float, C3: float, q_scale: float, T: float, cap: float) -> Tuple[float, float]:
    # map physical Cs to normalized [0,1]
    C1n = minmax_norm(C1, *BOUNDS_PHYS["C1"])
    C2n = minmax_norm(C2, *BOUNDS_PHYS["C2"])
    C3n = minmax_norm(C3, *BOUNDS_PHYS["C3"])
    Qtilde_T = integrate_Qtilde_RK45(C1n, C2n, C3n, T=T)
    Q_T = q_scale * Qtilde_T
    exceed = max(0.0, Q_T - cap)
    penalized = Q_T - exceed**2
    return Q_T, penalized

# ---------- Differential Evolution (DE) ----------
def de_optimize(q_scale: float, T: float, objective: str,
                pop_size: int = 50, gens: int = 60, F: float = 0.7, CR: float = 0.9,
                cap: float = 3500.0, seed: int = 20251008):
    rng = np.random.default_rng(seed)
    # init population
    pop = []
    for _ in range(pop_size):
        C1 = rng.uniform(*BOUNDS_PHYS["C1"])
        C2 = rng.uniform(*BOUNDS_PHYS["C2"])
        C3 = rng.uniform(*BOUNDS_PHYS["C3"])
        Q6, Q6pen = eval_Q6_and_penalized(C1, C2, C3, q_scale, T, cap)
        fit = Q6 if objective == "raw" else Q6pen
        pop.append([C1, C2, C3, Q6, Q6pen, fit])

    best = max(pop, key=lambda x: x[5]).copy()
    keys = list(BOUNDS_PHYS.keys())

    for _ in range(gens):
        for i in range(pop_size):
            idxs = list(range(pop_size)); idxs.remove(i)
            a, b, c = rng.choice(idxs, size=3, replace=False)
            A, B, C = pop[a], pop[b], pop[c]
            v = [
                A[0] + F*(B[0] - C[0]),
                A[1] + F*(B[1] - C[1]),
                A[2] + F*(B[2] - C[2]),
            ]
            # binomial crossover with clipping
            jrand = rng.integers(3)
            trial = [0.0, 0.0, 0.0]
            for j, k in enumerate(keys):
                if rng.random() < CR or j == jrand:
                    lo, hi = BOUNDS_PHYS[k]
                    trial[j] = float(np.clip(v[j], lo, hi))
                else:
                    trial[j] = pop[i][j]
            Q6, Q6pen = eval_Q6_and_penalized(trial[0], trial[1], trial[2], q_scale, T, cap)
            fit = Q6 if objective == "raw" else Q6pen
            if fit >= pop[i][5]:
                pop[i] = [trial[0], trial[1], trial[2], Q6, Q6pen, fit]
                if fit >= best[5]:
                    best = pop[i].copy()

    pop.sort(key=lambda x: x[5], reverse=True)
    return best, pop

# ---------- Diagonal CMA-ES ----------
def mirror_into_unit(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = np.where(x < 0, -x, x)
    x = np.where(x > 1, 2 - x, x)
    x = np.mod(x, 2.0)
    x = np.where(x > 1, 2 - x, x)
    return x

def cma_es_diagonal(q_scale: float, T: float, objective: str,
                    max_iters: int = 60, pop_size: int = None, sigma0: float = 0.2,
                    cap: float = 3500.0, seed: int = 20251008):
    rng = np.random.default_rng(seed)
    dim = 3
    if pop_size is None:
        pop_size = 4 + 3*dim  # ~13
    w = np.array([math.log(pop_size + 0.5) - math.log(i + 1) for i in range(pop_size)])
    w = w / w.sum()
    mu_eff = 1.0 / np.sum(w**2)

    cc = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2*(mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
    damps = 1 + 2*max(0, math.sqrt((mu_eff - 1)/(dim + 1)) - 1) + cs

    m = np.array([0.5, 0.5, 0.5])  # start from center (normalized)
    sigma = sigma0
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    Cdiag = np.ones(dim)
    D = np.sqrt(Cdiag)

    def ask():
        Z = rng.standard_normal(size=(pop_size, dim))
        X = m + sigma * Z * D
        X = mirror_into_unit(X)
        return X, Z

    def tell(X, Z, fit):
        nonlocal m, sigma, pc, ps, Cdiag, D
        idx = np.argsort(-fit)  # maximize
        Xsel, Zsel = X[idx], Z[idx]
        w_col = w.reshape(-1, 1)
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
        return m, sigma

    def z_to_phys(z):
        C1 = minmax_denorm(z[0], *BOUNDS_PHYS["C1"])
        C2 = minmax_denorm(z[1], *BOUNDS_PHYS["C2"])
        C3 = minmax_denorm(z[2], *BOUNDS_PHYS["C3"])
        return C1, C2, C3

    best = {"z": None, "Q6": -1e99, "Q6pen": -1e99}
    history = []

    for it in range(max_iters):
        Xz, Z = ask()
        fit = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            C1, C2, C3 = z_to_phys(Xz[i])
            Q6, Q6pen = eval_Q6_and_penalized(C1, C2, C3, q_scale, T, cap)
            if Q6 > best["Q6"]:
                best.update({"z": Xz[i].copy(), "Q6": Q6, "Q6pen": Q6pen})
            fit[i] = Q6 if objective == "raw" else Q6pen
        tell(Xz, Z, fit)
        history.append([it, best["Q6"], best["Q6pen"]])

    z = best["z"]
    C1b, C2b, C3b = minmax_denorm(z[0], *BOUNDS_PHYS["C1"]), minmax_denorm(z[1], *BOUNDS_PHYS["C2"]), minmax_denorm(z[2], *BOUNDS_PHYS["C3"])
    return {"best": {"C1": C1b, "C2": C2b, "C3": C3b, "Q6": best["Q6"], "Q6_pen": best["Q6pen"]},
            "history": history}

# ---------- Curve utility (fine RK4 for plotting/CSV) ----------
def curve_Q(C1: float, C2: float, C3: float, q_scale: float, T: float = 6.0, dt: float = 0.01):
    # simple RK4 for dense sampling (post-process only)
    C1n = minmax_norm(C1, *BOUNDS_PHYS["C1"])
    C2n = minmax_norm(C2, *BOUNDS_PHYS["C2"])
    C3n = minmax_norm(C3, *BOUNDS_PHYS["C3"])
    n = int(math.ceil(T/dt))
    ts = np.linspace(0.0, T, n+1)
    qtilde = 0.0
    Q = np.empty(n+1, dtype=float)
    Q[0] = 0.0
    for k in range(n):
        k1 = dQtilde_dt(qtilde, C1n, C2n, C3n)
        k2 = dQtilde_dt(qtilde + 0.5*dt*k1, C1n, C2n, C3n)
        k3 = dQtilde_dt(qtilde + 0.5*dt*k2, C1n, C2n, C3n)
        k4 = dQtilde_dt(qtilde + dt*k3, C1n, C2n, C3n)
        qtilde = qtilde + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if qtilde < 0: qtilde = 0.0
        Q[k+1] = q_scale * qtilde
    return ts, Q

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="", help="Path to cfg JSON containing q_scale (optional).")
    ap.add_argument("--method", type=str, default="cma", choices=["cma","de"], help="Optimizer")
    ap.add_argument("--objective", type=str, default="raw", choices=["raw","softcap"], help="Objective type")
    ap.add_argument("--cap", type=float, default=3500.0, help="Soft cap for Q(6h) when objective=softcap")
    ap.add_argument("--T", type=float, default=6.0, help="Time horizon (hours)")
    ap.add_argument("--seed", type=int, default=20251008, help="Random seed")
    # CMA
    ap.add_argument("--iters", type=int, default=60, help="CMA-ES iterations")
    ap.add_argument("--pop", type=int, default=13, help="CMA-ES population (default ~13)")
    ap.add_argument("--sigma0", type=float, default=0.2, help="CMA-ES initial sigma in normalized space")
    # DE
    ap.add_argument("--gens", type=int, default=60, help="DE generations")
    ap.add_argument("--depop", type=int, default=50, help="DE population size")
    ap.add_argument("--F", type=float, default=0.7, help="DE differential weight")
    ap.add_argument("--CR", type=float, default=0.9, help="DE crossover rate")
    ap.add_argument("--outdir", type=str, default="out_opt", help="Output directory")
    args = ap.parse_args()

    # q_scale
    q_scale = Q_SCALE_DEFAULT
    if args.cfg and os.path.isfile(args.cfg):
        try:
            cfg = json.loads(open(args.cfg, "r", encoding="utf-8").read())
            if isinstance(cfg, dict) and "q_scale" in cfg:
                q_scale = float(cfg["q_scale"])
        except Exception as e:
            print(f"[WARN] Failed to read cfg: {e}. Using default q_scale={q_scale}")

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    if args.method == "de":
        best, pop = de_optimize(q_scale, args.T, args.objective,
                                pop_size=args.depop, gens=args.gens, F=args.F, CR=args.CR,
                                cap=args.cap, seed=args.seed)
        C1b, C2b, C3b, Q6, Q6pen, fit = best
        print(f"[DE] Best: C1={C1b:.6f}, C2={C2b:.6f}, C3={C3b:.6f}, Q6={Q6:.6f}, Q6_pen={Q6pen:.6f}")

        # Save top-20
        top20 = sorted(pop, key=lambda x: x[5], reverse=True)[:20]
        path_top = os.path.join(args.outdir, "DE_top20.csv")
        with open(path_top, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["C1","C2","C3","Q6","Q6_pen","fitness"])
            for row in top20: w.writerow(row)

        # Save best curve
        ts, Q = curve_Q(C1b, C2b, C3b, q_scale, T=args.T, dt=0.01)
        path_curve = os.path.join(args.outdir, "DE_best_curve.csv")
        with open(path_curve, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["t_h","Q"]); w.writerows(zip(ts,Q))

    else:
        res = cma_es_diagonal(q_scale, args.T, args.objective,
                              max_iters=args.iters, pop_size=args.pop, sigma0=args.sigma0,
                              cap=args.cap, seed=args.seed)
        best = res["best"]; hist = res["history"]
        print(f"[CMA] Best: C1={best['C1']:.6f}, C2={best['C2']:.6f}, C3={best['C3']:.6f}, "
              f"Q6={best['Q6']:.6f}, Q6_pen={best['Q6_pen']:.6f}")

        # Save history
        path_hist = os.path.join(args.outdir, "CMA_history.csv")
        with open(path_hist, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["iter","best_raw","best_pen"]); w.writerows(hist)

        # Save best curve
        ts, Q = curve_Q(best["C1"], best["C2"], best["C3"], q_scale, T=args.T, dt=0.01)
        path_curve = os.path.join(args.outdir, "CMA_best_curve.csv")
        with open(path_curve, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["t_h","Q"]); w.writerows(zip(ts,Q))

    print(f"Done in {time.time()-t0:.2f}s. Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
