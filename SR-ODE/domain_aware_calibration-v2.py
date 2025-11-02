# pirsr_calibrate_ONLY_a_with_optional_micro_coref.py
# -----------------------------------------------------------
# EXACT model (fixed structure):
#   Qhat(t,c) = S * K_form(c; a, gamma) * K_time(t)
#   K_form = 1/((a - c_pol)*c_pg) + sqrt(c_pg*c_eth) + gamma * (c_pg*c_eth)
#   K_time(t) = t - sqrt(t) + exp(-t)   (FIXED)
#
# CURRENT METHOD:
#   - Default: calibrate ONLY the risky constant 'a' via domain-aware mapping
#              outside [20,30]; S and gamma are FROZEN at their initial values.
#   - Optional (disabled by default): tiny co-calibration of S,gamma within
#              +/-5% box around initials (CO_CALIBRATE=True).
#
#   Structure preserved. Try both 'lower' and 'upper' sides for 'a', pick by CV RMSE.
#   Optimisers: Differential Evolution (global) -> L-BFGS-B (local).
# -----------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import KFold

# ====== Data loading ======
USE_EXCEL = True
EXCEL_PATH = r"Symbolic Regression/data/Raw_IVPT_thirty.xlsx"
EXCEL_SHEET_X = "Formulas-train"  # first col is index to drop
EXCEL_SHEET_Y = "C-train"         # first col is index to drop

# ====== Domain for future optimisation (given) ======
DOMAIN_BOUNDS = {"c_pol": (20.0, 30.0), "c_eth": (10.0, 20.0), "c_pg": (10.0, 20.0)}
VAR_INDEX = {"c_pol": 0, "c_eth": 1, "c_pg": 2}
SAFETY_MARGIN = 0.0  # epsilon/tolerance; putting pole strictly outside is enough

# ====== Optional robustness penalty (OFF by default) ======
ROBUST_LAMBDA = 0.0
ROBUST_P = 2
ROBUST_SAMPLES = 600

# ====== Numerics ======
EPS_DENOM = 1e-8
DE_MAXITER = 80

# ====== Initial constants (FROZEN by default) ======
S_INIT = 2.053
GAMMA_INIT = -0.039
A_INIT = 24.897  # only for logging

# ====== Method switches ======
# Default method == ONLY 'a'. Set to True to allow S,gamma to micro-adjust (±5%).
CO_CALIBRATE = False
CO_BOX_RATIO = 0.05  # ±5%

# ====== Helpers & data structures ======
@dataclass
class Domain:
    c_pol: Tuple[float, float]
    c_eth: Tuple[float, float]
    c_pg:  Tuple[float, float]
    def bounds_of(self, coord: str) -> Tuple[float, float]:
        low, high = getattr(self, coord)
        return float(low), float(high)
    def sample(self, n: int, seed: int = 17) -> np.ndarray:
        rng = np.random.default_rng(seed)
        def stratified(low, high):
            bins = np.linspace(0, 1, n + 1)
            u = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(u)
            return low + (high - low) * u
        cols = []
        for name in ["c_pol", "c_eth", "c_pg"]:
            low, high = self.bounds_of(name)
            cols.append(stratified(low, high))
        return np.stack(cols, axis=1)

def rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y)**2)))

def kfold_cv_error(times: np.ndarray, C: np.ndarray, Y: np.ndarray,
                   params: Dict[str, float], n_splits: int = 4) -> float:
    kf = KFold(n_splits=min(n_splits, len(C)), shuffle=True, random_state=0)
    yhat = predict_Q(times, C, params)  # global params (no refit per fold)
    errs = []
    for _, test_idx in kf.split(C):
        errs.append(rmse(yhat[test_idx], Y[test_idx]))
    return float(np.mean(errs))

# ====== Fixed model pieces ======
def denom_with_sign(a: float, c_pol: np.ndarray, c_pg: np.ndarray) -> np.ndarray:
    d = (a - c_pol) * c_pg
    return d + EPS_DENOM * np.sign(d + 1e-16)

def K_time(times: np.ndarray) -> np.ndarray:
    t = times.reshape(1, -1)
    return t - np.sqrt(np.clip(t, 0.0, None)) + np.exp(-t)  # (1,T)

def K_form_pieces(C: np.ndarray, a: float):
    """
    Return components independent of gamma scaling of c_pg*c_eth:
    F0 = 1/((a - c_pol)*c_pg) + sqrt(c_pg*c_eth)   (N,1)
    Fprod = (c_pg*c_eth)                            (N,1)  (coefficient = gamma)
    """
    c_pol = C[:, [VAR_INDEX["c_pol"]]]
    c_eth = C[:, [VAR_INDEX["c_eth"]]]
    c_pg  = C[:, [VAR_INDEX["c_pg"]]]
    denom = denom_with_sign(a, c_pol, c_pg)
    inv_term  = 1.0 / denom
    sqrt_term = np.sqrt(np.clip(c_pg * c_eth, 0.0, None))
    F0 = inv_term + sqrt_term
    Fprod = c_pg * c_eth
    return F0, Fprod  # both (N,1)

def predict_Q(times: np.ndarray, C: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Q_hat = S * (F0 + gamma*Fprod) @ K_time
    """
    a = params["a"]
    S = params["S"]
    gamma = params["gamma"]
    F0, Fprod = K_form_pieces(C, a)     # (N,1),(N,1)
    F = F0 + gamma * Fprod              # (N,1)
    Tm = K_time(times)                  # (1,T)
    return S * (F @ Tm)                 # (N,T)

# ====== a mapping (one-sided, unconstrained beta) ======
def make_a_from_beta(side: str, domain: Domain, m: float):
    pL, pU = domain.c_pol
    def a_of(beta: float) -> float:
        v = np.exp(beta)  # >0
        if side == "lower":
            return pL - m - v
        elif side == "upper":
            return pU + m + v
        else:
            raise ValueError("side must be 'lower' or 'upper'")
    return a_of

def domain_risk_metric(Cs: np.ndarray, a: float) -> np.ndarray:
    c_pol = Cs[:, VAR_INDEX["c_pol"]]
    c_pg  = Cs[:, VAR_INDEX["c_pg"]]
    g = np.abs((a - c_pol) * c_pg)
    return 1.0 / (g + 1e-12)

# ====== Loss builders ======
def make_loss_only_a(side: str, domain: Domain, m: float,
                     times: np.ndarray, C: np.ndarray, Y: np.ndarray):
    """Loss(beta) with S,gamma frozen at initials."""
    a_map = make_a_from_beta(side, domain, m)
    Cs_rob = domain.sample(ROBUST_SAMPLES, seed=23) if ROBUST_LAMBDA > 0.0 else None

    def loss(z: np.ndarray) -> float:
        beta = float(z[0])
        a = a_map(beta)
        params = {"a": a, "S": S_INIT, "gamma": GAMMA_INIT}
        Yhat = predict_Q(times, C, params)
        L_fit = np.mean((Yhat - Y)**2)
        L = L_fit
        if ROBUST_LAMBDA > 0.0 and Cs_rob is not None:
            r = domain_risk_metric(Cs_rob, a)
            L += ROBUST_LAMBDA * float(np.mean(r**ROBUST_P))
        return float(L)

    def unpack_params(z_best: np.ndarray) -> Dict[str,float]:
        beta = float(z_best[0])
        a = a_map(beta)
        return {"a": a, "S": S_INIT, "gamma": GAMMA_INIT}

    return loss, unpack_params

def make_loss_with_micro_coref(side: str, domain: Domain, m: float,
                               times: np.ndarray, C: np.ndarray, Y: np.ndarray):
    """
    Loss(beta, S, gamma) with narrow box constraints around S_INIT, GAMMA_INIT.
    """
    a_map = make_a_from_beta(side, domain, m)
    Cs_rob = domain.sample(ROBUST_SAMPLES, seed=23) if ROBUST_LAMBDA > 0.0 else None

    def loss(z: np.ndarray) -> float:
        beta, S, gamma = float(z[0]), float(z[1]), float(z[2])
        a = a_map(beta)
        params = {"a": a, "S": S, "gamma": gamma}
        Yhat = predict_Q(times, C, params)
        L_fit = np.mean((Yhat - Y)**2)
        L = L_fit
        if ROBUST_LAMBDA > 0.0 and Cs_rob is not None:
            r = domain_risk_metric(Cs_rob, a)
            L += ROBUST_LAMBDA * float(np.mean(r**ROBUST_P))
        return float(L)

    def unpack_params(z_best: np.ndarray) -> Dict[str,float]:
        beta, S, gamma = float(z_best[0]), float(z_best[1]), float(z_best[2])
        a = a_map(beta)
        return {"a": a, "S": S, "gamma": gamma}

    # bounds for (beta, S, gamma)
    S_lo, S_hi = S_INIT * (1 - CO_BOX_RATIO), S_INIT * (1 + CO_BOX_RATIO)
    G_lo, G_hi = GAMMA_INIT * (1 - CO_BOX_RATIO), GAMMA_INIT * (1 + CO_BOX_RATIO)
    bounds = [(-3.0, 3.0), (S_lo, S_hi), (G_lo, G_hi)]
    return loss, unpack_params, bounds

# ====== Fit one side ======
def fit_side(side: str, times: np.ndarray, C: np.ndarray, Y: np.ndarray,
             domain: Domain, m: float, seed: int = 0):
    if not CO_CALIBRATE:
        loss_fn, unpack = make_loss_only_a(side, domain, m, times, C, Y)
        bounds = [(-3.0, 3.0)]  # beta in [-3,3] -> exp ~ [0.05, 20]
        de = differential_evolution(loss_fn, bounds=bounds, maxiter=DE_MAXITER, seed=seed, polish=True)
        x0 = de.x
        res = minimize(loss_fn, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        zbest = res.x
        params = unpack(zbest)
    else:
        loss_fn, unpack, bounds = make_loss_with_micro_coref(side, domain, m, times, C, Y)
        de = differential_evolution(loss_fn, bounds=bounds, maxiter=DE_MAXITER, seed=seed, polish=True)
        x0 = de.x
        res = minimize(loss_fn, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        zbest = res.x
        params = unpack(zbest)

    # metrics
    Yhat = predict_Q(times, C, params)
    rmse_fit = rmse(Yhat, Y)
    rmse_cv  = kfold_cv_error(times, C, Y, params, n_splits=4)
    pL, pU = domain.c_pol
    min_dist = min(abs(params["a"] - pL), abs(params["a"] - pU))
    return {
        "side": side,
        "params": params,
        "rmse_fit": rmse_fit,
        "rmse_cv": rmse_cv,
        "min_dist": float(min_dist)
    }

# ====== Data ======
def load_data():
    if USE_EXCEL:
        import pandas as pd
        df_X = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET_X)
        df_Y = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET_Y)
        C = df_X.iloc[:, 1:].to_numpy(dtype=float)[:, :3]   # [c_pol,c_eth,c_pg]
        Y = df_Y.iloc[:, 1:].to_numpy(dtype=float)          # (N,T)
        times = np.array([1,2,3,4,6,8,22,24,26,28], dtype=float)
        return times, C, Y
    else:
        times = np.array([1,2,3,4,6,8,22,24,26,28], dtype=float)
        C = np.array([
            [21.17, 19.95, 10.59],
            [25.22, 19.41, 19.29],
            [26.06, 12.00, 15.70],
            [20.95, 19.44, 12.14],
        ], dtype=float)
        Y = np.array([
            [3.42, 9.44, 17.23, 25.97, 42.82, 62.45, 198.53, 218.18, 239.48, 260.32],
            [2.00, 6.31, 12.27, 18.75, 31.44, 45.94, 152.98, 168.31, 184.27, 200.64],
            [2.38, 9.06, 16.96, 25.05, 42.85, 61.64, 201.93, 223.23, 244.23, 264.71],
            [2.37, 8.52, 16.39, 26.70, 47.48, 70.57, 247.10, 272.06, 298.10, 328.48],
        ], dtype=float)
        return times, C, Y

# ====== Main ======
def main():
    print("=== Domain-aware calibration: ONLY 'a' (S,gamma frozen) ===")
    domain = Domain(**DOMAIN_BOUNDS)
    times, C, Y = load_data()
    print(f"Domain: c_pol in {domain.c_pol}, c_eth in {domain.c_eth}, c_pg in {domain.c_pg}")
    print(f"Safety margin epsilon = {SAFETY_MARGIN}")
    print(f"Initials: a0={A_INIT}, S0={S_INIT}, gamma0={GAMMA_INIT}")
    print(f"Co-calibration (S,gamma): {CO_CALIBRATE} (box ±{int(CO_BOX_RATIO*100)}%)")

    res_lower = fit_side("lower", times, C, Y, domain, SAFETY_MARGIN, seed=0)
    res_upper = fit_side("upper", times, C, Y, domain, SAFETY_MARGIN, seed=1)

    best = res_lower if res_lower["rmse_cv"] <= res_upper["rmse_cv"] else res_upper
    other = res_upper if best is res_lower else res_lower

    def show(tag, r):
        p = r["params"]
        Yhat = predict_Q(times, C, p)
        ss_res = np.sum((Y - Yhat) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
        print(f"\n--- {tag} ---")
        print(f"side: {r['side']}")
        print(f"R^2(fit): {r2_fit:.4f} | RMSE(fit): {r['rmse_fit']:.4f} | RMSE(CV): {r['rmse_cv']:.4f}")
        print(f"a*: {p['a']:.6f} | S: {p['S']:.6f} | gamma: {p['gamma']:.6f} | min_dist(a->edges): {r['min_dist']:.3f}")

    show("Lower-side", res_lower)
    show("Upper-side", res_upper)

    print("\n=== SELECTED ===")
    show("Best", best)
    print("\nOther side printed above for comparison.")

if __name__ == "__main__":
    main()
