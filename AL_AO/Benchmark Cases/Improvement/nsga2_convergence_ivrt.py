from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Benchmark" / "Improvement"
DATA_PATH = ROOT / "Dataset" / "IVRT-S.xlsx"
PARAM_PATH = ROOT / "Dataset" / "fulldata-s" / "Bayesian_Optimisation_Results_fulldata.xlsx"


class YuKernel(Kernel):
    def __init__(self, v0, wl, a0, a1, v1):
        self.v0 = v0
        self.wl = np.atleast_1d(wl)
        self.a0 = a0
        self.a1 = a1
        self.v1 = v1

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X

        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        if self.wl.size != X.shape[-1]:
            raise ValueError("wl size must match the number of features")

        exp_term = np.sum(
            ((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2) / self.wl,
            axis=2,
        )
        exp_term = self.v0 * np.exp(-0.5 * exp_term)
        linear_term = self.a0 + self.a1 * np.dot(X, Y.T)
        noise_term = self.v1 * np.eye(X.shape[0]) if X is Y else np.zeros((X.shape[0], Y.shape[0]))
        return exp_term + linear_term + noise_term

    def diag(self, X):
        X = np.atleast_2d(X)
        return self.v0 + self.a0 + self.a1 * np.sum(X**2, axis=1) + self.v1

    def is_stationary(self):
        return False

    def __repr__(self):
        return (
            f"YuKernel(v0={self.v0}, wl={self.wl}, a0={self.a0}, "
            f"a1={self.a1}, v1={self.v1})"
        )


class IVRTProblem(ElementwiseProblem):
    def __init__(self, gpr, scaler_y, xl, xu):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=xl, xu=xu)
        self.gpr = gpr
        self.scaler_y = scaler_y

    def _evaluate(self, x, out, *args, **kwargs):
        y_pred_scaled, y_std = self.gpr.predict([x], return_std=True)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
        y_std = np.atleast_1d(y_std).ravel()[-1]
        out["F"] = np.array([-y_pred, -y_std])


class PopulationHistory(Callback):
    def __init__(self):
        super().__init__()
        self.F = []

    def notify(self, algorithm):
        self.F.append(algorithm.pop.get("F").copy())


def non_dominated(F):
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[nd_idx]


def load_initial_ivrt_model():
    formulas = pd.read_excel(DATA_PATH, sheet_name="F-S")
    responses = pd.read_excel(DATA_PATH, sheet_name="R-S")

    X = formulas.iloc[:, 1:].to_numpy(dtype=float)
    y = responses.iloc[:, -1].to_numpy(dtype=float)

    params = pd.read_excel(PARAM_PATH).iloc[0].to_dict()
    wl = [params[f"wl{i}"] for i in range(1, X.shape[1] + 1)]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = YuKernel(params["v0"], wl, params["a0"], params["a1"], params["v1"])
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10)
    gpr.fit(X_scaled, y_scaled)

    x_lower = np.array([20.0, 10.0, 10.0])
    x_upper = np.array([30.0, 20.0, 20.0])
    xl = scaler_X.transform([x_lower])[0]
    xu = scaler_X.transform([x_upper])[0]

    return gpr, scaler_y, xl, xu


def compute_hv_trace(populations):
    all_F = np.vstack(populations)
    span = np.ptp(all_F, axis=0)
    span[span == 0.0] = 1.0
    ref_point = np.max(all_F, axis=0) + 0.05 * span
    hv_indicator = HV(ref_point=ref_point)
    hv_values = np.array([hv_indicator.do(non_dominated(F)) for F in populations])
    return hv_values, ref_point


def plot_hv(hv_values, convergence_generation):
    generations = np.arange(1, len(hv_values) + 1)
    final_hv = hv_values[-1]
    threshold = 0.99 * final_hv

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(generations, hv_values, color="#1f77b4", linewidth=2.2)
    plt.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.4, label="99% of gen-500 HV")
    plt.axvline(convergence_generation, color="#2ca02c", linestyle=":", linewidth=1.6, label=f"Gen {convergence_generation}")
    plt.xlabel("NSGA-II generation")
    plt.ylabel("Hypervolume")
    # plt.title("NSGA-II Hypervolume Convergence on Initial IVRT GPR")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_nsga2_convergence.png", dpi=400, bbox_inches="tight")
    plt.close()


def plot_pareto_compare(F100, F500):
    nd100 = non_dominated(F100)
    nd500 = non_dominated(F500)
    front100 = -nd100
    front500 = -nd500

    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(
        front100[:, 0],
        front100[:, 1],
        s=44,
        facecolors="none",
        edgecolors="#ff7f0e",
        linewidths=1.4,
        label="Generation 100",
    )
    plt.scatter(
        front500[:, 0],
        front500[:, 1],
        s=28,
        color="#1f77b4",
        alpha=0.75,
        label="Generation 500",
    )
    plt.xlabel("Predicted mean")
    plt.ylabel("Predictive standard deviation")
    # plt.title("NSGA-II Pareto Front: 100 vs 500 Generations")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_nsga2_pareto_compare.png", dpi=400, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpr, scaler_y, xl, xu = load_initial_ivrt_model()

    problem = IVRTProblem(gpr, scaler_y, xl, xu)
    history = PopulationHistory()
    algorithm = NSGA2(pop_size=50)
    minimize(
        problem,
        algorithm,
        termination=("n_gen", 500),
        callback=history,
        verbose=False,
        seed=1,
    )

    hv_values, ref_point = compute_hv_trace(history.F)
    final_hv = hv_values[-1]
    convergence_generation = int(np.argmax(hv_values >= 0.99 * final_hv) + 1)

    plot_hv(hv_values, convergence_generation)
    plot_pareto_compare(history.F[99], history.F[499])

    pd.DataFrame(
        {
            "generation": np.arange(1, len(hv_values) + 1),
            "hypervolume": hv_values,
        }
    ).to_csv(OUT_DIR / "nsga2_convergence_hv_trace.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "IVRT-S initial 10 points",
                "pop_size": 50,
                "generations": 500,
                "reference_f1": ref_point[0],
                "reference_f2": ref_point[1],
                "final_hv": final_hv,
                "hv_99_percent_generation": convergence_generation,
                "hv_generation_100": hv_values[99],
                "hv_generation_500": final_hv,
            }
        ]
    )
    summary.to_csv(OUT_DIR / "nsga2_convergence_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved {OUT_DIR / 'fig_nsga2_convergence.png'}")
    print(f"Saved {OUT_DIR / 'fig_nsga2_pareto_compare.png'}")


if __name__ == "__main__":
    main()
