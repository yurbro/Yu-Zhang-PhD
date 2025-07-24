import numpy as np
import pandas as pd
from datetime import datetime  # 新增
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from lhs_sample import lhs_samples
# from ackley_func import ackley_max
# from zakharov_func import ackley_max, run_ackley        # TODO: this is zakharove function actually
from rastrigin_func import ackley_max, run_ackley  # This is actually the Rastrigin function, but we use it as a placeholder for the benchmark

# np.random.seed(42) 

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):

    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def upper_confidence_bound(X, X_sample, Y_sample, gpr, kappa=2.576):
    mu, sigma = gpr.predict(X, return_std=True)
    return mu + kappa * sigma

def probability_of_improvement(X, X_sample, Y_sample, gpr, xi=0.01):

    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0
    return pi

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):

    dim = X_sample.shape[1]
    min_val = 1e20
    best_x = None

    for i in range(n_restarts):
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(dim,))
        # simple random search
        for _ in range(100):
            xi = x0 + 0.01 * np.random.randn(dim)
            xi = np.clip(xi, [b[0] for b in bounds], [b[1] for b in bounds])
            acq_val = acquisition(xi.reshape(1, -1), X_sample, Y_sample, gpr)
            # print(f"acquisition({acquisition.__name__}) at {xi.round(4)} = {acq_val}")  # 打印采集函数值
            val = -acq_val
            if val < min_val:
                min_val = val
                best_x = xi

    return np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds]).reshape(1, -1)


def bayesian_optimization(objective_func, X_init, Y_init, bounds, n_iter, af_func, benchmark, acquisition_func):

    X_sample = X_init.copy()
    Y_sample = Y_init.copy()

    record = []

    gpr = GaussianProcessRegressor(alpha=1e-10, normalize_y=True)

    for i in range(n_iter):

        gpr.fit(X_sample, Y_sample)

        x_next = propose_location(acquisition_func, X_sample, Y_sample, gpr, bounds)

        y_next = objective_func(x_next.ravel())

        X_sample = np.vstack((X_sample, x_next))
        Y_sample = np.append(Y_sample, y_next)

        if y_next > np.max(Y_sample[:-1]):
            print(f"\033[95mIteration {i+1}: x_next={x_next.ravel()}, y_next={y_next}, New best {benchmark}={y_next}\033[0m")
        else:
            print(f"Iteration {i+1}: x_next={x_next.ravel()}, y_next={y_next}, Current best {benchmark}={np.max(Y_sample)}")

        if (i + 1) % 6 == 0:
            record.append({
            'iteration': i + 1,
            **{f'x{j+1}': x_next.ravel()[j] for j in range(x_next.shape[1])},
            'y_next': y_next,
            f'Current best {benchmark}': np.max(Y_sample),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
            })

    if record:
        df = pd.DataFrame(record)
        save_path = fr"Multi-Objective Optimisation\Benchmark\Package Module-III-rastrigin\BO-RE\single_bo_{af_func}_{benchmark}_{dim}D.xlsx"
        try:
            old_df = pd.read_excel(save_path)
            df = pd.concat([old_df, df], ignore_index=True)
        except FileNotFoundError:
            pass  
        df.to_excel(save_path, index=False)
        print(f"Saved record to {save_path}")

    return X_sample, Y_sample

def ackley_function(**kwargs) -> float:

    x = np.array([kwargs[f'x{i+1}'] for i in range(len(kwargs))], dtype=float)
    x_arr = np.atleast_2d(x).astype(float)
    n, d = x_arr.shape
    A = 10  # 常数
    return -(A * d + np.sum(x_arr**2 - A * np.cos(2 * np.pi * x_arr), axis=1))

def zakharov_function(**kwargs) -> float:
    
    x = np.array([kwargs[f'x{i+1}'] for i in range(len(kwargs))], dtype=float)
    x_arr = np.atleast_2d(x)
    d = x_arr.shape[1]
    idx = np.arange(1, d+1)
    sum1 = np.sum(x_arr**2, axis=1)
    sum2 = np.sum(0.5 * idx * x_arr, axis=1)
    return float(-(sum1 + sum2**2 + sum2**4))

def array_to_ackley(x):
    # x: 1D array
    return ackley_function(**{f'x{i+1}': x[i] for i in range(len(x))})

def array_to_zakharov(x):
    # x: 1D array
    return zakharov_function(**{f'x{i+1}': x[i] for i in range(len(x))})

def run_moo_initial_experiment(n_init, lb, ub, benchmark):
    """
    Run the initial experiment for multi-objective optimisation.
    """
    # 1. Initialise the experiment data by using the lhs_sampling
    X_init = lhs_samples(n_init, lb, ub)  # Shape (n_init, dim)
    # 2.1 Evaluate the initial samples using the zakharov function
    Y_init = ackley_max(X_init)
    # Y_init = np.array([zakharov_function(**{f'x{i+1}': x[i] for i in range(len(x))}) for x in X_init])
    for xi, yi in zip(X_init, Y_init):
        print(f"x={xi.round(3)} → {benchmark}={yi:.4f}")

    return X_init, Y_init

# Adaptive weight function for Bayesian optimisation
if __name__ == "__main__":

    # Initial conditions
    start_time = datetime.now()
    print(f"Starting Bayesian Optimization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    n_init = 10  # initial sample size
    n_iter = 180  # optimization iterations
    dim = 5  # dimensions
    af_func = 'ucb'  # acquisition function type
    benchmark = 'Rastrigin'  # benchmark function Rastrigin
    acquisition_func = {
        'ei': expected_improvement,
        'ucb': upper_confidence_bound,
        'poi': probability_of_improvement
    }
    lb, ub = np.array([-5.12] * dim), np.array([5.12] * dim)  # Lower and upper bounds
    X_init, Y_init = run_moo_initial_experiment(n_init, lb, ub, benchmark= benchmark)
    # Print initial samples and their corresponding objective function values
    print(f"Initial samples (shape {X_init.shape}):")
    print(X_init.round(3))
    print(f"Initial objective values (shape {Y_init.shape}):")
    print(Y_init.round(3))

    # set up the bounds for the optimization
    dim = X_init.shape[1]
    bounds = {f'x{i+1}': (-5.12, 5.12) for i in range(dim)}
    bounds = [(b[0], b[1]) for b in bounds.values()]
    # np.random.seed(42) 
    X_opt, Y_opt = bayesian_optimization(array_to_ackley, 
                                         X_init, Y_init, 
                                         bounds, n_iter=n_iter, 
                                         af_func=af_func,
                                         benchmark=benchmark, 
                                         acquisition_func=acquisition_func[af_func])
    print("Optimization completed.")

    # output the best result
    best_idx = np.argmax(Y_opt)
    best_x = X_opt[best_idx]
    best_y = Y_opt[best_idx]
    print(f"Best value: {best_y}")
    print(f"Best input: {best_x}")
    print("---------Successfully completed the Bayesian Optimization process---------")
    end_time = datetime.now()
    print(f"Ending time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {end_time - start_time}")