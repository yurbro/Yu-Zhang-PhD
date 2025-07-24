import numpy as np
import pandas as pd
from datetime import datetime  # 新增
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from lhs_sample import lhs_samples
# from ackley_func import ackley_max
# from zakharov_func import ackley_max, run_ackley        # TODO: this is zakharove function actually
# from rastrigin_func import ackley_max, run_ackley  # This is actually the Rastrigin function, but we use it as a placeholder for the benchmark
# from rosenbrock_func import ackley_max, run_ackley  # This is actually the Rosenbrock function, but we use it as a placeholder for the benchmark
# from griewank_func import ackley_max, run_ackley  # This is actually the Griewank function, but we use it as a placeholder for the benchmark
from sphere_func import ackley_max, run_ackley  # This is actually the Sphere function, but we use it as a placeholder for the benchmark

# np.random.seed(42) 

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Calculate Expected Improvement (EI) acquisition function
    X: Points to evaluate (n_points, dim)
    X_sample: Sampled points (n_sample, dim)
    Y_sample: Sampled observations (n_sample,)
    gpr: Trained GP model
    xi: Exploration parameter
    Returns: EI values (n_points,)
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def upper_confidence_bound(X, X_sample, Y_sample, gpr, kappa=2.576):
    """
    Calculate Upper Confidence Bound (UCB) acquisition function
    X: Points to evaluate (n_points, dim)
    X_sample: Sampled points (n_sample, dim)
    Y_sample: Sampled observations (n_sample,)
    gpr: Trained GP model
    kappa: Exploration parameter
    Returns: UCB values (n_points,)
    """
    mu, sigma = gpr.predict(X, return_std=True)
    return mu + kappa * sigma

def probability_of_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Calculate Probability of Improvement (PI) acquisition function
    X: Points to evaluate (n_points, dim)
    X_sample: Sampled points (n_sample, dim)
    Y_sample: Sampled observations (n_sample,)
    gpr: Trained GP model
    xi: Exploration parameter
    Returns: PI values (n_points,)
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0
    return pi

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    """
    Within given bounds, use multiple random restarts to optimize the acquisition function and find the best next sampling point
    bounds: List [(min, max), ...]
    Returns: Best point x (1, dim)
    """
    dim = X_sample.shape[1]
    min_val = 1e20
    best_x = None

    for i in range(n_restarts):
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(dim,))
        # Simple local search: random perturbation
        for _ in range(100):
            xi = x0 + 0.01 * np.random.randn(dim)
            xi = np.clip(xi, [b[0] for b in bounds], [b[1] for b in bounds])
            acq_val = acquisition(xi.reshape(1, -1), X_sample, Y_sample, gpr)
            # print(f"acquisition({acquisition.__name__}) at {xi.round(4)} = {acq_val}")  # Print acquisition function value
            val = -acq_val
            if val < min_val:
                min_val = val
                best_x = xi

    return np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds]).reshape(1, -1)


def bayesian_optimization(objective_func, X_init, Y_init, bounds, n_iter, af_func, benchmark, acquisition_func):
    """
    Main function for single-objective Bayesian optimization
    objective_func: Objective function, input (dim,) returns scalar
    X_init: Initial sample points, numpy array, (n_init, dim)
    Y_init: Initial observations, numpy array, (n_init,)
    bounds: Variable bounds list [(min, max), ...]
    n_iter: Number of iterations
    Returns: Sample points and corresponding observations
    """
    # Copy initial data
    X_sample = X_init.copy()
    Y_sample = Y_init.copy()

    # Record results every 6 iterations
    record = []

    # Define GP model
    # kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
    # kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(alpha=1e-10, normalize_y=True)

    for i in range(n_iter):
        # Train GP
        gpr.fit(X_sample, Y_sample)

        # Propose next point
        # acquisition_func = probability_of_improvement # Can be replaced with upper_confidence_bound or probability_of_improvement
        x_next = propose_location(acquisition_func, X_sample, Y_sample, gpr, bounds)

        # Calculate true value of objective function
        y_next = objective_func(x_next.ravel())

        # Add new sample
        X_sample = np.vstack((X_sample, x_next))
        Y_sample = np.append(Y_sample, y_next)

        # Check if a larger value is found
        if y_next > np.max(Y_sample[:-1]):
            print(f"\033[95mIteration {i+1}: x_next={x_next.ravel()}, y_next={y_next}, New best {benchmark}={y_next}\033[0m")
        else:
            print(f"Iteration {i+1}: x_next={x_next.ravel()}, y_next={y_next}, Current best {benchmark}={np.max(Y_sample)}")

        # Record every n iterations
        if (i + 1) % 6 == 0:
            record.append({
            'iteration': i + 1,
            **{f'x{j+1}': x_next.ravel()[j] for j in range(x_next.shape[1])},
            'y_next': y_next,
            f'Current best {benchmark}': np.max(Y_sample),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Added
            })

    # Save to Excel (append mode, do not overwrite existing results)
    if record:
        df = pd.DataFrame(record)
        save_path = fr"Multi-Objective Optimisation\Benchmark\Package Module-III-sphere\BO-RE\single_bo_{af_func}_{benchmark}_{dim}D.xlsx"
        try:
            # If the file exists, read old data and append
            old_df = pd.read_excel(save_path)
            df = pd.concat([old_df, df], ignore_index=True)
        except FileNotFoundError:
            pass  # If file does not exist, save new data directly
        df.to_excel(save_path, index=False)
        print(f"Saved record to {save_path}")

    return X_sample, Y_sample

def ackley_function(**kwargs) -> float:
    # Convert keyword arguments to ordered array
    x = np.array([kwargs[f'x{i+1}'] for i in range(len(kwargs))], dtype=float)
    """
    Calculate standard Griewank function value (minimization objective).
    Input:
        x: np.ndarray or list, shape can be (d,) or (n,d)
           Each row is a d-dimensional vector [x1, x2, ..., xd].
    Returns:
        np.ndarray, shape (n,)
    Formula:
        f(x) = sum(x_i^2) / 4000 - prod(cos(x_i / sqrt(i))) + 1
        where i=1,...,d
    """
    
    x_arr = np.atleast_2d(x).astype(float)
    return -(np.sum(x_arr ** 2, axis=1))

def array_to_ackley(x):
    # x: 1D array
    return ackley_function(**{f'x{i+1}': x[i] for i in range(len(x))})

def run_moo_initial_experiment(n_init, lb, ub, benchmark):
    """
    Run the initial experiment for multi-objective optimisation.
    """
    # 1. Initialize the experiment data using lhs_sampling
    X_init = lhs_samples(n_init, lb, ub)  # Shape (n_init, dim)
    # 2.1 Evaluate the initial samples using the griewank function
    Y_init = ackley_max(X_init)
    # Y_init = np.array([griewank_function(**{f'x{i+1}': x[i] for i in range(len(x))}) for x in X_init])
    for xi, yi in zip(X_init, Y_init):
        print(f"x={xi.round(3)} → {benchmark}={yi:.4f}")

    return X_init, Y_init

# example usage of adaptive weight function
if __name__ == "__main__":

    # Set initial parameters
    start_time = datetime.now()
    print(f"Starting Bayesian Optimization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    n_init = 10  # Number of initial sample points
    n_iter = 180  # Number of optimization iterations
    dim = 5  # Dimension
    af_func = 'ucb'  # Acquisition function type
    benchmark = 'Sphere'  # Benchmark function Sphere
    acquisition_func = {
        'ei': expected_improvement,
        'ucb': upper_confidence_bound,
        'poi': probability_of_improvement
    }
    lb, ub = np.array([-60] * dim), np.array([60] * dim)  # Lower and upper bounds
    X_init, Y_init = run_moo_initial_experiment(n_init, lb, ub, benchmark= benchmark)
    # Print initial sample points and corresponding objective values
    print(f"Initial samples (shape {X_init.shape}):")
    print(X_init.round(3))
    print(f"Initial objective values (shape {Y_init.shape}):")
    print(Y_init.round(3))

    # Set bounds according to the dimension of X
    dim = X_init.shape[1]
    bounds = {f'x{i+1}': (-60, 60) for i in range(dim)}
    bounds = [(b[0], b[1]) for b in bounds.values()]
    # np.random.seed(42) 
    X_opt, Y_opt = bayesian_optimization(array_to_ackley, 
                                         X_init, Y_init, 
                                         bounds, n_iter=n_iter, 
                                         af_func=af_func,
                                         benchmark=benchmark, 
                                         acquisition_func=acquisition_func[af_func])
    print("Optimization completed.")

    # Output the best optimization value and corresponding input parameters
    best_idx = np.argmax(Y_opt)
    best_x = X_opt[best_idx]
    best_y = Y_opt[best_idx]
    print(f"Best value: {best_y}")
    print(f"Best input: {best_x}")
    print("---------Successfully completed the Bayesian Optimization process---------")
    end_time = datetime.now()
    print(f"Ending time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {end_time - start_time}")