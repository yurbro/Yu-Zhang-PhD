#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   ack_GPR_MOO-RUN2-PROPOSED.py
# Time    :   2025/06/04 11:03:03
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk


import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tabulate import tabulate
from datetime import timedelta
from time import time
from icecream import ic
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback


class YuKernel(Kernel):
    def __init__(self, v0, wl, a0, a1, v1):
        self.v0 = v0
        self.wl = np.atleast_1d(wl)  # make sure wl is at least 1D
        self.a0 = a0
        self.a1 = a1
        self.v1 = v1
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X

        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Make sure self.wl has the correct shape, for example:
        if self.wl.size != X.shape[-1]:
            # Here is error handling or logic adjustment
            # For example, expand or adjust self.wl to match the number of features
            raise ValueError("wl size must match the number of features")

        # Exponential term
        exp_term = np.sum(((X[:, np.newaxis, :] - Y[np.newaxis, :, :])**2) / self.wl, axis=2)
        exp_term = self.v0 * np.exp(-0.5 * exp_term)

        # Linear term
        linear_term = self.a0 + self.a1 * np.dot(X, Y.T)

        # Noise term
        if X is Y:
            noise_term = self.v1 * np.eye(X.shape[0])
        else:
            noise_term = np.zeros((X.shape[0], Y.shape[0]))

        return exp_term + linear_term + noise_term

    def diag(self, X):
        """
        Return the diagonal of the kernel matrix.
        """
        return np.array([self.v0 + self.a0 + self.a1 * np.sum(X**2, axis=1) + self.v1] * X.shape[0])

    def is_stationary(self):
        """
        Returns whether the kernel is stationary.
        """
        return False
    
    def __repr__(self):
        return (f"YuKernel(v0={self.v0}, wl={self.wl}, a0={self.a0}, " 
                f"a1={self.a1}, v1={self.v1})")
    

# define evaluation metrics
def evaluate(y_test, y_pred_test):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred_test)
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Calculate R-squared (R^2)
    r_squared = 1 - (np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2))   # 1 - SSE/SST
    # r_squared = r2_score(y_test, y_pred_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred_test)
    # Calculate Explained Variance Score
    evs = explained_variance_score(y_test, y_pred_test)
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    return mse, rmse, r_squared, mae, evs, mape

# ---------------------- Leave-One-Out CV Function ----------------------
def gpr_loocv_cv(v0, a0, a1, v1, X, y, **kwargs):
    """
    Perform LOOCV for Gaussian Process Regression with YuKernel.
    Returns the negative average MSE across all folds (to maximize in BayesianOptimization).
    """
    wl = [kwargs[f'wl{i+1}'] for i in range(len(kwargs)) if f'wl{i+1}' in kwargs]
    loo = LeaveOneOut()
    mse_list = []

    # iterate over each left-out sample
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # scale features and target
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        X_tr_scaled = scaler_X.transform(X_train)
        X_te_scaled = scaler_X.transform(X_test)
        y_tr_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

        # define and fit GPR model
        kernel = YuKernel(v0, wl, a0, a1, v1)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-10)
        gpr.fit(X_tr_scaled, y_tr_scaled)

        # predict and invert scaling
        y_pred_scaled = gpr.predict(X_te_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # compute MSE for this fold
        mse_fold = mean_squared_error(y_test, y_pred)
        mse_list.append(mse_fold)

    # average MSE
    avg_mse = np.mean(mse_list)
    return -avg_mse

# -------------------------- Evaluation Helper ---------------------------
def evaluate_loocv(v0, a0, a1, v1, wl, X, y, **kwargs):
    """
    Final LOOCV evaluation to get detailed metrics.
    """
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        X_tr = scaler_X.transform(X_train)
        X_te = scaler_X.transform(X_test)
        y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        kernel = YuKernel(v0, wl, a0, a1, v1)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-10)
        gpr.fit(X_tr, y_tr)
        y_pred_s = gpr.predict(X_te)
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true_all - y_pred_all)**2) / np.sum((y_true_all - np.mean(y_true_all))**2)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    evs = explained_variance_score(y_true_all, y_pred_all)
    return mse, rmse, r2, mae, evs, y_pred_all, y_true_all

# Define a problem class
class MyProblem(ElementwiseProblem):

    def __init__(self, best_gpr, scaler_X, scaler_y, xl, xu, dim):
        super().__init__(n_var=dim,  # Number of input features
                         n_obj=2,  # Number of objectives
                         n_constr=0,  # Number of constraints
                         xl=xl,  # Lower bound for inputs
                         xu=xu)  # Upper bound for inputs

        self.best_gpr = best_gpr
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def _evaluate(self, x, out, *args, **kwargs):
        # x_scaled = self.scaler_X.transform([x])
        y_pred, y_std = self.best_gpr.predict([x], return_std=True)     # Here the output y_std is the standard deviation
        # ic(x, y_pred)

        y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        # y_std_original = y_std * self.scaler_y.scale_

        y_std_original = y_std[0]   
        y_pred_original = y_pred_original.squeeze()  # 去掉维度为1的维度

        # Just use the final sampling data as the target data
        mean_pred = y_pred_original
        mean_std = y_std_original[-1] 
        # ic(mean_pred, mean_std)

        out["F"] = np.array([-mean_pred, -mean_std])  # Objective function values

# Callback class to store the variables
class CollectParetoFronts(Callback):
    def __init__(self):
        super().__init__()
        self.pareto_fronts = []

    def notify(self, algorithm):
        pareto_front = algorithm.pop.get("F")
        # print(f'Pareto front shape: {pareto_front.shape}')  # 打印形状以检查
        self.pareto_fronts.append(pareto_front)

def train_gpr_model(X_df, y_df, lower_bound, upper_bound, run_num, method_type, path_data):
    """
    Train a Gaussian Process Regression model with the given parameters.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=2, shuffle=True)

    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    # y_test_scaled = scaler_y.transform(y_test)
    # ic(X_train_scaled, y_train_scaled)

    # parameters bounds for Bayesian Optimization 
    input_dim = X_df.shape[1]
    # Dynamically set the number of wl parameters based on input_dim
    params_bounds = {
        'a0': (lower_bound, upper_bound),
        'a1': (lower_bound, upper_bound),
        'v0': (lower_bound, upper_bound),
        'v1': (lower_bound, upper_bound),
    }
    for i in range(input_dim):
        params_bounds[f'wl{i+1}'] = (lower_bound, upper_bound)

    # Dynamically build the optimizer function signature for wl parameters
    def optimizer_func(a0, a1, v0, v1, **kwargs):
        # Only pass the correct number of wl arguments
        wl_kwargs = {f'wl{i+1}': kwargs[f'wl{i+1}'] for i in range(input_dim)}
        return gpr_loocv_cv(v0, a0, a1, v1, X_df, y_df, **wl_kwargs)

    optimizer = BayesianOptimization(
        f=optimizer_func,
        pbounds=params_bounds,
        random_state=2
    )
    acquisition_function = UtilityFunction(kind='ei', xi=0.1)
    # Running the optimization
    optimizer.maximize(init_points=2, n_iter=20, acquisition_function=acquisition_function)
    # get the best hyperparameters
    best_params = optimizer.max['params']
    # Save the best hyperparameters to an Excel file
    # check if the directory exists, if not, create it
    directory_run_best_params = f"{path_data}\RUN-{run_num}-{method_type}"
    if not os.path.exists(directory_run_best_params):
        os.makedirs(directory_run_best_params)
    directory_run_best_params_path = os.path.join(directory_run_best_params, "Bayesian_Optimisation_Results.xlsx")
    pd.DataFrame([best_params]).to_excel(directory_run_best_params_path, index=False)

    # use the best hyperparameters to create a new Gaussian Process Regression model
    wl_list = [best_params[f'wl{i+1}'] for i in range(input_dim)]
    best_gpr = GaussianProcessRegressor(kernel=YuKernel(best_params['v0'],
                                                        wl_list,
                                                        best_params['a0'],
                                                        best_params['a1'],
                                                        best_params['v1'])
                                                        # , alpha=noise_var
                                                        , n_restarts_optimizer=10, alpha=1e-10
                                                        )
    # train the model
    best_gpr.fit(X_train_scaled, y_train_scaled)

    # use the trained model to make predictions
    y_pred_test, sigma_test = best_gpr.predict(X_test_scaled, return_std=True)
    y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    # sigma_test = sigma_test * scaler_y.scale_
    sigma_test = sigma_test[0]

    # ic(y_pred_test, y_test, sigma_test)

    mse, rmse, r_squared, mae, evs, mape = evaluate(y_test, y_pred_test)

    # put the results in a table format
    table_data = [
        ["Mean Squared Error (MSE)", mse],
        ["Root Mean Squared Error (RMSE)", rmse],
        ["Mean Absolute Error (MAE)", mae],
        ["Explained Variance Score (EVS)", evs],
        ["Mean Absolute Percentage Error (MAPE)", mape],
        ["R-squared (R²)", f'\033[91m{r_squared}\033[0m']
        ]

    # Output results
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))

    formatted_params = '\n'.join([f'\033[92m{key}\033[0m: \033[94m{value}\033[0m' for key, value in best_params.items()])

    print("Best Hyperparameters:\n", formatted_params)
 
    return best_gpr, scaler_X, scaler_y

def MultiObjectiveOptimisation(d, best_gpr, scaler_X, scaler_y, popsize, gen, run_num, method, max_y, xl, xu, path_data, path_results, savefig):
    """
    Perform multi-objective optimization using Bayesian Optimization.
    """
    # Create problem instance
    problem = MyProblem(best_gpr, scaler_X, scaler_y, xl, xu, d)

    # Create an instance to collect data
    collect_pf = CollectParetoFronts()

    # Create algorithm instance
    algorithm = NSGA2(pop_size=popsize) # Population size is 10.

    # Execute optimization
    res = minimize(problem,
                algorithm,
                termination=('n_gen', gen),   # Termination condition is 50 generations
                callback=collect_pf,
                verbose=True,
                seed=1)

    # Output results
    print("The best solution by NSGA-II algorithm is: ")
    for i in range(len(res.X)):
        print(f"Solution-[{i}]: X = {np.round(res.X[i], 3)}, F = {np.round(res.F[i], 3)}, X_original= {np.round(scaler_X.inverse_transform([res.X[i]]), 3)}")

    # Save the Pareto front and the corresponding variables to a xlxs file
    pareto_front_df = pd.DataFrame(-res.F, columns=['Mean', 'Std'])
    pareto_front_df['X'] = [scaler_X.inverse_transform([x]) for x in res.X]

    # Check the directory exists, if not, create it
    directory_run = f"{path_data}\RUN-{run_num}-{method}"
    if not os.path.exists(directory_run):
        os.makedirs(directory_run)
    path_run_moo = os.path.join(directory_run, "pareto_front.xlsx")     # Save the Pareto front to an Excel file
    pareto_front_df.to_excel(path_run_moo, index=False)

    get_mean = []
    get_std = []


if __name__ == "__main__":
    # # set the parameters
    # lower_bound = 1e-2
    # upper_bound = 1e2

    # file_path = "Multi-Objective Optimisation\Benchmark\Package Module\Dataset\lhs_samples_ackley.xlsx"
    # run_num = 1
    # method = 'PROPOSED'  # Method name for the run
    # df = pd.read_excel(file_path, sheet_name=f'INITIAL')  # Read the initial dataset
    # # X = df[[f"x{i+1}" for i in range(3)]].to_numpy(dtype=float)  # Shape (20, 3)
    # # y = df['Ackley'].to_numpy(dtype=float)  # Shape (20,)

    # max_y = np.max(y)  # Get the maximum value in y for the incumbent best

    # # train the GPR model with the initial dataset
    # best_gpr, scaler_X, scaler_y= train_gpr_model(X, y, lower_bound, upper_bound, run_num, method)

    # # Perform multi-objective optimisation
    # popsize = 10  # Population size for NSGA-II
    # gen = 50  # Number of generations for NSGA-II

    # # define the lower and upper bounds for ibuprofen excipients
    # X_lower = np.array([-5, -5, -5])  # lower bounds of ibuprofen excipients
    # X_upper = np.array([10, 10, 10])   # upper bounds of ibuprofen excipients

    # # use the scaler to transform the lower and upper bounds
    # xl = scaler_X.transform([X_lower])[0]
    # xu = scaler_X.transform([X_upper])[0]
    # ic(xl, xu)

    # # Perform multi-objective optimisation
    # MultiObjectiveOptimisation(best_gpr, scaler_X, scaler_y, popsize, gen, run_num, method, max_y, xl, xu)

    # print("Successful multi-objective optimisation completed!!!")
    
    pass
