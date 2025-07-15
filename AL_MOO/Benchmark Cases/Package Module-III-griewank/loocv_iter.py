#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   loocv_iter.py
# Time    :   2025/06/13 12:37:01
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from icecream.icecream import ic
from matplotlib import pyplot as plt
import joblib

# ---------------------------- Coding Log ----------------------------
"""
    This script performs Gaussian Process Regression (GPR) with a custom kernel;

    Next plan:
    TODO: 1. test three different methods of inserting physical informations. (timely)

"""

# ---------------------------- Custom Kernel ----------------------------
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
        # Exponential term
        exp_term = np.sum(((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2) / self.wl, axis=2)
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
        return np.array([(self.v0 + self.a0 + self.a1 * np.sum(x**2) + self.v1) for x in X])

    def is_stationary(self):
        return False

    def __repr__(self):
        return (f"YuKernel(v0={self.v0}, wl={self.wl}, a0={self.a0}, a1={self.a1}, v1={self.v1})")

# ---------------------- Leave-One-Out CV Function ----------------------
def gpr_loocv_cv(v0, a0, a1, v1, **kwargs):
    """
    Perform LOOCV for Gaussian Process Regression with YuKernel.
    Returns the negative average MSE across all folds (to maximize in BayesianOptimization).
    """
    # 从 kwargs 中提取 wl 参数
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
def evaluate_loocv(v0, a0, a1, v1, wl):
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

# Plot true vs predicted values
def plot_results(y_true, y_pred, r2, iter, save_path):
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, c='red', edgecolors='k', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label=f'Ideal Fit (y=x)\nR²={r2:.3f}')
    plt.xlabel('Measured Release (µg/ml)')
    plt.ylabel('Predicted Release (µg/ml)')
    # plt.title(f'GPR Model: True vs Predicted')
    # plt.grid(True)
    plt.legend()
    # plt.savefig(f'{save_path}\\GPR_LOOCV_Results_{iter}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def run_plot_iter():
    """
    Run the plotting function for each iteration.
    """
    # This function is not used in the main script, but can be used for further iterations if needed.
    pass

# ------------------------------------ Main ------------------------------------
if __name__ == '__main__':
    # Load data
    dim = 3  # Specify the dimension
    method = 'RANDOM'  # Specify the method used
    # filename = f'lhs_samples_Ackley_{method}-{dim}D.xlsx'
    filename = f"lhs_samples_ackley_{method}-{dim}D.xlsx"
    path = 'Multi-Objective Optimisation\Benchmark\Package Module-III-rosenbrock\Dataset'
    save_path = f'Multi-Objective Optimisation\Benchmark\Package Module-III-rosenbrock\Loocv_RE\{method}'
    # Check if the save path exists, if not, create it
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Construct the full file path
    file_path = f'{path}\\{filename}'
    iter = 30
    sheet_name = f'RUN-{iter}-{method}-{dim}D'  # Specify the sheet name if needed
    
    # iterating through the iterations
    for iter in range(0, iter):
        # Load data for each iteration
        # sheet_name = f'RUN-{iter}-{method}-{dim}D'
        sheet_name = f'RUN-{iter}-{method}-{dim}D'
        print(f"Processing iteration {iter} with sheet name: {sheet_name}")
        # Load data
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values # 将y转换为一维数组
        # ic(X, y)

        # setup Bayesian Optimization
        n_features = X.shape[1]
        # create bounds dict dynamically
        upper_bounds = 1e2
        lower_bounds = 1e-2
        bounds = {f'wl{i+1}': (lower_bounds, upper_bounds) for i in range(n_features)}
        bounds.update({'v0': (lower_bounds, upper_bounds), 'a0': (lower_bounds, upper_bounds), 'a1': (lower_bounds, upper_bounds), 'v1': (lower_bounds, upper_bounds)})

        optimizer = BayesianOptimization(f=gpr_loocv_cv, pbounds=bounds, random_state=42)
        optimizer.maximize(init_points=2, n_iter=30, acquisition_function=UtilityFunction(kind='ei', xi=0.1))

        # best params
        best = optimizer.max['params']
        wl_best = [best[f'wl{i+1}'] for i in range(n_features)]

        # final evaluation
        mse, rmse, r2, mae, evs, y_pred, y_true = evaluate_loocv(
            best['v0'], best['a0'], best['a1'], best['v1'], wl_best)

        # present results
        metrics = [
            ["MSE", mse], ["RMSE", rmse], ["R²", r2],
            ["MAE", mae], ["Explained Var", evs]
        ]
        print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="pretty"))
        print("Best hyperparameters:")
        for k, v in best.items(): print(f"  {k}: {v}")

        # plot results
        plot_results(y_true, y_pred, r2, iter, save_path)

        # Record the R^2 value and other metrics for each iteration in the same Excel file
        results_file = f'{save_path}\\GPR_LOOCV_Results_metrics_{method}.xlsx'
        # Prepare the current iteration's results as a DataFrame row
        results_row = {
            'Iteration': iter,
            'R²': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Explained Variance': evs
        }
        # If the file exists, append; otherwise, create a new file
        if os.path.exists(results_file):
            existing_df = pd.read_excel(results_file)
            # Remove any existing row for this iteration (optional, to avoid duplicates)
            existing_df = existing_df[existing_df['Iteration'] != iter]
            updated_df = pd.concat([existing_df, pd.DataFrame([results_row])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([results_row])
        # Sort by iteration for clarity
        updated_df = updated_df.sort_values('Iteration').reset_index(drop=True)
        updated_df.to_excel(results_file, index=False)

    

    print(f"All iterations completed. Results saved to {results_file}")

