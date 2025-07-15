#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   GPR-Basement.py
# Time    :   2025/05/13 20:34:23
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import itertools
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
def plot_results(y_true, y_pred, r2):
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, c='red', edgecolors='k', alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label=f'Ideal Fit (y=x)\nR²={r2:.2f}\nRMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.2f}')
    plt.xlabel('Measured Release (µg/cm²)', fontsize=12)
    plt.ylabel('Predicted Release (µg/cm²)', fontsize=12)
    # plt.title(f'GPR Model: True vs Predicted')
    # plt.grid(True)
    plt.legend()
    plt.savefig('Multi-Objective Optimisation\Pareto_animation\\fulldata-s-final\GPR_LOOCV_Results_RMSE_R2.png', dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------ Main ------------------------------------
if __name__ == '__main__':
    # Load data
    # filename = 'Multi-Objective Optimisation\Dataset\IVRT-Pure.xlsx'
    filename = 'Multi-Objective Optimisation\Dataset\IVRT-Final.xlsx'
    # filename = 'Physics-informed GP\Data\IVRT-Pure.xlsx'

    X = pd.read_excel(filename, sheet_name='F-FINAL')
    y = pd.read_excel(filename, sheet_name='R-FINAL')

    X = X.iloc[:, 1:].values
    y = y.iloc[:, -1].values # 将y转换为一维数组
    ic(X, y)

    # setup Bayesian Optimization
    n_features = X.shape[1]
    # create bounds dict dynamically
    upper_bounds = 1e3
    lower_bounds = 1e-3
    bounds = {f'wl{i+1}': (lower_bounds, upper_bounds) for i in range(n_features)}
    bounds.update({'v0': (lower_bounds, upper_bounds), 'a0': (lower_bounds, upper_bounds), 'a1': (lower_bounds, upper_bounds), 'v1': (lower_bounds, upper_bounds)})

    optimizer = BayesianOptimization(f=gpr_loocv_cv, pbounds=bounds, random_state=42)
    optimizer.maximize(init_points=2, n_iter=50, acquisition_function=UtilityFunction(kind='ei', xi=0.1))

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
    plot_results(y_true, y_pred, r2)

    # 保存模型以备用
    # 保存最优参数和scaler

    best_gpr = GaussianProcessRegressor(kernel=YuKernel(
        best['v0'], wl_best, best['a0'], best['a1'], best['v1']),
        n_restarts_optimizer=5, alpha=1e-10
    )
    best_gpr.fit(StandardScaler().fit(X).transform(X), StandardScaler().fit(y.reshape(-1, 1)).transform(y.reshape(-1, 1)).flatten())
    # Save the model and scalers
    import joblib
    print("Saving the best model and scalers...")

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    X = scaler_X.transform(X)
    y = scaler_y.transform(y.reshape(-1, 1)).flatten()
    best_gpr.fit(X, y)
    joblib.dump(best_gpr, 'Multi-Objective Optimisation\Fixed Model\gpr_loocv_best_model.pkl')

    print("The model has been saved to Multi-Objective Optimisation\Fixed Model\gpr_loocv_best_model.pkl")

    # Global Sensitivity Analysis
    from SALib.sample import saltelli
    from SALib.analyze import sobol 
    from SALib.sample import saltelli
    # 1. Define the problem
    problem = {
        'num_vars': 3,
        'names': ['Poloxamer407', 'Ethanol', 'PG'],
        'bounds': [[20, 30], [10, 20], [10, 20]]
    }

    # 2. Sampling
    N = 512
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    print(f"Sobol sample shape: {param_values.shape}")  # debug

    # standardize the parameters
    # scaler_X = StandardScaler().fit(param_values)
    param_values = scaler_X.transform(param_values)
    ic(param_values, param_values.shape)
    # 3. Use GPR model for prediction
    Y_pred = best_gpr.predict(param_values)  # trained model
    ic(Y_pred, Y_pred.shape)

    # 4. Sensitivity Analysis
    Si = sobol.analyze(problem, Y_pred, calc_second_order=True, print_to_console=False)
    ic(Si)

    # Plot Sobol indices
    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(problem['num_vars'])
    ax.bar(x_pos - 0.15, Si['S1'],  width=0.3,  label='First-order $S_i$')
    ax.bar(x_pos + 0.15, Si['ST'], width=0.3,  label='Total-order $S_{Ti}$', alpha=0.6)
    # Display the specific Sobol index values on the bars
    for i, (s1, st) in enumerate(zip(Si['S1'], Si['ST'])):
        ax.text(i - 0.15, s1 + 0.02, f'{s1:.4f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + 0.15, st + 0.02, f'{st:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(problem['names'], fontsize=12)
    ax.set_ylabel('Sobol index', fontsize=12)
    ax.set_ylim(0, 1)
    # ax.set_title('Sobol Sensitivity Indices', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig('Multi-Objective Optimisation\Pareto_animation\GSA\Sobol_Sensitivity_Indices.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------------------------------
# 6. Main Effects Curve and Second-order Interaction Heatmap
# -------------------------------------------------------

def plot_main_effects(problem, param_values, Y_pred, n_bins=20, Si=None):
    """
    Plot main effects with legend showing Si['S1'] values for each parameter.
    Args:
        problem: dict, SALib problem definition
        param_values: np.ndarray, sampled parameter values (standardized)
        Y_pred: np.ndarray, predicted outputs
        n_bins: int, number of bins for main effect curve
        Si: dict, SALib sensitivity indices (optional, for S1 values in legend)
    """
    fig, ax = plt.subplots()
    legend_labels = []
    for i, name in enumerate(problem['names']):
        df = pd.DataFrame({'Xi': param_values[:, i], 'Y': Y_pred})
        # Bin the standardized min-max values
        xi_min, xi_max = df['Xi'].min(), df['Xi'].max()
        bins = np.linspace(xi_min, xi_max, n_bins + 1)
        df['bin'] = np.digitize(df['Xi'], bins) - 1
        main_effect = np.full(n_bins, np.nan)
        for b in range(n_bins):
            vals = df.loc[df['bin'] == b, 'Y']
            if len(vals) > 0:
                main_effect[b] = vals.mean()
        x_centers = 0.5 * (bins[:-1] + bins[1:])
        # x-axis normalization to 0-1
        if Si is not None and 'S1' in Si and 'ST' in Si:
            s1_val = Si['S1'][i]
            st_val = Si['ST'][i]
            label = f'{name} ($S_1$={s1_val:.3f}, $S_T$={st_val:.3f})'
        elif Si is not None and 'S1' in Si:
            s1_val = Si['S1'][i]
            label = f'{name} ($S_1$={s1_val:.3f})'
        else:
            label = f'{name}'
        ax.plot((x_centers - xi_min) / (xi_max - xi_min), main_effect, label=label)
    ax.set_xlabel('Scaled Process Factor, $x_i$', fontsize=12)
    ax.set_ylabel('Main Effect: E(y|$x_i$)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig('Multi-Objective Optimisation\Pareto_animation\GSA\Main_Effects_Curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the main effects plotting function
# Inverse transform Y_pred
Y_pred_real = scaler_y.inverse_transform(Y_pred.reshape(-1, 1)).flatten()
ic(Y_pred_real)

# then use Y_pred_real to plot the main effects curve
plot_main_effects(problem, param_values, Y_pred_real, Si=Si)

# Second-order interaction heatmap (only show the top few)
def plot_top_interactions(Si, top_k=3):
    import itertools, seaborn as sns  # seaborn is only used for heatmaps, for quick viewing
    pairs = list(itertools.combinations(range(problem['num_vars']), 2))
    S2 = np.array([Si['S2'][i][j] for i, j in pairs])
    idx_sorted = np.argsort(S2)[::-1][:top_k]
    for idx in idx_sorted:
        i, j = pairs[idx]
        print(f'Interaction {problem["names"][i]} & {problem["names"][j]}: S_ij = {S2[idx]:.3f}')
    
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(Si['S2'][idx_sorted], annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=[problem['names'][i] for i in range(problem['num_vars'])],
                yticklabels=[problem['names'][j] for j in range(problem['num_vars'])],
                cbar_kws={'label': 'Second-order Sobol index'})
    ax.set_title('Top Second-order Sobol Indices')
    plt.tight_layout()
    plt.savefig('Multi-Objective Optimisation\Pareto_animation\GSA\Top_Interactions_Heatmap.png', dpi=300, bbox_inches='tight')
    # Only show the top few interactions
    plt.show()

plot_top_interactions(Si, top_k=3)

def plot_interaction_surface(model, scaler_X, param_names, fixed_values, i, j, n_points=30):

    xi_range = np.linspace(0, 1, n_points)
    xj_range = np.linspace(0, 1, n_points)
    Xi, Xj = np.meshgrid(xi_range, xj_range)
    X_grid = np.tile(fixed_values, (n_points * n_points, 1))
    X_grid[:, i] = Xi.ravel()
    X_grid[:, j] = Xj.ravel()
    # Inverse transform to original space (if needed)
    X_grid_orig = scaler_X.inverse_transform(X_grid)
    # Predict
    Y_pred = model.predict(X_grid)
    Y_pred = Y_pred.reshape(Xi.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xi, Xj, Y_pred, cmap='viridis', alpha=0.8)
    ax.set_xlabel(param_names[i])
    ax.set_ylabel(param_names[j])
    ax.set_zlabel('Predicted Output')
    plt.title(f'Interaction Effect: {param_names[i]} & {param_names[j]}')
    plt.show()

for i, j in itertools.combinations(range(problem['num_vars']), 2):
    plot_interaction_surface(best_gpr, scaler_X, problem['names'], np.mean(param_values, axis=0),
                            i=i, j=j, n_points=30)
