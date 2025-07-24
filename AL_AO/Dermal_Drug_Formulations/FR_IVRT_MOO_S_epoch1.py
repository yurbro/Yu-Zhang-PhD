#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   FR_IVRT_MOO.py
# Time    :   2025/05/20 16:36:58
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import numpy as np
import os
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
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
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
from sklearn.cluster import KMeans
# from kneed import KneeLocator
from collections import Counter
from PoE import calculate_current_best, method2_probability, PoE_decision_v2
from PoI import probability_of_improvement

"""
    29/09/2024: Delete the k-means clustering and relevant code.
    21/05/2025: Here we try to use the full dataset to train GPR and NSGA-II to find the best formulation design.
    22/5/2025: Test the new created dataset IVRT-S.xlsx.
"""
    
class YuKernel(Kernel):
    def __init__(self, v0, wl, a0, a1, v1):
        self.v0 = v0
        self.wl = np.atleast_1d(wl)  # Ensure wl is an array
        self.a0 = a0
        self.a1 = a1
        self.v1 = v1
    
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X

        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Ensure self.wl has the correct shape, e.g.:
        if self.wl.size != X.shape[-1]:
            # Error handling or logic adjustment
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
    


# Bayesian Optimization for GPR model with YuKernel kernel function and cross-validation for hyperparameter tuning 
def gpr_model_cv(v0, a0, a1, v1, wl1, wl2, wl3):
    # combine the three wavelengths into a single array
    wl = [wl1, wl2, wl3]
    # define the model
    gpr = GaussianProcessRegressor(kernel=YuKernel(v0, wl, a0, a1, v1), n_restarts_optimizer=10, alpha=1e-10)

    # fit the model on the training set
    gpr.fit(X_train_scaled, y_train_scaled)

    # predict the values in the test set
    y_pred = gpr.predict(X_test_scaled)

    # transform the predicted data to the original scale
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # ic(y_pred, y_test)

    # calculate the mean squared error
    mse_gpr = mean_squared_error(y_test, y_pred)
    rmse_gpr = np.sqrt(mse_gpr)
    r2_gpr = 1- (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    print('---------------------------------------------------------------------')
    print(f'Validation MSE: {mse_gpr}, Validation RMSE: {rmse_gpr}, Validation R^2: {r2_gpr}')

    return -mse_gpr

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

# ------------------------------------------ Main Code ------------------------------------------
# Start time
start_time = time()   
# Load the Excel file
# Read the data
file_path = "Multi-Objective Optimisation\Dataset\IVRT-S.xlsx"
# ------------------------------------------ Whole Dataset ------------------------------------------
# Read the whole dataset
formulas_df = pd.read_excel(file_path, sheet_name='F-S')          # Read the whole dataset
c_df = pd.read_excel(file_path, sheet_name='R-S')              # Read the whole dataset

"""
Good: Ave-s4: random_state=12, shuffle=True; upper_bound = 1e-03, lower_bound = 1e3,(kind='ei', xi=1e-1).

"""

ic(formulas_df, c_df)

X = np.array(formulas_df.iloc[:, 1:].values)  # Extract features from formulation data

y = np.array(c_df.iloc[:, -1].values)     # TODO: Just use the final sampling data t28 as the target data (Extract labels from C data)
ic(X.shape, y.shape)

# Calculate the mean of the target data
# y_mean = np.mean(y, axis=1)
y_final = y
max_y = np.max(y_final)
max_y_hat = 167.64 # 

ic(y_final, y_final.shape, max_y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11) # Ave: 4, 9, 44  ave-s4: 12, 14; FS,SR: 4, 1, 6(0.99), 7(GOOD), 10, 11, 12  
# SR-Corrected:6,7,9,10
# LHS-S: 7, 6 (GOOD), 10 (Full bigger than others), 11 (half more than best), 14 (1e2, 1/3 more than)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# y_test_scaled = scaler_y.transform(y_test)
ic(X_train_scaled, y_train_scaled)

# # calculate the upper and lower bounds for the input data
# X_min = np.min(X_train_scaled, axis=0)
# X_max = np.max(X_train_scaled, axis=0)
# ic(X_min, X_max)

# # set the hyperparameter space for the NSGA-II algorithm to search
# xl = np.array([int(x)-1 for x in X_min])  # lower bounds based on the int(X_min)
# xu = np.array([int(x)+1 for x in X_max])  # upper bounds based on the int(X_max) 
# ic(xl, xu)

X_lower = np.array([20.00, 10.00, 10.00])  # Actual lower bounds of ibuprofen excipients
X_upper = np.array([30.00, 20.00, 20.00])   # Actual upper bounds of ibuprofen excipients

xl = scaler_X.transform([X_lower])[0]  # Standardize using training set scaler
xu = scaler_X.transform([X_upper])[0]
ic(xl, xu)

# Setting the hyperparameter space
upper_bound = 1e-4
lower_bound = 1e4
params_bounds = {
    'a0': (upper_bound, lower_bound),
    'a1': (upper_bound, lower_bound),
    'v0': (upper_bound, lower_bound),
    'v1': (upper_bound, lower_bound),
    'wl1': (upper_bound, lower_bound),
    'wl2': (upper_bound, lower_bound),
    'wl3': (upper_bound, lower_bound)
}

# Building the Bayesian optimization model
optimizer = BayesianOptimization(f=gpr_model_cv, pbounds=params_bounds, random_state=1)
# optimizer = BayesianOptimization(f=gpr_loocv_cv, pbounds=params_bounds, random_state=1)

# Balance the trade-off between exploration and exploitation
# acquisition_function = UtilityFunction(kind='ucb', kappa=1)  # Upper Confidence Bound (UCB) for exploration
acquisition_function = UtilityFunction(kind='ei', xi=0.1)  # Expected Improvement (EI) for exploitation
# acquisition_function = UtilityFunction(kind='poi', xi=1e-1)  # Probability of Improvement (POI) for exploitation

# Running the optimization
optimizer.maximize(
                    init_points=2,
                    n_iter=40,
                    acquisition_function=acquisition_function   
                    )

# get the best hyperparameters
best_params = optimizer.max['params']
# Save the best hyperparameters to an Excel file
pd.DataFrame([best_params]).to_excel('Multi-Objective Optimisation\Dataset\\fulldata-s\Bayesian_Optimisation_Results_fulldata.xlsx', index=False)

best_gpr = GaussianProcessRegressor(kernel=YuKernel(best_params['v0'],
                                                    [best_params['wl1'], best_params['wl2'], best_params['wl3']],
                                                    best_params['a0'],
                                                    best_params['a1'],
                                                    best_params['v1'])
                                                    # , alpha=noise_var
                                                    , n_restarts_optimizer=10, alpha=1e-10
                                                    )  # Create a new Gaussian Process Regression model using the best parameters

best_gpr.fit(X_train_scaled, y_train_scaled)  # Train the model

y_pred_test, sigma_test = best_gpr.predict(X_test_scaled, return_std=True)
y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
# sigma_test = sigma_test * scaler_y.scale_
sigma_test = sigma_test[0]

ic(y_pred_test, y_test, sigma_test)
# ic(X_test_scaled, X_test_scaled.shape)

mse, rmse, r_squared, mae, evs, mape = evaluate(y_test, y_pred_test)

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

end_time = time()

elapsed_time = end_time - start_time

X_label = pd.read_excel('Gassian Processes\Permeation_Kernel\IVPT Dataset.xlsx', sheet_name='F1', usecols=[0])  # 从 'F1' 表格中获取时间数据

print(f'Time elapsed: \033[93m{str(timedelta(seconds=elapsed_time))}\033[0m')  # 使用 timedelta 格式化时间

# Calculate the Kolmogorov-Smirnov statistic and the Jensen-Shannon divergence
ks_stat, ks_pvalue = ks_2samp(y_test.flatten(), y_pred_test.flatten())
print(f'Kolmogorov-Smirnov Statistic: {ks_stat}, p-value: {ks_pvalue}')

bins = np.histogram_bin_edges(y_test.flatten(), bins='auto', range=(y_test.flatten().min(), y_test.flatten().max()))
hist1, _ = np.histogram(y_test.flatten(), bins=bins, density=True)
hist2, _ = np.histogram(y_pred_test.flatten(), bins=bins, density=True)

js_divergence = jensenshannon(hist1, hist2)
print(f'Jensen-Shannon Divergence: {js_divergence}')

# ------------------------------------------ Multi-objective Optimization ------------------------------------------
class MyProblem(ElementwiseProblem):

    def __init__(self, best_gpr, scaler_X, scaler_y):
        super().__init__(n_var=3,  # Number of input features
                         n_obj=2,  # Number of objectives
                         n_constr=0,  # Number of constraints
                         xl=xl,  # Lower bounds for input
                         xu=xu)  # Upper bounds for input
        
        self.best_gpr = best_gpr
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def _evaluate(self, x, out, *args, **kwargs):
        # x_scaled = self.scaler_X.transform([x])
        y_pred, y_std = self.best_gpr.predict([x], return_std=True)     # The output y_std is the standard deviation
        # ic(x, y_pred)

        y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        # y_std_original = y_std * self.scaler_y.scale_

        y_std_original = y_std[0]   
        y_pred_original = y_pred_original.squeeze()  # Remove dimensions of size 1

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


problem = MyProblem(best_gpr, scaler_X, scaler_y)

collect_pf = CollectParetoFronts()

# define the Crossover and Mutation
# crossover = {'name': 'real_sbx', 'prob': 0.9, 'eta': 15}    # the probability range between 0.7 - 0.95
# mutation = {'name': 'real_pm', 'eta': 20}       # the probability range between 0.05 - 0.2

algorithm = NSGA2(pop_size=10) 


res = minimize(problem,
               algorithm,
               termination=('n_gen', 50),   
               callback=collect_pf,
               verbose=True,
               seed=1)


print("The best solution by NSGA-II algorithm is: ")
for i in range(len(res.X)):
    print(f"Solution-[{i}]: X = {np.round(res.X[i], 3)}, F = {np.round(res.F[i], 3)}, X_original= {np.round(scaler_X.inverse_transform([res.X[i]]), 3)}")

# Save the Pareto front and the corresponding variables to a xlxs file
pareto_front_df = pd.DataFrame(-res.F, columns=['Mean', 'Std'])
pareto_front_df['X'] = [scaler_X.inverse_transform([x]) for x in res.X]
# pareto_front_df.to_excel(r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Multi-Objective Optimisation\Pareto_animation\pareto_front.xlsx', index=False)
pareto_front_df.to_excel("Multi-Objective Optimisation\Dataset\\fulldata-s\pareto_front_fulldata.xlsx", index=False)


# plot = Scatter()
# plot.add(res.F, color="blue", alpha=0.5)
# plot.show()

get_mean = []
get_mean = []
get_std = []

# Plotting
plt.figure(figsize=(10, 6))
for i, pareto_front in enumerate(collect_pf.pareto_fronts):
    # Assume mean is at the first position of the objective function, std at the second
    mean_values = -pareto_front[:, 0]  # Use negative sign because minimization was performed on negative mean
    std_values = -pareto_front[:, 1]
    plt.scatter(mean_values, std_values, label=f'Iteration {i+1}', alpha=0.7)

plt.title('Pareto Fronts Over Iterations')
plt.xlabel('Prediction Mean')
plt.ylabel('Prediction Standard Deviation (Uncertainty)')
plt.grid(True)

plt.savefig('Multi-Objective Optimisation\Pareto_animation\\fulldata-s\pareto_fronts_overview_fulldata.png', dpi=300, bbox_inches='tight')
# plt.show()

# Save figure to file
folder_path = 'Multi-Objective Optimisation\Pareto_animation\\fulldata-s'

# Reconstruct the max_y that is a point (it's our response) at the 28th hour (final sampling time). TODO: take care of the max_y once I forget it.
# max_y = 336.67          

for i, pareto_front in enumerate(collect_pf.pareto_fronts):
    # plt.figure(figsize=(10, 8))
    mean_values = -pareto_front[:, 0]
    std_values = -pareto_front[:, 1]
    plt.scatter(mean_values, std_values, c='red', edgecolors='k', alpha=0.7)
    # ic(mean_values, std_values)

    # Calculate the maximum cumulative concentration value in current pareto front iteration
    cumulative_concentrations = np.add(-mean_values, std_values)
    # ic(cumulative_concentrations)
    max_index = np.argmax(cumulative_concentrations)
    max_point_mean = mean_values[max_index]
    max_point_std = std_values[max_index]

    # Draw a vertical dashed line at the highest cumulative concentration point // Change to draw the highest cumulative of permeated in current dataset, from 'max_point_mean' to 'max_y'
    plt.axvline(x=max_y, color='black', linestyle='--', label='The incumbent best', alpha=0.8)

    # Add text label for the highest cumulative concentration point
    # plt.text(max_point_mean, max_point_std, f'({max_point_mean:.3f}, {max_point_std:.3f})',
    #          color='black', verticalalignment='top')
    
    plt.title(f'Pareto Front at Iteration {i+1}')
    plt.xlabel('Prediction Mean')
    plt.ylabel('Prediction Standard Deviation (Uncertainty)')
    plt.legend()
    # plt.grid(True)
    
    # Build the full file path
    file_name = f'Pareto_Iteration_{i+1}.png'
    file_path = os.path.join(folder_path, file_name)
    
    # Save the figure to the specified path
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory
# Plot the dynamic Pareto front over iterations using animation
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=50, alpha=0.5)

xlim_min = -min([pf[:, 0].min() for pf in collect_pf.pareto_fronts])
xlim_max = -max([pf[:, 0].max() for pf in collect_pf.pareto_fronts])
ylim_min = -min([pf[:, 1].min() for pf in collect_pf.pareto_fronts])
ylim_max = -max([pf[:, 1].max() for pf in collect_pf.pareto_fronts])
ax.set_xlim(xlim_min-2, xlim_max+2)
ax.set_ylim(ylim_min-0.001, ylim_max+0.001)
ax.set_title('Pareto Fronts Over Iterations')
ax.set_xlabel('Prediction Mean')
ax.set_ylabel('Prediction Standard Deviation (Uncertainty)')

# Initialize dashed line and text objects
vline = ax.axvline(x=xlim_min, color='black', linestyle='--', alpha=0.8)
text = ax.text(xlim_min, ylim_min, '', fontsize=9, color='black', verticalalignment='top')

def init():
    scat.set_offsets(np.empty((0, 2)))
    vline.set_xdata([xlim_min, xlim_min])  # set the initial x-axis position of the vertical line
    text.set_text('')  # 清空文本
    return scat, vline, text

def update(frame):
    pareto_front = collect_pf.pareto_fronts[frame]
    mean_values = -pareto_front[:, 0]
    std_values = -pareto_front[:, 1]
    data = np.column_stack((mean_values, std_values))
    scat.set_offsets(data)

    # Update the vertical line
    cumulative_concentrations = np.add(mean_values, std_values)
    max_index = np.argmax(cumulative_concentrations)
    max_point_mean = mean_values[max_index]
    vline.set_xdata([max_point_mean, max_point_mean])

    max_point_std = std_values[max_index]

    # Update show the text label of the vertical line
    max_point_mean = max_y                                  # TODO: caution this place that has been changed to the max_y
    text.set_position((max_point_mean, max_point_std))
    text.set_text(f'({max_point_mean:.3f}, {max_point_std:.3f})')

    ax.set_title(f'Iteration-{frame + 1}')
    print(f'Updating frame {frame}: data size {data.shape}')  # 调试信息
    return scat, vline, text

ani = FuncAnimation(fig, update, frames=len(collect_pf.pareto_fronts),
                    init_func=init, blit=False, repeat_delay=50000)


plt.show()

# If you need to save the animation as a file
ani.save("Multi-Objective Optimisation\Pareto_animation\\fulldata-s\pareto_animation_fulldata.gif", writer='imagemagick', fps=2)

# Make decisions based on multi-objective optimization results
for i in range(len(res.X)):
    x = res.X[i]
    mean_pred, std_pred = res.F[i]
    get_mean.append(-mean_pred)
    get_std.append(-std_pred)

get_mean = np.array(get_mean)
get_std = np.array(get_std)

ic(get_mean, get_std)               # Pareto solution mean and std
# Plot the pure mean&std with threshold of best concentration
plt.errorbar(range(len(get_mean)), get_mean, yerr=get_std, fmt='o', ecolor='blue', capsize=3, color='red', marker='o', mfc='white', mec='red', mew=2)

plt.axhline(y=max_y, color='grey', linestyle='--', label='The incumbent best')    # TODO: This line should be the highest line of current history dataset.
# plt.text(0, max_point_Efficient_solution_cumulative_concentration, f'({max_point_mean:.3f}, {max_point_std:.3f})', color='black', verticalalignment='bottom')
plt.xlabel('Pareto Points No.')
plt.ylabel('Cumulative amount of ibu release (μg/cm²)')
plt.legend()
plt.savefig('Multi-Objective Optimisation\Pareto_animation\\fulldata-s\cumulative_concentration.png', dpi=300, bbox_inches='tight')
plt.show()