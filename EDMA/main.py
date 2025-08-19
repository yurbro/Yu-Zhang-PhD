#/usr/bin/env python
# coding: utf-8
# Author: Yu Zhang - University of Surrey
# Date: 01/05/2024 11:32

import numpy as np
import pandas as pd
from icecream import ic
from Utils.yukernel import YuKernel
from Utils.gpr_model_cv_update_single_formula import gpr_model_cv_update_single_formula, create_params_bounds, evaluate_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from Utils.decision_function import decision_objective
from Utils.quantification_uncertainty import q_uncertainty
from Early_Decision.GPR_BO_Decision.Bin.dynamic_bayesian_optimization import run_dynamic_bayesian_optimization


# Define the decision rules

file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
# file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
Initial_data = pd.read_excel(file_name, sheet_name='Initial-Data')
test_data = pd.read_excel(file_name, sheet_name='F-3')
title_columns = Initial_data.columns.drop('Time')

# Initial position setting
Initial_position = 9
X = Initial_data[title_columns[1:Initial_position]].values
y = Initial_data[title_columns[Initial_position:]].values

All_series_points_initial_data = Initial_data[title_columns[3:]].values       # Get all the series points include posterior values

# Split the data into training and testing and get the indices of the test set
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, 
                                                                                 range(len(y)), 
                                                                                 test_size=0.1, 
                                                                                 random_state=1)

# 获取 y_test 对应的完整序列数据
full_y_test_data = All_series_points_initial_data[indices_test]

# Standardize the training data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Setting the bounds for the parameters
params_bounds = create_params_bounds(X_train_scaled)

# # Build the Bayesian optimization model
# optimizer = BayesianOptimization(
# f=lambda **params: gpr_model_cv_update_single_formula(
#     params['v0'], params['a0'], params['a1'], params['v1'], 
#     [params[f'wl{i+1}'] for i in range(X_train.shape[1])], 
#     X_train, y_train, X_test, y_test, scaler_y
#         ),
#     pbounds=params_bounds,
#     random_state=1,
#     )

# # Balance the exploration and exploitation
# # acquisition_function = UtilityFunction(kind="ucb", kappa=10, xi=0.0)
# acquisition_function = UtilityFunction(kind="ei", kappa=0.01, xi=0.0)

# # Run the optimization
# optimizer.maximize(init_points=5, n_iter=20, acquisition_function=acquisition_function)

# # Get the best parameters
# best_params = optimizer.max['params']

# # Use the best parameters to train the model
# best_gpr = GaussianProcessRegressor(kernel=YuKernel(best_params['v0'],
#                                                     [best_params[f'wl{i+1}'] for i in range(X_train.shape[1])],
#                                                     best_params['a0'],
#                                                     best_params['a1'],
#                                                     best_params['v1']))

best_params, best_gpr = run_dynamic_bayesian_optimization(X, y, params_bounds,n_iterations=20)

# Fit the model
best_gpr.fit(X_train_scaled, y_train_scaled)

# predict the values on the test set
y_pred_initial, sigma_initial = best_gpr.predict(X_test_scaled, return_std=True)

# Transform the predicted data to the original scale
y_pred_initial = scaler_y.inverse_transform(y_pred_initial)
sigma_initial = sigma_initial[0]

# Calculate metrics by using evaluate_metrics function
evaluate_metrics(y_test, y_pred_initial)

# Calculate the quantification of uncertainty
Q_x, average_Q_x = q_uncertainty(y_pred_initial, sigma_initial, y_test)


# Get the labels for the x-axis of the plot
x_labels = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
start_position = Initial_position - 3
x_labels_pred = x_labels[start_position:]

# Plot the true and predicted values
plt.figure(figsize=(8, 6))  # 调整大小以适应标签
for i in range(y_test.shape[0]):
    # 绘制真实值的线
    plt.plot(x_labels, full_y_test_data[i, :], 'b-o', label=f'FranzCell-{i+1}', linewidth=2, markersize=5, alpha=0.6)  # 真实值用蓝色线和点表示

    # 绘制预测值的线和误差条
    plt.plot()
    plt.errorbar(x_labels_pred, y_pred_initial[i, :], yerr=1.96*sigma_initial[i, :], fmt='r-^', label=f'Predicted values Fz-{i+1}', capsize=5, linewidth=2, markersize=5)  # 预测值用红色线和点表示，带误差条
    plt.xlabel('Output Variables')
    plt.ylabel('Values')
    plt.title(f'Comparison of True and Predicted Values with Uncertainty for Formula-1')
    plt.xticks(rotation=45)  # 如果标签过多或过长，可以旋转以便更好的显示
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数,确保子图之间的空间以及与图形边缘的空间合适
    plt.show()



# -------------------------------------------------------------- Test the model --------------------------------------------------------------
# Import the test data
X_test_f = test_data[title_columns[1:Initial_position]].values
y_test_f = test_data[title_columns[Initial_position:]].values
All_series_points_test_f = test_data[title_columns[3:]].values       # Get all the series points include posterior values

# Standardize the test data
X_test_f_scaled = scaler_X.transform(X_test_f)

# Predict the values on the test set
y_pred_f, sigma_f = best_gpr.predict(X_test_f_scaled, return_std=True)

# Transform the predicted data to the original scale
y_pred_f = scaler_y.inverse_transform(y_pred_f)
# sigma_f = scaler_y.inverse_transform(sigma_f[0])
sigma_f = sigma_f[0]
ic(y_pred_f, sigma_f, y_test_f)

# Calculate metrics by using evaluate_metrics function
evaluate_metrics(y_test_f, y_pred_f)

# 假设 title_columns 已经定义并包含了你想要的列名
x_labels = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
start_position = Initial_position - 3
x_labels_pred = x_labels[start_position:]
# ic(x_labels)

# 遍历所有样本
# for idx in range(y_test_f.shape[0]):
plt.figure(figsize=(8, 6))  # 调整大小以适应标签
for i in range(y_test_f.shape[0]):
    # 绘制真实值的线
    plt.plot(x_labels, All_series_points_test_f[i, :], 'b-o', label=f'FranzCell-{i+1}', linewidth=2, markersize=5, alpha=0.6)  # 真实值用蓝色线和点表示

    # 绘制预测值的线和误差条
    plt.errorbar(x_labels_pred, y_pred_f[i, :], yerr=1.96*sigma_f[i, :], fmt='r-^', label=f'Predicted values Fz{i+1}', capsize=5, linewidth=2, markersize=5)  # 预测值用红色线和点表示，带误差条
    plt.xlabel('Output Variables')
    plt.ylabel('Values')
    plt.title(f'Comparison of True and Predicted Values with Uncertainty for Formula-1')
    plt.xticks(rotation=45)  # 如果标签过多或过长，可以旋转以便更好的显示
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数,确保子图之间的空间以及与图形边缘的空间合适
    plt.show()

# # Calculate the decision objective
# posterior_regions_each_moment = decision_objective(title_columns, Initial_position, y_test_scaled)

