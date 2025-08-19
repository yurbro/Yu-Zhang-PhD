#/usr/bin/env python
# coding: utf-8
# Author: Yu Zhang - University of Surrey
# Date: 01/05/2024 11:32

import numpy as np
import pandas as pd
from icecream import ic
from Utils.yukernel import YuKernel
from Utils.gpr_model_cv import gpr_model_cv_update_single_formula, create_params_bounds, evaluate_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from Utils.decision_function import decision_expected_improvement
from Utils.quantification_uncertainty import q_uncertainty
from Early_Decision.GPR_BO_Decision.Bin.dynamic_bayesian_optimization import run_dynamic_bayesian_optimization
from Utils.probability_of_reaching_better_permeation import probability_of_prediction


# Define the decision rules

file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
# file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
Initial_data = pd.read_excel(file_name, sheet_name='Initial-Data')
test_data = pd.read_excel(file_name, sheet_name='F-4')
title_columns = Initial_data.columns.drop('Time')

# Initial position setting
Initial_position = 7
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

# Build the Bayesian optimization model
optimizer = BayesianOptimization(
f=lambda **params: gpr_model_cv_update_single_formula(
    params['v0'], params['a0'], params['a1'], params['v1'], 
    [params[f'wl{i+1}'] for i in range(X_train.shape[1])], 
    X_train, y_train, X_test, y_test, scaler_y
        ),
    pbounds=params_bounds,
    random_state=1,
    )

# Balance the exploration and exploitation
# acquisition_function = UtilityFunction(kind="ucb", kappa=1, xi=0.0)
acquisition_function = UtilityFunction(kind="ei", kappa=0.1, xi=0.0)

# Run the optimization
optimizer.maximize(init_points=5, n_iter=20, acquisition_function=acquisition_function)

# Get the best parameters
best_params = optimizer.max['params']

# Use the best parameters to train the model
best_gpr = GaussianProcessRegressor(kernel=YuKernel(best_params['v0'],
                                                    [best_params[f'wl{i+1}'] for i in range(X_train.shape[1])],
                                                    best_params['a0'],
                                                    best_params['a1'],
                                                    best_params['v1']))

# best_params, best_gpr = run_dynamic_bayesian_optimization(X, y, params_bounds,n_iterations=100)

# Fit the model
best_gpr.fit(X_train_scaled, y_train_scaled)

# predict the values on the test set
y_pred_initial, sigma_initial = best_gpr.predict(X_test_scaled, return_std=True)

# Transform the predicted data to the original scale
y_pred_initial = scaler_y.inverse_transform(y_pred_initial)
sigma_initial = sigma_initial[0]
ic(y_pred_initial, sigma_initial)
# Calculate metrics by using evaluate_metrics function
evaluate_metrics(y_test, y_pred_initial)

# ----------------------------------------------------- Real data prediction -----------------------------------------------------
# Real data prediction
X_test_f = test_data[title_columns[1:Initial_position]].values
y_test_f = test_data[title_columns[Initial_position:]].values

# Standardize the test data
X_test_f_scaled = scaler_X.transform(X_test_f)
y_test_f_scaled = scaler_y.transform(y_test_f)

# Predict the values on the test set
y_pred_f, sigma_f = best_gpr.predict(X_test_f_scaled, return_std=True)

# Transform the predicted data to the original scale
y_pred_f = scaler_y.inverse_transform(y_pred_f)
sigma_f = sigma_f[0]

# Calculate metrics by using evaluate_metrics function
evaluate_metrics(y_test_f, y_pred_f)


# ----------------------------------------------------- Quantification of uncertainty -----------------------------------------------------
# Quantification of uncertainty
ic(y_pred_f, sigma_f)

prob_to_better = probability_of_prediction(y_pred_f, sigma_f, y)
ic(np.mean(y_pred_f, axis=1), np.mean(sigma_f, axis=1), prob_to_better)

# Get the title labels of the test data
title_labels_f = test_data['Time'].values
for tl, prob in zip(title_labels_f, prob_to_better):
    print(f"FranzCell: {tl} - Probability of reaching or exceeding the best permeation effect: {prob} %")














# ----------------------------------------------------- Decision making -----------------------------------------------------
# # Decision making
# # Calculate the EI value
# threshold = np.max(np.mean(y_test_f, axis=0))
# ic(threshold)
# ei_value, decision_result = decision_expected_improvement(best_gpr, scaler_X, scaler_y, X_test_f, np.max(np.mean(y_test_f, axis=0)), threshold=5)
# ic(ei_value)

# for i, ei in enumerate(ei_value.flatten()):
#     if ei > threshold:
#         print(f"样本 {i+1} 的 EI 为 {ei:.3f}，建议继续实验。")
#     else:
#         print(f"样本 {i+1} 的 EI 为 {ei:.3f}，无需进一步实验。")