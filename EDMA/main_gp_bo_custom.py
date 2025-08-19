#/usr/bin/env python
# coding: utf-8
# Author: Yu Zhang - University of Surrey
# Date: 07/05/2024 16:57

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from icecream import ic
from Utils.yukernel import YuKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt.util import UtilityFunction
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.utils import create_result
from skopt.plots import plot_convergence
from Utils.expected_improvement import custom_expected_improvement
from Utils.probability_of_reaching_better_permeation import probability_of_prediction
from Utils.gpr_model_cv import gpr_model_cv_prob, evaluate_metrics, create_params_bounds
from Utils.plotting import plot_bo, plot_entire_formulation_group, plot_prob_to_better
from Utils.skopt_space import create_skopt_bounds



# main function
def main(Initial_data, test_data, title_columns, Initial_position, Formula):
    
    # Initial position setting
    X = Initial_data[title_columns[1:Initial_position]].values      # Get the input data
    y = Initial_data[title_columns[Initial_position:]].values       # Get the output data

    All_series_points_initial_data = Initial_data[title_columns[3:]].values       # Get all the series points include posterior values

    # Use the historical data as the training data
    X_train= X
    y_train = y

    # Use real data as the test data
    X_test = test_data[title_columns[1:Initial_position]].values
    y_test = test_data[title_columns[Initial_position:]].values
    # ic(X_train, y_train, X_test, y_test)
    ic(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Standardize the training data and test data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Setting the bounds for the parameters
    pd_upper = 1e3
    pd_lower = 1e-3
    space = create_skopt_bounds(X_train_scaled, pd_upper, pd_lower)

    # Build the Bayesian optimization model
    # optimizer = BayesianOptimization(
    # f=lambda **params: gpr_model_cv_prob(
    #     params['v0'], params['a0'], params['a1'], params['v1'], 
    #     [params[f'wl{i+1}'] for i in range(X_train.shape[1])], 
    #     X_train, y_train, X_test, y_test, scaler_y
    #         ),
    #     pbounds=params_bounds,
    #     random_state=1997,
    #     )
    kernel = YuKernel(v0=1.0, wl=np.array([1.0, 1.0, 1.0]), a0=1.0, a1=0.5, v1=0.1)
    gp_estimator = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-5)
    optimizer = Optimizer(
                        dimensions=space,
                        base_estimator=gp_estimator,  # 使用自定义 GP 估计器
                        random_state=42,
                        acq_func='EI'  # 选择期望提升作为采集函数
                        )

    # Balance the exploration and exploitation
    # acquisition_function = UtilityFunction(kind="ucb", kappa=10, xi=0.0)
    # acquisition_function = UtilityFunction(kind="ei", xi=0.0001)

    # Run the optimization
    # optimizer.maximize(init_points=5, n_iter=10, acquisition_function=acquisition_function)

    # Get the best parameters
    best_params = optimizer.max['params']
    print(f'The best parameters are: {best_params}')

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
    y_pred_f, sigma_f = best_gpr.predict(X_test_scaled, return_std=True)

    # Transform the predicted data to the original scale
    if y_pred_f.ndim == 1:
            y_pred_f = y_pred_f.reshape(-1, 1)
    y_pred_f = scaler_y.inverse_transform(y_pred_f)
    sigma_f = sigma_f[0]
    # ic(y_pred_initial, sigma_initial)
    # Calculate metrics by using evaluate_metrics function
    evaluate_metrics(y_test, y_pred_f)


    # ----------------------------------------------------- Quantification of uncertainty -----------------------------------------------------
    # Quantification of uncertainty
    ic(y_pred_f, sigma_f)

    prob_to_better, current_best = probability_of_prediction(y_pred_f, sigma_f, y)
    y_pred_f_mean = np.mean(y_pred_f, axis=1)
    sigma_f_mean = np.mean(sigma_f, axis=1)
    y_test_mean = np.mean(y_test, axis=1)
    ic(y_pred_f_mean, sigma_f_mean, y_test_mean)
    print("Now the current best permeation value is: ", current_best)

    # Get the title labels of the test data
    title_labels_f = test_data['Time'].values
    prob_to_better = np.round(prob_to_better * 100, 2)
    for tl, prob in zip(title_labels_f, prob_to_better):
        print(f"FranzCell: {tl} - Probability of reaching or exceeding the best permeation effect: {prob} %")

    print(f"The average probability of reaching or exceeding the best permeation effect is: {np.round(np.mean(prob_to_better), 2)} %")

    plot_entire_formulation_group(y_pred_f_mean, sigma_f_mean, y_test_mean, title_labels_f, Formula, Initial_position)
    plot_prob_to_better(prob_to_better, title_labels_f)

    # plot
    # for i in range(len(title_labels_f)):
    #     plot_bo(y_pred_f, sigma_f, y_test, Initial_position, i)



