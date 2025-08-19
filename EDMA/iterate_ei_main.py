#/usr/bin/env python
# coding: utf-8
# Author: Yu Zhang - University of Surrey
# Date: 01/05/2024 11:32

import numpy as np
import pandas as pd
from icecream import ic
from Utils.yukernel import YuKernel
from Utils.gpr_model_cv import gpr_model_cv_update_single_formula, create_params_bounds, evaluate_metrics, gpr_model_cv_prob, gpr_cv_score
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
from Utils.plotting import plot_bo, plot_entire_formulation_group, plot_prob_to_better, create_gif
from Utils.expected_improvement import custom_expected_improvement


def main_split_data(Initial_data, test_data, title_columns, Initial_position, Formula, xi_values):
    # Initial position setting
    X = Initial_data[title_columns[1:Initial_position]].values      # Get the input data
    y = Initial_data[title_columns[Initial_position:]].values       # Get the output data
    

    All_series_points_initial_data = Initial_data[title_columns[3:]].values       # Get all the series points include posterior values

    # Split the data into training and testing and get the indices of the test set
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y,      
                                                                                 range(len(y)), 
                                                                                 test_size=0.1, 
                                                                                 random_state=123)
    # Get the full test data
    full_y_test_data = All_series_points_initial_data[indices_test]
    Time_label = Initial_data['Time'].values
    title_labels_f = Time_label[indices_test]

    # Standardize the training data and test data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Setting the bounds for the parameters
    pd_upper = 1e3
    pd_lower = 1e-3
    params_bounds = create_params_bounds(X_train_scaled, pd_upper, pd_lower)

    # Build the Bayesian optimization model
    iter_num = 10
    optimizer = BayesianOptimization(
    f=lambda **params: gpr_model_cv_prob(
        params['v0'], params['a0'], params['a1'], params['v1'], 
        [params[f'wl{i+1}'] for i in range(X_train.shape[1])], 
        X_train, y_train, X_test, y_test, scaler_y, title_labels_f, Formula,  Initial_position, iter_num
            ),
        pbounds=params_bounds,
        random_state=1997,
        )

    # Balance the exploration and exploitation
    # acquisition_function = UtilityFunction(kind="ucb", kappa=1, xi=0.0)
    acquisition_function = UtilityFunction(kind="ei", xi=xi_values)

    # Run the optimization
    optimizer.maximize(init_points=5, n_iter=30, acquisition_function=acquisition_function)

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

    # predict the values on the real test dataset
    X_real = test_data[title_columns[1:Initial_position]].values
    y_real = test_data[title_columns[Initial_position:]].values

    X_real_scaled = scaler_X.transform(X_real)
    # y_real_scaled = scaler_y.transform(y_real)

    y_pred_f, sigma_f = best_gpr.predict(X_real_scaled, return_std=True)

    # Transform the predicted data to the original scale
    if y_pred_f.ndim == 1:
            y_pred_f = y_pred_f.reshape(-1, 1)
    y_pred_f = scaler_y.inverse_transform(y_pred_f)
    sigma_f = sigma_f[0]

    # Calculate metrics by using evaluate_metrics function
    evaluate_metrics(y_real, y_pred_f)

    # ----------------------------------------------------- Quantification of uncertainty -----------------------------------------------------
    # Quantification of uncertainty
    ic(y_pred_f, sigma_f)

    prob_to_better, current_best = probability_of_prediction(y_pred_f, sigma_f, y)
    ic(np.mean(y_pred_f, axis=1), np.mean(sigma_f, axis=1))
    print("Now the current best permeation value is: ", current_best)

    # Get the title labels of the test data
    title_labels_r = test_data['Time'].values
    prob_to_better_round = np.round(prob_to_better * 100, 2)
    for tl, prob in zip(title_labels_r, prob_to_better_round):
        print(f"FranzCell: {tl} - Probability of reaching or exceeding the best permeation effect: {prob} %")

    print(f"The average probability of reaching or exceeding the best permeation effect is: {np.round(np.mean(prob_to_better_round), 2)} %")

    total_prob = np.mean(prob_to_better_round)

    return prob_to_better, total_prob, title_labels_r


if __name__ == '__main__':
    # Load the data
    file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula-3.xlsx'
    # file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
    Initial_data = pd.read_excel(file_name, sheet_name='Initial-Data_Small')
    Formula = 'F-2'
    test_data = pd.read_excel(file_name, sheet_name=Formula)
    title_columns = Initial_data.columns.drop('Time')

    Initial_position = 3            # if Ip=4 means the initial position is 4, which is the first posterior value

    prob_all, prob_total, title_real = main_split_data(Initial_data, test_data, title_columns, Initial_position, Formula, 0.1)


    # # setting the initial data
    # xi_start = 0.1
    # xi_end = 0.0001
    # n_iterations = 30
    # xi_values = np.linspace(xi_start, xi_end, n_iterations)     # calculate the xi values
    # record_prob = []        # record the probability of reaching better permeation value
    # record_prob_total = []  # record the total probability of reaching better permeation value

    # # Variables to track the maximum probability and corresponding xi value
    # max_prob_total = 0
    # max_xi_value = 0

    # # run the dynamic bo-ei
    # for i in range(n_iterations):
    #     print(f"Run the iteration {i+1}, xi value is: {xi_values[i]}")
    #     # main_split_data
    #     prob_all, prob_total, title_real = main_split_data(Initial_data, test_data, title_columns, Initial_position, Formula, xi_values[i])

    #     record_prob.append(prob_all)
    #     record_prob_total.append(prob_total)

    #     # Update the maximum probability and corresponding xi value
    #     if prob_total > max_prob_total:
    #         max_prob_total = prob_total
    #         max_xi_value = xi_values[i]

    # ic(record_prob, record_prob_total)   

    # print(f"The average probability of reaching or exceeding the best permeation effect is: {np.round(np.mean(record_prob_total), 2)} %")
    # print(f"The maximum total probability of {np.round(max_prob_total,2)} % was achieved with xi value {max_xi_value}") 

    # # plot the probability of reaching better permeation value
    # num_groups = len(record_prob)
    # num_bars = len(record_prob[0])

    # plt.figure(figsize=(8, 6))

    # for tr, i in zip(title_real, range(num_bars)):
    #     # 获取第i个概率值的所有实验组数据
    #     probabilities = [group[i] for group in record_prob]
    #     # 绘制折线图
    #     plt.plot(xi_values, probabilities, marker='o', label=f'{tr}', alpha=0.5)

    # # 设置图例
    # plt.legend()

    # # 设置x轴刻度和标签
    # plt.xticks(xi_values, [f'{xi:.4f}' for xi in xi_values], rotation=45)
    # plt.xlabel('Xi Values of Expected Improvement')
    # plt.ylabel('Probability (%)')

    # # 设置标题
    # plt.title(f'Probabilities for Different xi Values in Each Experiment Group - {Formula}')

    # # 显示图形
    # plt.tight_layout()  
    # plt.show()

    # # plot the total probability of reaching better permeation value
    # plt.figure(figsize=(8, 6))
    # plt.plot(xi_values, record_prob_total, marker='o', color='r', alpha=0.5, label='Total Probability')

    # # 设置x轴刻度和标签
    # plt.xticks(xi_values, [f'{xi:.4f}' for xi in xi_values], rotation=45)
    # plt.xlabel('Xi Values of Expected Improvement')
    # plt.ylabel('Total Probability (%)')

    # # 设置标题
    # plt.title(f'Total Probability for Different xi Values {Formula}')

    # # 显示图形
    # plt.tight_layout()  
    # plt.show()