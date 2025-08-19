#/usr/bin/env python
# coding: utf-8
# Author: Yu Zhang - University of Surrey
# Date: 13/05/2024 20:12

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
# from Early_Decision.GPR_BO_Decision.Bin.dynamic_bayesian_optimization import run_dynamic_bayesian_optimization
from Utils.probability_of_reaching_better_permeation import probability_of_prediction
from Utils.plotting import plot_bo, plot_entire_formulation_group, plot_prob_to_better, create_gif
from Utils.expected_impovement_decision import ei_decision, ei_decision_v2, ei_decision_auto
# from Utils.difference_analysis import analyze_data, decide_method, calculate_probability
from Utils.split_dataset_for_threshold_moo import create_train_test_sets
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

"""
    30/09/2024: This code is for seeking the threshold of multi-objective optimisation by using the NSGA-II algorithm.

    New version: 01/10/2024
    It's turn to mainly focus on finding the threshold of multi-objective optimisation by using the NSGA-II algorithm without any other no need functions.
    After split the dataset, we conbine the test data to predict instead of using one by one module to predict the data.
"""

def count_trues_before_first_false(decision_results):
    count = 0
    for result in decision_results:
        if result == False:
            break
        count += 1
    return count

def evaluate_metrics_moo(y_test_e, y_pred_test_e):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_e, y_pred_test_e)
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Calculate R-squared (R^2)
    r_squared = r2_score(y_test_e, y_pred_test_e)
    # r_squared = 1 - np.sum((y_test_e - y_pred_test_e)**2) / np.sum((y_test_e - np.mean(y_test_e))**2)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test_e, y_pred_test_e)
    # Calculate Explained Variance Score
    evs = explained_variance_score(y_test_e, y_pred_test_e)
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_e - y_pred_test_e) / y_test_e)) * 100

    table_data_real = [
            ["Mean Squared Error (MSE)", mse],
            ["Root Mean Squared Error (RMSE)", rmse],
            ["Mean Absolute Error (MAE)", mae],
            ["Explained Variance Score (EVS)", evs],
            ["Mean Absolute Percentage Error (MAPE)", mape],
            ["R-squared (R²)", f'\033[91m{r_squared}\033[0m']
            ]
        # Output results
    print(tabulate(table_data_real, headers=["Metric", "Value"], tablefmt="pretty"))

    return mse, rmse, r_squared, mae, evs, mape

# Main
def main_split_data(Initial_data, test_data, title_columns, Initial_position, Formula, xi_values):
    # Initial position setting
    
    X = Initial_data[title_columns[:Initial_position]].values      # Get the input data
    y = Initial_data[title_columns[-1]].values       # Get the output data

    # ic(X, y, X.shape, y.shape)
    print(f'The Input varibles are: {title_columns[:Initial_position]}, the targets are: {title_columns[Initial_position:]}')

    # Split the data into training and testing and get the indices of the test set
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y,      
                                                                                 range(len(y)), 
                                                                                 test_size=0.2, 
                                                                                 random_state=1)
    # Get the full test data
   
    Time_label = Initial_data['Time'].values
    title_labels_f = Time_label[indices_test]
    ic(title_labels_f)

    # Standardize the training data and test data
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

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
    f=lambda **params: gpr_model_cv_update_single_formula(
        params['v0'], params['a0'], params['a1'], params['v1'], 
        [params[f'wl{i+1}'] for i in range(X_train.shape[1])], 
        X_train, y_train, X_test, y_test, scaler_y
            ),
        pbounds=params_bounds,
        random_state=1997,
        )

    # Balance the exploration and exploitation
    acquisition_function = UtilityFunction(kind="ucb", kappa=2.576, xi=0.0)
    # acquisition_function = UtilityFunction(kind="ei", xi=1e-1)

    # Run the optimization
    optimizer.maximize(init_points=5, n_iter=20, acquisition_function=acquisition_function)

    # Get the best parameters
    best_params = optimizer.max['params']
    print(f'The best parameters are: {best_params}')

    # Use the best parameters to train the model
    best_gpr = GaussianProcessRegressor(kernel=YuKernel(best_params['v0'],
                                                        [best_params[f'wl{i+1}'] for i in range(X_train.shape[1])],
                                                        best_params['a0'],
                                                        best_params['a1'],
                                                        best_params['v1']))

    # Fit the model
    best_gpr.fit(X_train_scaled, y_train_scaled)

    # Predict the values on the test dataset
    y_pred_test, sigma = best_gpr.predict(X_test_scaled, return_std=True)

    # Transform the predicted data to the original scale
    if y_pred_test.ndim == 1:
        y_pred_test = y_pred_test.reshape(-1, 1)
    y_pred_test = scaler_y.inverse_transform(y_pred_test)
    sigma = sigma[0]

    # Calculate metrics by using evaluate_metrics function
    print("The evaluation metrics are in the X_test: ")
    ic(y_test, y_pred_test)
    evaluate_metrics_moo(y_test, y_pred_test)

    # predict the values on the real test dataset
    X_real = test_data[title_columns[:Initial_position]].values
    y_real = test_data[title_columns[-1]].values

    ic(X_real, y_real)

    X_real_scaled = scaler_X.transform(X_real)
    # y_real_scaled = scaler_y.transform(y_real)

    y_pred_f, sigma_f = best_gpr.predict(X_real_scaled, return_std=True)

    # Transform the predicted data to the original scale
    if y_pred_f.ndim == 1:
            y_pred_f = y_pred_f.reshape(-1, 1)
    y_pred_f = scaler_y.inverse_transform(y_pred_f)
    sigma_f = sigma_f[0]

    # Calculate metrics by using evaluate_metrics function
    print("The evaluation metrics are in the X_real data: ")
    ic(y_real, y_pred_f)
    evaluate_metrics_moo(y_real, y_pred_f)

    # ----------------------------------------------------- Quantification of uncertainty -----------------------------------------------------
    # Quantification of uncertainty
    ic(y_pred_f, sigma_f)

    # ----------------------------------------------------- Decision-making -----------------------------------------------------
    # decision-making
    prob_to_better, total_prob, decision, current_best, ei_m1 = ei_decision(y_pred_f, sigma_f, y, 0.12, 0.1)
    # prob_to_better, current_best = probability_of_prediction(y_pred_f, sigma_f, y)
    # ic(np.mean(y_pred_f), np.mean(sigma_f))
    # print("Now the current best permeation value is: ", current_best)

    # expected_improvement
    print(f'The expected improvement value is: {ei_m1}')

    # decision-making result
    if decision:
        print("\033[92mThe decision is to continue the experiment based on method-1.\033[0m")
    else:
        print("\033[91mThe decision is to stop the experiment based on method-1.\033[0m")

    # Get the title labels of the test data
    title_labels_r = test_data['Time'].values
    prob_to_better_round = np.round(prob_to_better * 100, 2)
    for tl, prob in zip(title_labels_r, prob_to_better_round):
        print(f"FranzCell: {tl} - Probability of reaching or exceeding the best permeation effect: {prob} %")

    print(f"The average probability of reaching or exceeding the best permeation effect is: {np.round(np.mean(prob_to_better_round), 2)} %")

    # method-2 to calculate the total probability of reaching a better permeation value
    prob_to_better_v2, decision_v2, current_best_v2, ei_m2 = ei_decision_v2(y_pred_f, sigma_f, y, 0.00, 0.1)
    ic(prob_to_better_v2, prob_to_better_v2.shape)
    print("Now the current best permeation value is: ", current_best_v2)
    print(f'The expected improvement value is: {ei_m2}')

    prob_to_better_auto = 0


    return prob_to_better, total_prob, title_labels_r, decision, prob_to_better_v2, current_best_v2, prob_to_better_auto, ei_m1, ei_m2


if __name__ == '__main__':
    start_time = pd.Timestamp.now()
    # Load the data
    # file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Loocv.xlsx'
    # Initial_data = pd.read_excel(file_name, sheet_name='Initial')

    # file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-MOO.xlsx'
    file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-MOO.xlsx'
    
    Formulas, Initial_data = create_train_test_sets(file_name=file_name)
    
    title_columns = Initial_data.columns.drop('Time')

    # record the prob. of different input data and different calculation methods
    prob_total_r1 = []
    prob_total_r2 = []
    prob_total_r_auto = []
    record_current_best = []
    record_ei = []
    record_ei_m2 = []
    record_decision_result = []
    record_decision_result_m2_ei= []
    record_decision_result_m1 = []
    
    prob_total_r2_dict = {}
    prob_total_r1_dict = {}
    
    # set a counter to record the stop position
    stop_position = 0
    Delta_ei_initial = 0
    threshold = 0.1
    # Iterate the different input initial data from 3 to 9, and each time we can based on the previous decision to decide the experiment of current formulations whether to continue or stop.
    conbine_data = pd.DataFrame()
    for f in tqdm(Formulas):
        print(f"Now the formulation is: {f}")
        record_decision_result_m2 = []
        prob_total_r2 = []
        test_data = pd.read_excel(file_name, sheet_name=f)

        # ic(test_data, test_data.shape,type(test_data))
        if isinstance(test_data, pd.DataFrame):
            conbine_data = pd.concat([conbine_data, test_data], ignore_index=True)
        else:
            print(f"Warning: The data of {f} is not a DataFrame.")

    i = 3
    
    test_data = conbine_data
    ic(test_data, test_data.shape, type(test_data))

    print(f"Now the initial position is: {i}")
    prob_all, prob_total, title_real, decision, prob_total_v2, current_best_v2, prob_total_auto, ei_m1, ei_m2 = main_split_data(Initial_data, test_data, title_columns, i, f, 0.1)
    
    prob_total_r_auto.append(prob_total_auto)
    # record_current_best.append(current_best)
    record_ei.append(ei_m1)
    record_ei_m2.append(ei_m2)

    # Get the title labels of the test data
    title_real = test_data['Time'].values
    prob_to_better_round = np.round(prob_all * 100, 2)
    for tl, prob in zip(title_real, prob_to_better_round):
        # add into prob_total_r2_dict
        prob_total_r1_dict[tl] = prob
        
    print(f'The probability of reaching a better permeation value based on method-1 is: {prob_total_r2_dict}')
    ic(prob_total_r1_dict)

    end_time = pd.Timestamp.now()
    print(f'The total time is: {end_time - start_time}')

    # Justify the greater than or less than the y_best value and record them, respectively.
    # 初始化两个列表用于记录大于和小于 y_best 的配方
    greater_than_y_best = []
    less_than_or_equal_to_y_best = []

    # 遍历 test formulation and compare the value with the y_best value
    for ff in Formulas:
        test_data = pd.read_excel(file_name, sheet_name=ff)
        formu = test_data[title_columns[-1]].values
        print(f'The formulation {ff} has the value: {formu[0]}')
        print(f'The current best value is: {current_best_v2}')
        if formu[0] > current_best_v2:
            greater_than_y_best.append(ff)
        else:
            less_than_or_equal_to_y_best.append(ff)
    
    print(f'The formulations greater than the current best value are: {greater_than_y_best}')
    print(f'The formulations less than or equal to the current best value are: {less_than_or_equal_to_y_best}')

    # Plot the histogram of the probability of reaching a better permeation value


    # 创建柱状图
    formulations = list(prob_total_r1_dict.keys())
    values = [prob*0.01 for prob in prob_total_r1_dict.values()]  # 每个配方对应的概率
    # 创建颜色列表，使用绿色表示大于 y_best，红色表示小于等于 y_best
    colors = ['green' if formulation in greater_than_y_best else 'red' for formulation in formulations]

    formulations = [1, 2, 3, 4]
    # plt.figure(figsize=(10, 6))
    plt.bar(formulations, values, color=colors)

    for y in [i/10 for i in range(1, 11)]:
        plt.axhline(y=y, color='grey', linestyle='--')

    # 添加标题和标签
    # plt.title('Probability of Better Permeation Value by Formulation', fontsize=14)
    plt.xlabel('Formulation No.', fontsize=12)
    plt.ylabel('Probability', fontsize=12)

    # 设置y轴范围和刻度
    plt.ylim(0, 1.0)
    plt.yticks([i/10 for i in range(11)])  # 0.1, 0.2, ..., 1.0

    # 显示图形
    # plt.savefig(r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Probability_better by Formulation.png', dpi=300, bbox_inches='tight')
    plt.savefig(r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Probability_better by Formulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # # Plot the probability of reaching a better permeation value
    # # fig = plt.figure(figsize=(8, 6))
    # x_labels = [str(i) for i in range(3, 13)]\
    # # Plot y-axis
    # for f in Formulas:
    #     plt.plot(x_labels, prob_total_r2_dict[f], label=f)
    # # plt.plot(x_labels, prob_total_r2, label='Probability', color='b')
    # # Draw horizontal lines from 0.1 to 1.0
    # for y in [i/10 for i in range(1, 11)]:
    #     plt.axhline(y=y, color='grey', linestyle='--')
    # plt.axhline(y=threshold, color='grey', linestyle='--')
    # plt.xlabel('The number of input data variables', fontsize=12)
    # plt.ylabel('Probability', fontsize=12)
    # # plt.title(f'Overall probability of reaching a better permeation value of each formulation')
    # plt.legend(fontsize=12)
    # # plt.savefig(r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Overall probability of reaching.png', dpi=300, bbox_inches='tight')
    # plt.savefig(r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Overall probability of exceeding.png', dpi=300, bbox_inches='tight')
    # plt.show()
    

    # # plot the probability of reaching a better permeation value
    # threshold = 0.5
    # x_labels = [str(i) for i in range(3, 13)]
    # plt.figure(figsize=(10, 6))
    # # plt.plot(x_labels, prob_total_r1, label='Method-1')
    # # plt.plot(x_labels, prob_total_r2, label='Method-2')
    # # plt.scatter(x_labels, prob_total_r_auto, marker='o', color='g')
    # # plt.plot(x_labels, prob_total_r_auto, label='Auto Method', linestyle='--', color='g')
    # plt.plot(x_labels, prob_total_r1, label='Probability')
    # # plot the threshold line
    # plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold=0.5')
    # # plt.axhline(y=0.5, color='g', linestyle='--', label='Threshold=0.5')
    # # plt.axvspan(0, 6, color='yellow', alpha=0.2)        # highlight the area
    # plt.xlabel('The number of input data variables')
    # plt.ylabel('Probability')
    # plt.title(f'Overall probability of reaching a better permeation value of {Formula}')
    # plt.legend()
    # plt.show()

    # # plot the current best permeation value
    # x_labels_current_best = [1, 2, 3, 4, 6, 8, 22, 24, 26, 28]
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_labels_current_best, record_current_best, '-o')
    # plt.xlabel('The number of input data variables')
    # plt.ylabel('Current best permeation value')
    # plt.axvspan(1, 8, color='yellow', alpha=0.2)        # highlight the area
    # plt.title(f'Current best permeation value of {Formula} based on different input data variables')
    # plt.show()
