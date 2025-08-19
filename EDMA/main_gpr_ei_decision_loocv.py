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
from Utils.difference_analysis import analyze_data, decide_method, calculate_probability

def count_trues_before_first_false(decision_results):
    count = 0
    for result in decision_results:
        if result == False:
            break
        count += 1
    return count

# Main
def main_split_data(Initial_data, test_data, title_columns, Initial_position, Formula, xi_values):
    # Initial position setting
    
    X = Initial_data[title_columns[:Initial_position]].values      # Get the input data
    y = Initial_data[title_columns[Initial_position:]].values       # Get the output data

    print(f'The Input varibles are: {title_columns[:Initial_position]}, the targets are: {title_columns[Initial_position:]}')

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
    # acquisition_function = UtilityFunction(kind="ucb", kappa=10, xi=0.0)
    acquisition_function = UtilityFunction(kind="ei", xi=xi_values)

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

    # best_params, best_gpr = run_dynamic_bayesian_optimization(X, y, params_bounds,n_iterations=100)

    # Fit the model
    best_gpr.fit(X_train_scaled, y_train_scaled)

    # predict the values on the real test dataset
    X_real = test_data[title_columns[:Initial_position]].values
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

    # ----------------------------------------------------- Decision-making -----------------------------------------------------
    # decision-making
    prob_to_better, total_prob, decision, current_best, ei_m1 = ei_decision(y_pred_f, sigma_f, y, 0.15, 0.1)
    # prob_to_better, current_best = probability_of_prediction(y_pred_f, sigma_f, y)
    ic(np.mean(y_pred_f, axis=1), np.mean(sigma_f, axis=1))
    print("Now the current best permeation value is: ", current_best)

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
    prob_to_better_v2, decision_v2, current_best_v2, ei_m2 = ei_decision_v2(y_pred_f, sigma_f, y, 0.15, 0.1)
    ic(prob_to_better_v2, prob_to_better_v2.shape)
    print("Now the current best permeation value is: ", current_best_v2)
    print(f'The expected improvement value is: {ei_m2}')
    # decision-making result
    if decision_v2:
        print("\033[92mThe decision is to continue the experiment based on decision method-2.\033[0m")
    else:
        print("\033[91mThe decision is to stop the experiment based on decision method-2.\033[0m")
        
    print(f'The total probability of reaching a better permeation value based on decision method-2 is: {np.round(prob_to_better_v2 * 100, 2)} %')

    # automatic choose the method to calculate the probability of reaching a better permeation value
    prob_to_better_auto, decision_auto, current_best_auto, ei_auto = ei_decision_auto(y_pred_f, sigma_f, y, 0.1)
    print("Now the current best permeation value is: ", current_best_auto)
    print(f'The expected improvement value is: {ei_auto}')
    # decision-making result
    if decision_auto:
        print("\033[92mThe decision is to continue the experiment based on auto decision method.\033[0m")
    else:
        print("\033[91mThe decision is to stop the experiment based on auto decision method.\033[0m")

    print(f'The total probability of reaching a better permeation value based on auto decision method is: {np.round(prob_to_better_auto * 100, 2)} %')
    # y_pred_f_mean = np.mean(y_pred_f, axis=1)
    # sigma_f_mean = np.mean(sigma_f, axis=1)
    # y_test_mean = np.mean(y_real, axis=1)
    # plot_entire_formulation_group(y_pred_f_mean, sigma_f_mean, y_test_mean, title_labels_r, Formula, Initial_position)
    # plot_prob_to_better(prob_to_better, title_labels_r)

    # # plot
    # for i in range(len(title_labels_r)):
    #     plot_bo(y_pred_f, sigma_f, y_test, Initial_position, i, title_labels_r[i])


    return prob_to_better, total_prob, title_labels_r, decision, prob_to_better_v2, current_best, prob_to_better_auto, ei_m1, ei_m2


if __name__ == '__main__':
    start_time = pd.Timestamp.now()
    # Load the data
    file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Loocv.xlsx'
    # file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
    Initial_data = pd.read_excel(file_name, sheet_name='Initial')
    Formula = 'F-15'
    test_data = pd.read_excel(file_name, sheet_name=Formula)
    title_columns = Initial_data.columns.drop('Time')

    # Initial_position = 6            # if Ip=4 means the initial position is 4, which is the first posterior value

    # prob_all, prob_total, title_real, decision = main_split_data(Initial_data, test_data, title_columns, Initial_position, Formula, 0.1)

    # Iteration 10 formulations and record their decision results and probabilities

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
    record_decision_result_m2 = []
    
    # set a counter to record the stop position
    stop_position = 0
    Delta_ei_initial = 0
    threshold = 0.2
    # Iterate the different input initial data from 3 to 9, and each time we can based on the previous decision to decide the experiment of current formulations whether to continue or stop.
    for i in range(3, 13):
        print(f"Now the initial position is: {i}")
        prob_all, prob_total, title_real, decision, prob_total_v2, current_best, prob_total_auto, ei_m1, ei_m2 = main_split_data(Initial_data, test_data, title_columns, i, Formula, 0.1)
        prob_total_r1.append(prob_total)
        prob_total_r2.append(prob_total_v2)
        prob_total_r_auto.append(prob_total_auto)
        record_current_best.append(current_best)
        record_ei.append(ei_m1)
        record_ei_m2.append(ei_m2)
        Delta_ei_m1 = ei_m1 - Delta_ei_initial
        Delta_ei_initial = ei_m1
        Delta_ei_m2 = ei_m2 - Delta_ei_initial
        Delta_ei_initial = ei_m2

        # decision-making based on method-1 and EI
        if prob_total < threshold and Delta_ei_m1 < 0:
        # if prob_total_v2 < threshold:
            decision_1_ei = False
            print(f'\033[91mWhen the input data has {i}-variables, the decision is to stop the experiment.\033[0m')
            # break
        else:
            decision_1_ei = True
            print(f'\033[92mWhen the input data has {i}-variables, the decision is to continue the experiment.\033[0m')
            # stop_position += 1

        record_decision_result.append(decision_1_ei)

        # decision-making based on method-2 and EI
        if prob_total_v2 < threshold and Delta_ei_m2 < 0:
        # if prob_total_v2 < threshold:
            decision_2_ei = False
            print(f'\033[91mWhen the input data has {i}-variables, the decision is to stop the experiment.\033[0m')
            # break
        else:
            decision_2_ei = True
            print(f'\033[92mWhen the input data has {i}-variables, the decision is to continue the experiment.\033[0m')
            # stop_position += 1
        record_decision_result_m2_ei.append(decision_2_ei)

        # decision-making based on method-1 
        if prob_total < threshold:
            decision_1 = False
            print(f'\033[91mWhen the input data has {i}-variables, the decision is to stop the experiment.\033[0m')
            # break
        else:
            decision_1 = True
            print(f'\033[92mWhen the input data has {i}-variables, the decision is to continue the experiment.\033[0m')
            # stop_position += 1

        record_decision_result_m1.append(decision_1)

        # decision-making based on method-2 
        if prob_total_v2 < threshold:
            decision_2 = False
            print(f'\033[91mWhen the input data has {i}-variables, the decision is to stop the experiment.\033[0m')
            # break
        else:
            decision_2 = True
            print(f'\033[92mWhen the input data has {i}-variables, the decision is to continue the experiment.\033[0m')
            # stop_position += 1

        record_decision_result_m2.append(decision_2)

    stop_position = count_trues_before_first_false(record_decision_result)
    print(f'The stop position based on Method_1 + EI is: {stop_position}')
    print(f'The decision results are: {record_decision_result}')

    stop_position_m2_ei = count_trues_before_first_false(record_decision_result_m2_ei)
    print(f'The stop position based on method-2 + EI is: {stop_position_m2_ei}')
    print(f'The decision results based on method-2 and EI are: {record_decision_result_m2_ei}')

    stop_position_m1 = count_trues_before_first_false(record_decision_result_m1)
    print(f'The stop position based on method-1 is: {stop_position_m1}')
    print(f'The decision results based on method-1 are: {record_decision_result_m1}')

    stop_position_m2 = count_trues_before_first_false(record_decision_result_m2)
    print(f'The stop position based on method-2 is: {stop_position_m2}')
    print(f'The decision results based on method-2 are: {record_decision_result_m2}')

    end_time = pd.Timestamp.now()
    print(f'The total time is: {end_time - start_time}')

    if False in record_decision_result:
        # Plot the probability of reaching a better permeation value and expected improvement value
        fig, ax1 = plt.subplots(figsize=(8, 6))
        # threshold = 0.5
        x_labels = [str(i) for i in range(3, 13)]

        # Plot left y-axis
        # ax1.plot(x_labels, prob_total_r1, label='Probability', color='b')
        ax1.plot(x_labels, prob_total_r1, label='Method-1', color='b')
        ax1.plot(x_labels, prob_total_r2, label='Method-2', color='r')
        ax1.axhline(y=threshold, color='grey', linestyle='--')
        ax1.axhline(0.2, color='grey', linestyle='--')
        ax1.axhline(0.3, color='grey', linestyle='--')
        ax1.axhline(0.4, color='grey', linestyle='--')
        ax1.axhline(0.5, color='grey', linestyle='--')
        ax1.axhline(0.6, color='grey', linestyle='--')
        ax1.axhline(0.7, color='grey', linestyle='--')
        ax1.axhline(0.8, color='grey', linestyle='--')
        ax1.axhline(0.9, color='grey', linestyle='--')
        ax1.axhline(1.0, color='grey', linestyle='--')
        ax1.axvline(x=stop_position, color='y', linestyle='--', label='Stop Point M1+EI')
        ax1.axvline(x=stop_position_m2_ei, color='purple', linestyle='--', label='Stop Point M1+EI')
        ax1.set_xlabel('The number of input data variables', fontsize=12)
        ax1.set_ylabel('Probability', color='b', fontszie=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title(f'Overall probability of reaching a better permeation value of {Formula}')

        # Plot right y-axis
        # ax2 = ax1.twinx()
        # ax2.plot(x_labels, record_ei, '-', label='EI Value M1', color='g')
        # ax2.plot(x_labels, record_ei_m2, '-', label='EI Value M2', color='orange')
        # ax2.set_ylabel('Expected Improvement', color='g')
        # ax2.tick_params(axis='y', labelcolor='g')

        # add the legend
        fig.legend(loc='upper right', fontsize='small')
        plt.show()

    else:
        # Plot the probability of reaching a better permeation value and expected improvement value
        fig, ax1 = plt.subplots(figsize=(8, 6))
        # threshold = 0.5
        x_labels = [str(i) for i in range(3, 13)]

        # Plot left y-axis
        # ax1.plot(x_labels, prob_total_r1, label='Probability', color='b')
        ax1.plot(x_labels, prob_total_r1, label='Method-1', color='b')
        ax1.plot(x_labels, prob_total_r2, label='Method-2', color='r')
        ax1.axhline(y=threshold, color='grey', linestyle='--')
        ax1.axhline(0.2, color='grey', linestyle='--')
        ax1.axhline(0.3, color='grey', linestyle='--')
        ax1.axhline(0.4, color='grey', linestyle='--')
        ax1.axhline(0.5, color='grey', linestyle='--')
        ax1.axhline(0.6, color='grey', linestyle='--')
        ax1.axhline(0.7, color='grey', linestyle='--')
        ax1.axhline(0.8, color='grey', linestyle='--')
        ax1.axhline(0.9, color='grey', linestyle='--')
        ax1.axhline(1.0, color='grey', linestyle='--')
        # ax1.axvline(x=stop_position, color='y', linestyle='--', label='Stop Point')
        ax1.set_xlabel('The number of input data variables')
        ax1.set_ylabel('Probability', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title(f'Overall probability of reaching a better permeation value of {Formula}')

        # Plot right y-axis
        # ax2 = ax1.twinx()
        # ax2.plot(x_labels, record_ei, '-', label='EI Value', color='g')
        # ax2.plot(x_labels, record_ei_m2, '-', label='EI Value M2', color='orange')
        # ax2.set_ylabel('Expected Improvement', color='g')
        # ax2.tick_params(axis='y', labelcolor='g')

        # add the legend
        fig.legend(loc='upper left', bbox_to_anchor=(0.8, 0.8), fontsize='medium')
        plt.show()

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

    # # plot the expected improvement value
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_labels, record_ei, '-o')
    # plt.xlabel('The number of input data variables')
    # plt.ylabel('Expected improvement value')
    # plt.title(f'Expected improvement value of {Formula} based on different input data variables')
    # plt.show()
