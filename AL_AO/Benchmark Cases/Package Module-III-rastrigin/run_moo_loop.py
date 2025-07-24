#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   run_moo_loop.py
# Time    :   2025/06/06 11:39:27
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# import necessary libraries
import os, sys
import numpy as np
import pandas as pd
import time
from lhs_sample import lhs_samples
# from ackley_func import ackley_max, run_ackley
from rastrigin_func import ackley_max, run_ackley        # TODO: this is zakharove function actually
from multi_objective_optimisation import train_gpr_model, MultiObjectiveOptimisation
from acquisition_function import run_acquisition_function
from adaptive_weight_func import update_normalised_weights_and_allocate, calculate_accuracy
from tqdm import trange

def run_moo_initial_experiment(n_init, d, lb, ub, directory, method, benchmark):
    """
    Run the initial experiment for multi-objective optimisation.
    """
    # 1. Initialise the experiment data by using the lhs_sampling
    X_init = lhs_samples(n_init, lb, ub)  # Shape (20, 3)
    # 2.1 Evaluate the initial samples using the ackley function
    Y_init = ackley_max(X_init)  # Shape (20,)
    for xi, yi in zip(X_init, Y_init):
        print(f"x={xi.round(3)} → {benchmark}={yi:.4f}")
    # 2.2 Save the initial samples and their evaluations to an Excel file
    # keep three decimal places for better readability
    X_init = X_init.round(3)  # less than 3 decimal places
    Y_init = Y_init.round(3)
    df = pd.DataFrame(X_init, columns=[f"x{i+1}" for i in range(d)])
    df[benchmark] = Y_init
    file = f"lhs_samples_{benchmark}_{method}.xlsx"  # File name based on the method
    # check if the directory exists, if not, create it in the current working directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, file)
    # Save the DataFrame to an Excel file
    sheetname_initial = f'RUN-0-{method}'  # Initial run number is 0
    df.to_excel(path, index=False, sheet_name=sheetname_initial)
    print(f"LHS samples and {benchmark} values saved to '{file}'.")

    return path

def run_moo_loop(path, path_data, path_df, d, lb, ub,
                 run_num, method,
                 lower_bound, upper_bound,
                 popsize, gen,
                 total_selection, prev_weights,
                 accuracies, alpha, rounding, benchmark, savefig):
    """
    Run the multi-objective optimisation loop.
    """
    # 3. Define the parameters of moo
    file_path_optimisation = path
    # 3.1 Load the dataset from the Excel file and set parameters
    sheetname = f'RUN-{run_num}-{method}'
    df = pd.read_excel(file_path_optimisation, sheet_name=sheetname)  # Use the current run sheet
    X = df[[f"x{i+1}" for i in range(d)]].to_numpy(dtype=float)  # Shape (20, 3)
    Y = df[benchmark].to_numpy(dtype=float)  # Shape (20,)

    # 3.2 Detect the maximum value in Y
    Y_max = np.max(Y)
    print(f"Maximum value in Y: {Y_max:.4f}")

    # 4. Train the GPR model
    run_num = run_num + 1
    best_gpr, scaler_X, scaler_y= train_gpr_model(X, Y, lower_bound, upper_bound, run_num, method, path_data)

    # 5. Perform multi-objective optimisation
    xl = scaler_X.transform([lb])[0]  # Transform lower bounds using scaler
    xu = scaler_X.transform([ub])[0]  # Transform upper bounds using scaler
    MultiObjectiveOptimisation(d, best_gpr, scaler_X, scaler_y, popsize, gen, run_num, method, Y_max, xl, xu, path_data, path_df, savefig)
    print("------Successfully completed the multi-objective optimisation process------")

    print(f"To be here, we completed the multi-objective optimisation process with the method: {method} and run number: {run_num}.")

    # 6. Calculate the acquisition function values
    # directory_af = "Multi-Objective Optimisation\Benchmark\Package Module\Results"
    path_acquisition = os.path.join(path_df, f"Pareto-{run_num}-{method}", "Acq-Func")
    if not os.path.exists(path_acquisition):
        os.makedirs(path_acquisition)
    path_data_run = os.path.join(directory, f"RUN-{run_num}-{method }")
    path_data = directory

    # Check the method
    if method not in [f'EI-{d}D', f'HV-{d}D', f'RANDOM-{d}D']:
        new_weights, allocation = update_normalised_weights_and_allocate(
            prev_weights, accuracies, total_selection, alpha, rounding, epsilon=1e-9
            )
        k_EI = allocation['ei']  # Number of points to select based on EI
        # report the current weights and allocation
        print(f"Current weights: {new_weights}, Allocation: {allocation}")
    else:
        k_EI = 0  # If method is EI, HV or RANDOM, set k_EI to 0
        new_weights = None  # If method is EI, HV or RANDOM, set new_weights to None
        allocation = None   # If method is EI, HV or RANDOM, set allocation to None
        alpha = None  # If method is EI, HV or RANDOM, set alpha to None

    print(f"Running acquisition function for run number {run_num} with method {method}...")
    selected_path, selected_sheetname = run_acquisition_function(d, run_num, path_acquisition, path_data, path_data_run, method, total_selection, k_EI, benchmark, savefig)

    # 7. Calculate the results of selected points by using the benchmark function
    # 7.1 Load the Pareto front data
    run_ackley(d, run_num, selected_path, path_data_run, file_path_optimisation, method, selected_sheetname)
    print(f"The calculation of the Ackley function for the selected points is completed and save to the lhs_samples_ackley.xlsx.")

    # 7.2 Restructure the sheet contents

    sheet_prev = f'RUN-{run_num-1}-{method}'
    sheet_curr = f'RUN-{run_num}-{method}'

    with pd.ExcelFile(file_path_optimisation) as xls:
        if sheet_prev in xls.sheet_names and sheet_curr in xls.sheet_names:
            df_prev = pd.read_excel(xls, sheet_name=sheet_prev)
            df_curr = pd.read_excel(xls, sheet_name=sheet_curr)
            # Ensure the columns are consistent
            cols = [f"x{i+1}" for i in range(d)] + ["Ackley"]
            df_prev = df_prev[cols]
            df_curr = df_curr[cols]
            df_concat = pd.concat([df_prev, df_curr], ignore_index=True)
            with pd.ExcelWriter(file_path_optimisation, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_concat.to_excel(writer, sheet_name=sheet_curr, index=False)
            print(f"Sheet '{sheet_curr}' has been updated by concatenating '{sheet_prev}' and '{sheet_curr}'.")
        else:
            print(f"Sheet '{sheet_prev}' or '{sheet_curr}' not found in the Excel file.")

    print("All processes completed successfully. The results are saved in the specified directory.")

    return new_weights, allocation, path_data_run, selected_sheetname  # Return the updated weights and allocation for the next iteration

# create the main iteration function


if __name__ == "__main__":
    # Define parameters
    n_init = 10         # Initial sample size
    d = 5               # Dimension of the problem
    lb = np.array([-5.12] * d)  # Lower bounds
    ub = np.array([5.12] * d)   # Upper bounds
    method = f'RANDOM-{d}D'  # Method name for the optimisation; can be 'EI', 'HV', 'RANDOM', or 'PROPOSED'
    benchmark = 'Ackley'  # Benchmark function to be used, which is Zakharov in this case; more benchmarks can be added later
    directory = "Multi-Objective Optimisation\\Benchmark\\Package Module-III-rastrigin\\Dataset"  # Directory to save the initial experiment data
    directory_af = "Multi-Objective Optimisation\\Benchmark\\Package Module-III-rastrigin\\Results" # Directory to save the acquisition function results
    lower_bound, upper_bound = 1e-1, 1e1  # Define the bounds for tuning the GPR model
    popsize = 10  # Population size for the optimisation
    gen = 50  # Number of generations for the optimisation
    total_selection = 6  # Total number of selections to be made
    prev_weights = {"ei": 0.5, "hv": 0.5}  # Initial weights for EI and HV, which will be updated in each run
    accuracies = {"ei": 0.0, "hv": 0.0}  # Initial accuracies for EI and HV, because the first run dont have previous weights, next run will be updated
    alpha = 0.5  # Smoothing factor for weight updates, typically between 0 and 1 and it's a trade-off between exploration and exploitation
    rounding = 'floor'  # Rounding method for point allocation
    savefig = False  # Whether to save the figures or not, default is True or False 

    # set the start time for the experiment
    start_time = time.time()

    # Run the initial experiment
    path = run_moo_initial_experiment(n_init, d, lb, ub, directory, method, benchmark)

    # set the iteration parameters
    iteration = 30      # Number of iterations for the optimisation loop

    for i in trange(iteration, desc="Optimisation Progress", unit="iter"):
        print(f"--- Iteration {i+1}/{iteration} ---")
        new_weights, allocation, path_data_run, selected_sheetname = run_moo_loop(
                                                                        path=path,
                                                                        path_data=directory,
                                                                        path_df=directory_af,
                                                                        d=d,  # Dimension of the problem
                                                                        lb=lb,
                                                                        ub=ub,
                                                                        run_num=i,  # Run number starts from 0
                                                                        method=method,
                                                                        lower_bound=lower_bound,
                                                                        upper_bound=upper_bound,
                                                                        popsize=popsize,
                                                                        gen=gen,
                                                                        total_selection=total_selection,  # Total number of selections to be made
                                                                        prev_weights=prev_weights,  # Initial weights for EI and HV
                                                                        accuracies=accuracies,  # Initial accuracies for EI and HV
                                                                        alpha=alpha,  # Smoothing factor for weight updates
                                                                        rounding=rounding,  # Rounding method for point allocation
                                                                        benchmark=benchmark,  # Benchmark function to be used
                                                                        savefig=savefig  # Whether to save the figures or not
                                                                        )
        # Update the weights and accuracies for the next iteration
        prev_weights = new_weights

        # Check the method, if it's not the proposed method, then weights and allocation are not needed.
        if method not in [f'EI-{d}D', f'HV-{d}D', f'RANDOM-{d}D']:
            # The proposed method below
            # Calculate the accuracies based on the current selected points versus the best known values in updated historical data
            # Load the Pareto front data from the calculated results
            sheet_name_ei, sheet_name_hv = selected_sheetname[0], selected_sheetname[1]
            # Load the Pareto front data for EI and HV from the Excel file
            pareto_front_ei = pd.read_excel(f'{path_data_run}\\Top-RUN{i+1}-{method}_{benchmark}_result.xlsx', sheet_name=sheet_name_ei)
            pareto_front_hv = pd.read_excel(f'{path_data_run}\\Top-RUN{i+1}-{method}_{benchmark}_result.xlsx', sheet_name=sheet_name_hv)
            # Load the best known values from the previous historical data
            latest_df = pd.read_excel(f"{path}", sheet_name=f'RUN-{i+1}-{method}')
            y_current_best = latest_df[benchmark].max()
            print(f"Current iteration-{i+1} best {benchmark} value: {y_current_best:.4f}")
            # Calculate the accuracies for EI and HV
            accuracies['ei'] = calculate_accuracy(pareto_front_ei[benchmark].values, y_current_best)
            accuracies['hv'] = calculate_accuracy(pareto_front_hv[benchmark].values, y_current_best)  # Assuming HV is also based on Ackley values
            # Report the updated weights and accuracies
            print(f"Updated weights: {prev_weights}, Updated accuracies: {accuracies}")
            
            # Record the current best {benchmark} value, updated weights, accuracies, allocation, and method for this iteration
            output_path_ybest = f"{directory}\\{benchmark.lower()}_best_values_{method}.xlsx"
            sheet_name = f"Best_{benchmark}_Values"
            # Prepare a DataFrame with iteration, best value, weights, accuracies, allocation, and method
            record_y_best = pd.DataFrame({
                'Iteration': [i+1],
                f'{benchmark}_Best': [y_current_best],
                'Weight_EI': [prev_weights['ei']],
                'Weight_HV': [prev_weights['hv']],
                'Accuracy_EI': [accuracies['ei']],
                'Accuracy_HV': [accuracies['hv']],
                'Allocation_EI': [allocation.get('ei', None)],
                'Allocation_HV': [allocation.get('hv', None)],
                'Method': [method],
                'Alpha': [alpha],
                'timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                'timeconsumed': [time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))],
                'popsize': [popsize],
                'gen': [gen]
            })

        elif method in [f'EI-{d}D', f'HV-{d}D', f'RANDOM-{d}D']:
            # Calculate the accuracy for the EI, HV and Random methods
            pareto_front_method = pd.read_excel(f'{path_data_run}\\Top-RUN{i+1}-{method}_{benchmark}_result.xlsx', sheet_name=selected_sheetname)
            # Load the best known values from the previous historical data
            latest_df = pd.read_excel(f"{path}", sheet_name=f'RUN-{i+1}-{method}')
            y_current_best = latest_df[benchmark].max()
            acc_method = calculate_accuracy(pareto_front_method[benchmark].values, y_current_best)
            print(f"Current iteration-{i+1} best {benchmark} value: {y_current_best:.4f}, Accuracy for {method}: {acc_method:.4f}")
            # Record the current best {benchmark} value, accuracies, and method for this iteration
            output_path_ybest = f"{directory}\\{benchmark.lower()}_best_values_{method}.xlsx"
            sheet_name = f"Best_{benchmark}_Values"
            # Prepare a DataFrame with iteration, best value, accuracies, and method
            record_y_best = pd.DataFrame({
                'Iteration': [i+1],
                f'{benchmark}_Best': [y_current_best],
                'Weight_EI': [None],  # No weights for EI, HV or Random methods
                'Weight_HV': [None],
                f'Accuracy_{method}': [acc_method],  # Accuracy for the current method
                'Accuracy_EI': [acc_method if method == 'EI' else None],
                'Accuracy_HV': [acc_method if method == 'HV' else None],
                'Allocation_EI': [None],  # No allocation for EI, HV or Random methods
                'Allocation_HV': [None],
                'Method': [method],
                'Alpha': [None],  # No alpha for EI, HV or Random methods
                'timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                'timeconsumed': [time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))],
                'popsize': [popsize],
                'gen': [gen]
            })
        
        # If the file exists, append; else, create new
        if os.path.exists(output_path_ybest):
            try:
                existing = pd.read_excel(output_path_ybest, sheet_name=sheet_name)
            except Exception:
                existing = pd.DataFrame(columns=[
                    'Iteration', f'{benchmark}_Best', 'Weight_EI', 'Weight_HV',
                    'Accuracy_EI', 'Accuracy_HV', 'Allocation_EI', 'Allocation_HV', 'Method', 'Alpha', 'timestamp', 'timeconsumed',
                    'popsize', 'gen'
                ])
            updated = pd.concat([existing, record_y_best], ignore_index=True)
        else:
            updated = record_y_best
        # Save back to the same sheet, overwrite
        if os.path.exists(output_path_ybest):
            with pd.ExcelWriter(output_path_ybest, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                updated.to_excel(writer, index=False, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(output_path_ybest, engine='openpyxl', mode='w') as writer:
                updated.to_excel(writer, index=False, sheet_name=sheet_name)

        print(f"--- End of Iteration {i+1}/{iteration} ---\n")
    
    # 如果y_current_best结果为0，则终止循环
    if y_current_best == 0:
        print("The best Ackley value is 0, terminating the loop.")
        print("No further iterations will be performed.")
        sys.exit()

    print("All iterations completed successfully. The results are saved in the specified directory.")

    # Print the total time taken for the experiment
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for the experiment: {total_time // 3600:.0f} h, {total_time // 60:.0f} min, {total_time % 60:.2f} sec")

