import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from Utils.yukernel import YuKernel
from Utils.quantification_uncertainty import q_uncertainty
from Utils.gpr_model_cv_update_single_formula import gpr_model_cv_update_single_formula, evaluate_model, create_params_bounds


def run_dynamic_bayesian_optimization(X, y, params_bounds, n_iterations=25):
    """
    Run Bayesian optimization with dynamic adjustment of the acquisition function parameters.

    Args:
    X_train (np.array): Training features.
    y_train (np.array): Training target values.
    X_test (np.array): Testing features.
    y_test (np.array): Testing target values.
    params_bounds (dict): Parameter bounds for Bayesian Optimization.
    n_iterations (int): Number of iterations for the optimization process.

    Returns:
    dict: The best parameters found.
    GaussianProcessRegressor: The trained GPR model with the best parameters.
    """

    # Initialize scaler
    # Split the data into training and testing and get the indices of the test set
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=13)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=lambda **params: gpr_model_cv_update_single_formula(
            params['v0'], params['a0'], params['a1'], params['v1'], 
            [params[f'wl{i+1}'] for i in range(X_train.shape[1])], 
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_y
        ),
        pbounds=params_bounds,
        random_state=1,
    )

    # next points to probe

    # Initially set a neutral exploration-exploitation balance
    kappa = 2.5  # or another value based on your initial preference
    xi = 0.01    # small exploration

    # Optimization loop with dynamic acquisition function adjustment
    for i in range(n_iterations):
        # acquisition_function = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)
        acquisition_function = UtilityFunction(kind="ei", kappa=kappa, xi=xi)
        optimizer.maximize(init_points=0, n_iter=1, acquisition_function=acquisition_function)  # perform one iteration at a time
        # next_point_to_probe = optimizer.suggest(acquisition_function)
        # print(f'Next point to probe: {next_point_to_probe}')
        # Update the model with the best parameters found so far
        best_params = optimizer.max['params']
        best_gpr = GaussianProcessRegressor(kernel=YuKernel(best_params['v0'],
                                                            [best_params[f'wl{i+1}'] for i in range(X_train.shape[1])],
                                                            best_params['a0'],
                                                            best_params['a1'],
                                                            best_params['v1']))
        best_gpr.fit(X_train_scaled, y_train_scaled)
        y_pred_initial, sigma_initial = best_gpr.predict(X_test_scaled, return_std=True)

        # Transform the predicted data to the original scale
        y_pred_initial = scaler_y.inverse_transform(y_pred_initial)
        sigma_initial = sigma_initial[0]

        # Calculate the quantification of uncertainty
        Q_x, average_Q_x = q_uncertainty(y_pred_initial, sigma_initial, y_test)
        print(f'This iteration-{i} average_Q_x is: {average_Q_x}')

        # Adjust the acquisition function parameters based on the average Q(x)
        if average_Q_x < 1:
            kappa += 0.1  # Increase kappa to encourage more exploration if uncertainty is high
        else:
            kappa -= 0.1  # Decrease kappa to encourage more exploitation if predictions are reliable
        kappa = max(0.1, min(10, kappa))  # Ensure kappa stays within reasonable bounds

    return best_params, best_gpr

# # Example usage (assuming that data loading and preprocessing, and utility definitions are done)
# file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'
# Initial_data = pd.read_excel(file_name, sheet_name='Initial-Data')
# title_columns = Initial_data.columns.drop('Time')

# # Initial position setting
# Initial_position = 7

# # Define the input and output variables
# X = Initial_data[title_columns[1:Initial_position]].values
# y = Initial_data[title_columns[Initial_position:]].values

# params_bounds = create_params_bounds(X)  # Assuming create_params_bounds is defined
# best_params, best_gpr = run_dynamic_bayesian_optimization(X, y, params_bounds, n_iterations=20)
