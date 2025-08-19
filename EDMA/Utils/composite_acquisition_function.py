import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from quantification_uncertainty import q_uncertainty
from gpr_model_cv_update_single_formula import gpr_model_cv_update_single_formula, create_params_bounds, evaluate_metrics
from sklearn.model_selection import train_test_split
from yukernel import YuKernel

def calculate_EI(mu, sigma, y_max, xi=0.01):
    """
    Calculate the expected improvement at a given point.

    Args:
    mu (float): Predicted mean from the GP model.
    sigma (float): Predicted standard deviation from the GP model.
    y_max (float): Maximum observed value of the target function.
    xi (float): Exploration-exploitation trade-off parameter.

    Returns:
    float: The expected improvement value.
    """
    if sigma > 0:
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei
    return 0.0

def custom_acquisition_function(x, model, y_max, bounds):
    """
    Custom acquisition function that integrates model uncertainty quantification (Q).

    Args:
    x (np.array): Candidate points for sampling.
    model (GaussianProcessRegressor): Trained GP model.
    y_max (float): The maximum value of the target function observed so far.
    bounds (dict): Parameter bounds.

    Returns:
    float: Custom acquisition value for the given point x.
    """
    # Calculate traditional EI
    mu, sigma = model.predict(x, return_std=True)
    ei = calculate_EI(mu, sigma, y_max)  # Define this function to calculate EI based on mu, sigma, and y_max
    
    # Calculate Q value
    q = q_uncertainty(mu, sigma, x, confidence_level=1.96)  # Define this function to calculate Q based on model predictions and actual data
    
    # Combine EI and Q
    return ei * q

def optimize_with_custom_acquisition_function(model, X_train, y_train, scaler_y, bounds, n_iter=30):
    for _ in range(n_iter):
        # Assuming bounds is a list of tuples (min, max) for each dimension
        x_next_candidates = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (100, len(bounds)))
        acq_values = [custom_acquisition_function(x, model, X_train, y_train, scaler_y) for x in x_next_candidates]
        
        # Select the next point to sample
        x_next = x_next_candidates[np.argmax(acq_values)]

        # Sample the objective function (replace with actual function call if available)
        y_next = objective_function(x_next)

        # Update the training dataset
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

        # Refit the model
        model.fit(X_train, y_train)

    return model, X_train, y_train

def objective_function(v0, a0, a1, v1, wls, X_train_scaled, y_train_scaled, X_test_scaled, y_test):
    # 将参数转化为模型所需的形式
    wl = [wls[f'wl{i+1}'] for i in range(len(wls))]

    # 定义 GPR 模型
    gpr = GaussianProcessRegressor(kernel=YuKernel(v0, wl, a0, a1, v1), n_restarts_optimizer=10, alpha=1e-10)
    
    # 训练模型
    gpr.fit(X_train_scaled, y_train_scaled)

    # 对测试集进行预测
    y_pred = gpr.predict(X_test_scaled)
    y_pred= StandardScaler().inverse_transform(y_pred)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    
    return -mse 

# def run_bayesian_optimization(X_train_scaled, y_train_scaled, bounds):
#     # Assuming the GPR model and data preparation code is set up here
#     gpr = setup_gpr_model(X_train_scaled, y_train_scaled)
#     y_max = np.max(y_train_scaled)  # Current best value

#     optimizer = BayesianOptimization(
#         f=lambda x: -custom_acquisition_function(x, gpr, y_max, bounds),
#         pbounds=bounds,  # Example bounds
#         random_state=1,
#         verbose=2
#     )

#     optimizer.maximize(
#         init_points=2,
#         n_iter=10,
#     )

#     best_params = optimizer.max['params']
#     print("Best parameters:", best_params)

# def setup_gpr_model(v0, a0, a1, v1, wls, X_train, y_train, X_test, y_test):
#     # Combine all wavelengths into a single array from variable-length argument
#     wl = list(wls)

#     # Define the GPR model
#     gpr = GaussianProcessRegressor(kernel=YuKernel(v0, wl, a0, a1, v1), n_restarts_optimizer=10, alpha=1e-10)

#     # Standardize the training data
#     scaler_X = StandardScaler()
#     scaler_X.fit(X_train)  # Fit only on training data
#     X_train_scaled = scaler_X.transform(X_train)
#     X_test_scaled = scaler_X.transform(X_test)

#     # Standardize the target variable
#     scaler_y = StandardScaler()
#     scaler_y.fit(y_train)  # Fit only on training data
#     y_train_scaled = scaler_y.transform(y_train)

#     gpr.fit(X_train_scaled, y_train_scaled)

#     return gpr

# def optimize_with_custom_acquisition(X_train, y_train, bounds, n_iter=20):
#     """
#     Run Bayesian optimization using a custom acquisition function.

#     Args:
#     X_train (np.array): Training data features.
#     y_train (np.array): Training data targets.
#     bounds (dict): Bounds for the parameters.
#     n_iter (int): Number of iterations to perform.

#     Returns:
#     dict: Best parameters found.
#     """
#     def black_box_function(**params):
#         # Assume a function to evaluate the model or experiment

#         return gpr_model_cv_update_single_formula  # Define this function based on your experiment

#     optimizer = BayesianOptimization(
#         f=black_box_function,
#         pbounds=bounds,
#         random_state=1,
#         verbose=2
#     )

#     for _ in range(n_iter):
#         # Find the next sampling point with the highest custom acquisition value
#         next_point = optimizer.suggest(lambda x: custom_acquisition_function(x, model, y_max, bounds))
#         # Evaluate the black-box function
#         target = black_box_function(**next_point)
#         # Update the optimizer
#         optimizer.register(params=next_point, target=target)

#     return optimizer.max

# Example usage
# paras_bounds = create_params_bounds(X_train)  # Define parameter bounds

# best_params = optimize_with_custom_acquisition(X_train, y_train, paras_bounds, n_iter=20)
