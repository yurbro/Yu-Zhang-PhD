
import numpy as np
import pandas as pd
from Utils.yukernel import YuKernel
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from Utils.gpr_model_cv import gpr_model_cv_update_single_formula, create_params_bounds, evaluate_metrics
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from icecream import ic

def gpr_yukernel(X_train, y_train, X_test, y_test, scaler_y, params_bounds, X_train_scaled, y_train_scaled, X_test_scaled):

    # Build the Gaussian Process Regressor model
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
    acquisition_function = UtilityFunction(kind="ucb", kappa=10, xi=0.0)
    # acquisition_function = UtilityFunction(kind="ei", xi=0.1)

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
    y_pred_f, sigma_f = best_gpr.predict(X_test_scaled, return_std=True)

    # Transform the predicted data to the original scale
    if y_pred_f.ndim == 1:
            y_pred_f = y_pred_f.reshape(-1, 1)
    y_pred_f = scaler_y.inverse_transform(y_pred_f)
    sigma_f = sigma_f[0]

    # Calculate metrics by using evaluate_metrics function
    mse, rmse, r_squared, mae, evs, mape = evaluate_metrics(y_test, y_pred_f)

    return y_pred_f, rmse

def gpr_basemodel(X_train, y_train, X_test, y_test, scaler_y, X_train_scaled, y_train_scaled, X_test_scaled):
    # Define the kernel: a product of a constant kernel and an RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    # Initialize the Gaussian Process Regressor
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=1997)

    # Fit the model on the scaled training data
    gpr.fit(X_train_scaled, y_train_scaled)

    # Predict the values on the scaled test dataset
    y_pred_f, sigma_f = gpr.predict(X_test_scaled, return_std=True)

    # Transform the predicted data to the original scale
    if y_pred_f.ndim == 1:
        y_pred_f = y_pred_f.reshape(-1, 1)
    y_pred_f = scaler_y.inverse_transform(y_pred_f)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred_f)
    print(f'MSE: {mse}')
    rmse = np.sqrt(mse)
    print(f'RMSE: {rmse}')

    return y_pred_f, rmse

def main(Initial_position, num):
    # file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula-2.xlsx'
    # file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Database_EachFormula.xlsx'

    file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Real-2.xlsx'
    # file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Real-2.xlsx'
    Initial_data = pd.read_excel(file_name, sheet_name='GPR_metric')

    # Initial_data = pd.read_excel(file_name, sheet_name='Initial-Data_Small')
    # test_data = pd.read_excel(file_name, sheet_name='F-3')
    title_columns = Initial_data.columns.drop('Time')

    # Initial position setting
    
    X = Initial_data[title_columns[1:Initial_position]].values
    y = Initial_data[title_columns[Initial_position:]].values

    All_series_points_initial_data = Initial_data[title_columns[3:]].values       # Get all the series points include posterior values

    # Split the data into training and testing and get the indices of the test set
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, 
                                                                                    range(len(y)), 
                                                                                    test_size=0.2, 
                                                                                    random_state=14)    # 14, 12

    # Standardize the training data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    ic(X_test_scaled.shape)

    # 检查是否有 NaN
    has_nan = np.isnan(X).any()
    print(f"X 中是否包含 NaN: {has_nan}")

    # 打印包含 NaN 的位置
    nan_positions = np.argwhere(np.isnan(X))
    print(f"包含 NaN 的位置: {nan_positions}")

    # Setting the bounds for the parameters
    pd_upper = 1e3
    pd_lower = 1e-3
    params_bounds = create_params_bounds(X_train_scaled, pd_upper, pd_lower)

    # get the y_pred_gpr_yukernel 
    y_pred_gpr_yukernel, rmse_gpr_yukernel = gpr_yukernel(X_train, y_train, X_test, y_test, scaler_y, params_bounds, X_train_scaled, y_train_scaled, X_test_scaled)
    ic(y_pred_gpr_yukernel)
    # get the y_pred_gpr_basemodel
    y_pred_gpr_basemodel, rmse_gpr_basement = gpr_basemodel(X_train, y_train, X_test, y_test, scaler_y, X_train_scaled, y_train_scaled, X_test_scaled)
    # ic(y_pred_gpr_basemodel)


    # plot the comparison of the performance of different models
    y_test = y_test[:, -1]
    plt.scatter(y_test, y_pred_gpr_yukernel[:, -1], color='b', alpha=0.5)

    # plt.scatter(y_test, y_pred_gpr_basemodel, color='g', label='(b)', alpha=0.5)

    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', linewidth=2, label='Identity Line')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Diagonal Line')
    plt.xlabel('Measured cumulative amount of Ibu (μg/cm²)', fontsize=12)
    plt.ylabel('Predicted cumulative amount of Ibu (μg/cm²)', fontsize=12)
    plt.legend(fontsize=12)
    # plt.title(f'Comparison of Measured vs. Predicted in different models - {Initial_position} variables')
    plt.savefig(rf'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Comparison_Models-{i}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(rf'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Comparison_Models-{i}.png', dpi=300, bbox_inches='tight')
    plt.show()


    return rmse_gpr_yukernel, rmse_gpr_basement


if __name__ == '__main__':
    # Initial_position = 7

    # Record
    record_gpr_yukernel = []
    record_gpr_basemodel = []
    # iteration different initial position to get the rmse values and plot the comparison of the performance of different models
    for i in range(3,10):

        Initial_position = i

        rmse_gpr_yukernel, rmse_gpr_basement = main(Initial_position, i)
        
        record_gpr_basemodel.append(rmse_gpr_basement)
        record_gpr_yukernel.append(rmse_gpr_yukernel)
    
    # plot the comparison of the performance of different models
    plt.plot(range(3,10), record_gpr_yukernel, color='b', label='(a)', alpha=0.7)
    plt.plot(range(3,10), record_gpr_basemodel, color='g', label='(b)', alpha=0.7)
    plt.xlabel('The number of input data variables')
    plt.ylabel('RMSE')
    plt.legend()
    # plt.title('Comparison of RMSE in different models')
    plt.show()