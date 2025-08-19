
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

    return y_pred_f, rmse, r_squared

def main(Initial_position, num):

    # file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Real-2.xlsx'
    file_name = r'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Real-2.xlsx'
    Initial_data = pd.read_excel(file_name, sheet_name='GPR_metric')

    # Initial_data = pd.read_excel(file_name, sheet_name='Initial-Data_Small')
    # test_data = pd.read_excel(file_name, sheet_name='F-3')
    title_columns = Initial_data.columns.drop('Time')

    # Initial position setting
    
    X = Initial_data[title_columns[0:Initial_position]].values
    y = Initial_data[title_columns[-1]].values.reshape(-1, 1)

    ic(X, y)

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
    y_pred_gpr_yukernel, rmse_gpr_yukernel, r2 = gpr_yukernel(X_train, y_train, X_test, y_test, scaler_y, params_bounds, X_train_scaled, y_train_scaled, X_test_scaled)
    ic(y_pred_gpr_yukernel)

    # plot the comparison of the performance of different models
    y_test = y_test[:, -1]
    y_pred = y_pred_gpr_yukernel[:, -1]
    
    # plt.scatter(y_test, y_pred_gpr_yukernel[:, -1], color='b', alpha=0.5)

    # # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', linewidth=2, label='Identity Line')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Diagonal Line')
    # plt.xlabel('Measured cumulative amount of Ibu (μg/cm²)', fontsize=12)
    # plt.ylabel('Predicted cumulative amount of Ibu (μg/cm²)', fontsize=12)
    # plt.legend(fontsize=12)
    # # plt.title(f'Comparison of Measured vs. Predicted in different models - {Initial_position} variables')
    # # plt.savefig(rf'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Comparison_Models-{i}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(rf'C:\Users\yuzha\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Comparison_Models-{i}.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return y_test, y_pred, rmse_gpr_yukernel, r2


if __name__ == '__main__':
    
    # Record
    record_gpr_rmse = []
    record_gpr_r2 = []
    record_gpr_pred = []
    record_y_ture = []

    # iteration different initial position to get the rmse values and plot the comparison of the performance of different models
    for i in range(3,10):

        Initial_position = i

        y_test, y_pred, rmse, r2 = main(Initial_position, i)

        record_gpr_pred.append(y_pred)
        record_y_ture.append(y_test)
        record_gpr_rmse.append(rmse)
        record_gpr_r2.append(r2)
    
    # plot all results in one figure
    colors = ['b', 'g', 'r', 'c', 'm']

    for i in range(len(record_gpr_pred)):
        plt.scatter(record_y_ture[i], record_gpr_pred[i], alpha=0.6,
        color=colors[i % len(colors)], label=f'Iteration {i + 1}')
    
    # 添加参考对角线
    plt.plot([min(np.concatenate(record_y_ture)), max(np.concatenate(record_y_ture))],
            [min(np.concatenate(record_y_ture)), max(np.concatenate(record_y_ture))],
            'k--', label='Diagonal Line')

    # 图例和标签
    plt.xlabel('Measured cumulative amount of Ibu (μg/cm²)', fontsize=12)
    plt.ylabel('Predicted cumulative amount of Ibu (μg/cm²)', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Measured vs Predicted Values Across Iterations', fontsize=14)
    plt.grid(True)
    plt.show()

    # average rmse
    ave_rmse = np.average(record_gpr_rmse)
    ave_r2 = np.average(record_gpr_r2)
    print(f"The average rmse is {ave_rmse}, and average r2 is {ave_r2}")

    # # plot the comparison of the performance of different models
    # plt.plot(range(3,10), record_gpr_yukernel, color='b', label='(a)', alpha=0.7)
    # plt.xlabel('The number of input data variables')
    # plt.ylabel('RMSE')
    # plt.legend()
    # # plt.title('Comparison of RMSE in different models')
    # plt.show()