from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from icecream import ic
from tabulate import tabulate
from Utils.yukernel import YuKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from Utils.quantification_uncertainty import q_uncertainty

def gpr_model_cv_update_single_formula(v0, a0, a1, v1, wls, X_train, y_train, X_test, y_test, scaler_y):
    # Combine all wavelengths into a single array from variable-length argument
    wl = list(wls)

    # Define the GPR model
    gpr = GaussianProcessRegressor(kernel=YuKernel(v0, wl, a0, a1, v1), n_restarts_optimizer=10, alpha=1e-10)

    # Standardize the training data
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)  # Fit only on training data
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize the target variable
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)  # Fit only on training data
    y_train_scaled = scaler_y.transform(y_train)

    # Fit the model
    gpr.fit(X_train_scaled, y_train_scaled)

    # Predict and evaluate
    return evaluate_model(gpr, X_test_scaled, y_test, scaler_y)

def evaluate_model(gpr, X_test_scaled, y_test, scaler_y, w1=0.7, w2=0.3):
    # Predict the values on the test set
    y_pred_t, sigma_t = gpr.predict(X_test_scaled, return_std=True)

    if y_pred_t.ndim == 1:
        y_pred_t = y_pred_t.reshape(-1, 1)

    # Transform the predicted data to the original scale
    y_pred_t = scaler_y.inverse_transform(y_pred_t)
    sigma_t = sigma_t[0]

    # Calculate the average_Q_x
    Q_x, average_Q_x = q_uncertainty(y_pred_t, sigma_t, y_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred_t)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_test - y_pred_t) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    # r2 = r2_score(y_test, y_pred_t)
    # ic(r2)

    # 计算复合得分
    score = w1 * mse + w2 * (1 - average_Q_x)
    var_y = np.var(y_test)
    nMSE = mse / var_y if var_y != 0 else mse
    Q_score = average_Q_x / (1 + nMSE)

    print('---------------------------------------------------------------------')
    print(f'Valid MSE: {mse}, Valid RMSE: {rmse}, Valid R^2: {r2}, , Composite_score: {Q_score}')

    return Q_score

def create_params_bounds(X_train):
    feature_count = X_train.shape[1]
    params_bounds = {
        'v0': (1e-3, 1e3),
        'a0': (1e-3, 1e3),
        'a1': (1e-3, 1e3),
        'v1': (1e-3, 1e3),
        **{f'wl{i+1}': (1e-3, 1e3) for i in range(feature_count)}  # Create bounds for each feature
    }
    return params_bounds

# define evaluation metrics
def evaluate_metrics(y_test_e, y_pred_test_e):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_e, y_pred_test_e)
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Calculate R-squared (R^2)
    # r_squared = r2_score(y_test, y_pred_test)
    r_squared = 1 - np.sum((y_test_e - y_pred_test_e)**2) / np.sum((y_test_e - np.mean(y_test_e))**2)

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