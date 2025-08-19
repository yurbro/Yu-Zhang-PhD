

import numpy as np
from icecream import ic
from scipy.stats import norm

def decision_objective(title_colums, Initial_position, y_test_scaled):
    """
        This function will quantify the difference between the predictions and the actual values of the target variable in the test set.
    """
    # Set a parameter space to record the posterior values of each sampling points
    posterior_regions_each_moment = {}

    # Get the moment of the test set
    moments = title_colums[Initial_position:]

    # find the maximum value and minimum value of the target variable of each moment in the test set that will be added to the posterior_regions_each_moment
    # iterate through the target variables
    for i in range(y_test_scaled.shape[1]):
        max_val = np.max(y_test_scaled[:, i])
        min_val = np.min(y_test_scaled[:, i])
        posterior_regions_each_moment[f'{moments[i]}'] = {'max': max_val, 'min': min_val}

    # ic(posterior_regions_each_moment)

    return posterior_regions_each_moment

def decision_expected_improvement(best_gpr, scaler_X, scaler_y, x_new, current_best, threshold=0.01, xi=0.1):
    """
    根据预期改进来决定是否继续实验。
    
    参数:
        x_new (np.array): 新配方的数据点。
        current_best (float): 当前观测到的最佳结果。
        threshold (float): 决定是否继续实验的阈值。
    
    返回:
        bool: 如果应该继续实验则返回 True, 否则返回 False。
    """
    x_new_scaled = scaler_X.transform(x_new)  # 标准化新数据点
    mu, sigma = best_gpr.predict(x_new_scaled, return_std=True)
    mu = scaler_y.inverse_transform(mu)  # 将预测结果转换回原始比例
    sigma = sigma[0] 
    ic(mu, sigma)

    # 计算预期改进
    # improvement = mu - current_best - xi
    # z = improvement / sigma 
    # ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    improvement = mu - current_best - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    return ei, ei > threshold
