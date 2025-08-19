
from scipy.stats import norm
import numpy as np
from icecream import ic
from scipy import stats


"""
    For this function, we're going to calculate the probability of reaching a better permeation value using the predicted mean and standard deviation values
    by using the GPR model. The function will return the probability of reaching a better permeation value.
"""

# def probability_of_prediction(mu, sigma, y_test):
#     """
#     计算达到或超过最大渗透效果的概率。
    
#     参数:
#         mu (float): 预测的均值。
#         sigma (float): 预测的标准差。
#         current_best (float): 目前最好的渗透效果。
    
#     返回:
#         float: 超过最大渗透效果的概率。
#     """

#     current_best = calculate_current_best(y_test)
#     print("Now the current best permeation value is: ", current_best)

#     # if sigma > 0:
#     z_score = (current_best - np.mean(mu, axis=1)) / np.mean(sigma, axis=1)
#     probability = 1 - norm.cdf(z_score)
#     # else:
#     #     # 如果方差为0，判断均值是否已经超过最佳值
#     #     probability = 1.0 if mu > current_best else 0.0

#     return probability

def probability_of_prediction(mu, sigma, y_best, phi):
    """
    计算达到或超过最大渗透效果的概率。

    参数:
        mu (np.array): 预测的均值数组。
        sigma (np.array): 预测的标准差数组。
        y_test (np.array): 测试集的真实渗透效果。

    返回:
        np.array: 每个预测点超过最大渗透效果的概率。
    """
    current_best = calculate_current_best(y_best, phi=0.0)
    # print("Now the current best permeation value is: ", current_best)

    mu_mean = np.mean(mu, axis=1)
    sigma_mean = np.mean(sigma, axis=1)
    print('the method-1 mu_mean is: ', mu_mean)
    print('the method-1 sigma_mean is: ', sigma_mean)

    # 初始化概率数组，处理方差为零的情况
    probability = np.zeros_like(mu_mean)

    # 计算方差不为零的情况
    nonzero_sigma_mask = sigma_mean > 0
    z_score = np.zeros_like(mu_mean)
    z_score[nonzero_sigma_mask] = (current_best - mu_mean[nonzero_sigma_mask]) / sigma_mean[nonzero_sigma_mask]
    ic(z_score[nonzero_sigma_mask])
    probability[nonzero_sigma_mask] = 1 - norm.cdf(z_score[nonzero_sigma_mask])

    # 方差为零的情况
    probability[~nonzero_sigma_mask] = (mu_mean[~nonzero_sigma_mask] > current_best).astype(float)

    probability = np.round(probability, 4)      # 保留四位小数
    # probability = np.round(probability * 100, 2)     # 转换为百分比

    return probability, current_best

# method-2 to calculate the probability of reaching a better permeation value and total probability
def probability_of_prediction_m2(mu, sigma, y_best, phi):

    """
        This method is not use all repeated experiments value to calculate the probability of reaching a better permeation value.
        This function will calculate the total probability of reaching a better permeation value by using a mean value of the repeated experiments in each sampling point.
    """
    current_best = calculate_current_best(y_best, phi=0.0)
    # print("Now the current best permeation value is: ", current_best)

    # this is different from the previous method, this prediction value denotes the whole formulation group's prediction value
    mu_mean = np.mean(np.mean(mu, axis=0))        
    sigma_mean = np.mean(np.mean(sigma, axis=0))
    print('the method-2 mu_mean is: ', mu_mean)
    print('the method-2 sigma_mean is: ', sigma_mean)
    ic(mu_mean, sigma_mean)
    # 初始化概率数组，处理方差为零的情况
    probability = np.zeros_like(mu_mean)

    # 计算方差不为零的情况
    nonzero_sigma_mask = sigma_mean > 0
    z_score = np.zeros_like(mu_mean)
    z_score[nonzero_sigma_mask] = (current_best - mu_mean[nonzero_sigma_mask]) / sigma_mean[nonzero_sigma_mask]
    ic(z_score[nonzero_sigma_mask])
    probability[nonzero_sigma_mask] = 1 - norm.cdf(z_score[nonzero_sigma_mask])

    # 方差为零的情况
    probability[~nonzero_sigma_mask] = (mu_mean[~nonzero_sigma_mask] > current_best).astype(float)

    probability = np.round(probability, 4)      # 保留四位小数
    # probability = np.round(probability * 100, 2)     # 转换为百分比


    return probability, current_best

# Calculate the current_best permeation value
def calculate_current_best(y_best, phi):
    """
    计算当前最好的渗透效果。
    
    参数:
        y_test (np.array): 测试集的真实渗透效果。
    
    返回:
        float: 最好的渗透效果。
    """
    # ic(y_best, phi, y_best.shape, len(y_best.shape))
    if len(y_best.shape) == 1:
        current_best = np.max(y_best)
    else:  
        current_best = np.max(np.mean(y_best, axis=1))

    current_best = current_best * (1 - phi)       

    return current_best

# # 假设 mu, sigma, current_best 已知
# prob = probability_of_improvement(mu, sigma, current_best)
# print("达到或超过最大渗透效果的概率是：", prob)


def method1_probability(mu_matrix, sigma_matrix, y_best, phi=0.0):
    probabilities = []
    n, m = mu_matrix.shape

    ic(mu_matrix, sigma_matrix, sigma_matrix.shape)

    y_best = calculate_current_best(y_best, phi)

    ic(y_best)
    if n == 1:
        mu_mean = mu_matrix
        sigma_mean = sigma_matrix
    else:
        mu_mean = np.mean(mu_matrix, axis=1)
        sigma_mean = np.mean(sigma_matrix, axis=1)
    z = (y_best - mu_mean) / sigma_mean
    p = 1 - stats.norm.cdf(z)
    # p = norm.cdf(z)
    # probabilities.append(p)
    # P_F1 = np.mean(probabilities)
    P_F1 = np.round(p, 4)
    return P_F1, y_best, mu_mean, sigma_mean

def method2_probability(mu_matrix, sigma_matrix, y_best, phi=0.0):
    mu_F = np.mean(mu_matrix)
    y_best = calculate_current_best(y_best, phi)
    sigma_F = np.sqrt(np.sum(sigma_matrix ** 2) / (mu_matrix.size))
    z = (y_best - mu_F) / sigma_F
    P_F2 = 1 - stats.norm.cdf(z)
    # P_F2 = norm.cdf(z)
    return P_F2, y_best, mu_F, sigma_F

