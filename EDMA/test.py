# import numpy as np
# import matplotlib.pyplot as plt

# from bayes_opt import BayesianOptimization
# from bayes_opt import UtilityFunction
# from icecream.icecream import ic

# from scipy.stats import norm

# mu = 146.05
# y_best = 107.94
# sigma = 32.0832

# z_score = (y_best - mu) / sigma
# print(z_score)

# probability = 1 - norm.cdf(z_score)
# print(probability)

# from scipy.stats import norm
# import numpy as np
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2)

# x = np.linspace(norm.ppf(0.0001),
#                 norm.ppf(0.9999), 100)
# print(x)
# rv = norm()
# ax[0].bar(x, rv.pdf(x), width=0.01, label='pdf')
# ax[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
# ax[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
# ax[0].set_title("pdf")
# ax[1].plot(x, rv.cdf(x), '-^', linewidth=1, label='cdf', markersize=2)
# ax[1].set_xticks([-3, -2, -1, 0, 1, 2, 3])
# ax[1].set_yticks(np.linspace(0.0, 1.0, 5))
# ax[1].set_title("cdf")
# plt.show()

# # 时间点
# time_points = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
# # 预测概率（示例数据）
# probabilities = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96])
# # 阈值
# threshold = 0.5

# plt.figure(figsize=(8, 6))
# plt.plot(time_points, probabilities, marker='o', color='y', label='Prediction Probability')
# plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 0.5')
# plt.xlabel('Time Points')
# plt.ylabel('Probability')
# plt.title('Scenario 1: Continue Experiment')
# plt.legend()
# plt.grid()
# plt.show()

# # 预测概率（示例数据）
# probabilities = np.array([0.6, 0.65, 0.55, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15])

# plt.figure(figsize=(8, 6))
# plt.plot(time_points, probabilities, marker='o', color='y',label='Prediction Probability')
# plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 0.5')
# plt.axvline(x=4, color='g', linestyle='--', label='Stop Point')
# plt.xlabel('Time Points')
# plt.ylabel('Probability')
# plt.title('Scenario 2: Stop Experiment at Time Point 4')
# plt.legend()
# plt.grid()
# plt.show()

# # 预测概率（示例数据）
# probabilities = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])

# plt.figure(figsize=(8, 6))
# plt.plot(time_points, probabilities, marker='o', color='y',label='Prediction Probability')
# plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 0.5')
# plt.axvline(x=2, color='g', linestyle='--', label='Review Point')
# plt.axvline(x=1, color='orange', linestyle='--', label='Initial Stop Point')
# plt.xlabel('Time Points')
# plt.ylabel('Probability')
# plt.title('Scenario 3: Initial Stop, Later Exceeds Threshold')
# plt.grid()
# plt.legend()
# plt.show()

# # 预测概率（示例数据）
# probabilities = np.array([0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03])

# plt.figure(figsize=(8, 6))
# plt.plot(time_points, probabilities, marker='o', color='y',label='Prediction Probability')
# plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 0.5')
# plt.axvline(x=1, color='g', linestyle='--', label='Stop Point')
# plt.xlabel('Time Points')
# plt.ylabel('Probability')
# plt.title('Scenario 4: Initial Stop, Consistently Below Threshold')
# plt.legend()
# plt.grid()
# plt.show()


import numpy as np
import scipy.stats as stats

# 方法1: 分别计算各组实验达到 y_best 的概率
def method1_probability(mu_matrix, sigma_matrix, y_best):
    probabilities = []
    n, m = mu_matrix.shape
    for i in range(n):
        group_probabilities = []
        for j in range(m):
            z = (y_best - mu_matrix[i, j]) / sigma_matrix[i, j]
            p = 1 - stats.norm.cdf(z)
            group_probabilities.append(p)
        probabilities.append(np.mean(group_probabilities))
    P_F1 = np.mean(probabilities)
    return P_F1

# 方法2: 合并数据，计算整体达到 y_best 的概率
def method2_probability(mu_matrix, sigma_matrix, y_best):
    mu_F = np.mean(mu_matrix)
    sigma_F = np.sqrt(np.sum(sigma_matrix ** 2) / (mu_matrix.size))
    z = (y_best - mu_F) / sigma_F
    P_F2 = 1 - stats.norm.cdf(z)
    return P_F2

# 方法1: 分别计算各组实验的EI值
def method1_EI(mu_matrix, sigma_matrix, y_best):
    EI_values = []
    n, m = mu_matrix.shape
    for i in range(n):
        group_EI = []
        for j in range(m):
            mu = mu_matrix[i, j]
            sigma = sigma_matrix[i, j]
            z = (mu - y_best) / sigma
            EI = (mu - y_best) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            group_EI.append(EI)
        EI_values.append(np.mean(group_EI))
    overall_EI = np.mean(EI_values)
    return overall_EI

# 方法2: 合并数据，计算整体的EI值
def method2_EI(mu_matrix, sigma_matrix, y_best):
    mu_F = np.mean(mu_matrix)
    sigma_F = np.sqrt(np.sum(sigma_matrix ** 2) / (mu_matrix.size))
    z = (mu_F - y_best) / sigma_F
    EI = (mu_F - y_best) * stats.norm.cdf(z) + sigma_F * stats.norm.pdf(z)
    return EI

# 示例数据
mu = np.array([[10, 12], [15, 17], [20, 22]])
sigma = np.array([[1, 1.5], [2, 2.5], [3, 3.5]])
y_best = 18

# 计算概率
P_F1 = method1_probability(mu, sigma, y_best)
P_F2 = method2_probability(mu, sigma, y_best)

# 计算EI值
EI_F1 = method1_EI(mu, sigma, y_best)
EI_F2 = method2_EI(mu, sigma, y_best)

print(f"Method 1 Probability: {P_F1}")
print(f"Method 2 Probability: {P_F2}")
print(f"Method 1 Expected Improvement: {EI_F1}")
print(f"Method 2 Expected Improvement: {EI_F2}")
