import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Utils.probability_of_reaching_better_permeation import probability_of_prediction, probability_of_prediction_m2

def analyze_data(y_pred_f, sigma_f):
    """
    分析数据的均值、标准差、变异系数，并绘制箱线图，进行 ANOVA 和 Levene 检验。
    
    参数:
    y_pred_f (np.array): 预测结果的均值矩阵 (m x n)
    sigma_f (np.array): 预测结果的标准差矩阵 (m x n)
    
    返回:
    dict: 包含均值、标准差、变异系数、ANOVA 和 Levene 检验结果的字典
    """
    results = {}

    # 计算每个实验组的总体均值、标准差和变异系数
    if y_pred_f.ndim == 1:
        y_pred_f = y_pred_f.reshape(1, -1)
        sigma_f = sigma_f.reshape(1, -1)

    means = np.mean(y_pred_f, axis=1)
    stds = np.mean(sigma_f, axis=1)
    cvs = stds / means
    
    # 保存结果
    for i in range(len(means)):
        results[f'Group {i+1}'] = {'mean': means[i], 'std': stds[i], 'cv': cvs[i]}
    
    # 转换为 DataFrame 以绘制箱线图
    data_df = pd.DataFrame(y_pred_f.T, columns=[f'Group {i+1}' for i in range(y_pred_f.shape[0])])
    sns.boxplot(data=data_df)
    plt.title('Box Plot of Groups')
    # plt.show()
    plt.close()
    
    # 进行 ANOVA 检验
    f_val, p_val_anova = stats.f_oneway(*y_pred_f)
    results['ANOVA'] = {'F': f_val, 'p': p_val_anova}
    
    # 进行 Levene 检验
    levene_val, levene_p_val = stats.levene(*y_pred_f)
    results['Levene'] = {'W': levene_val, 'p': levene_p_val}
    
    return results

def decide_method(results):
    """
    根据分析结果判断使用哪个方法来计算概率。
    
    参数:
    results (dict): analyze_data 函数返回的分析结果字典
    
    返回:
    str: 'method1' 或 'method2'
    """
    # 检查均值和标准差
    means = [results[key]['mean'] for key in results if 'Group' in key]
    stds = [results[key]['std'] for key in results if 'Group' in key]
    
    # 检查变异系数
    cvs = [results[key]['cv'] for key in results if 'Group' in key]
    
    # 判断条件
    mean_diff = np.max(means) - np.min(means)
    std_diff = np.max(stds) - np.min(stds)
    cv_diff = np.max(cvs) - np.min(cvs)
    
    p_val_anova = results['ANOVA']['p']
    p_val_levene = results['Levene']['p']
    
    # 根据条件选择方法
    if cv_diff > 0.1 * np.mean(cvs):
        return 'method1'
    if p_val_anova < 0.05 or p_val_levene < 0.05:
        return 'method1'
    return 'method2'

def calculate_probability(method, y_best, y_pred_f, sigma_f):
    """
    根据选择的方法计算达到最佳渗透值的概率。
    
    参数:
    method (str): 'method1' 或 'method2'
    y_best (float): 最佳渗透值
    y_pred_f (np.array): 预测结果的均值矩阵 (m x n)
    sigma_f (np.array): 预测结果的标准差矩阵 (m x n)
    
    返回:
    float: 达到最佳渗透值的概率
    """
    if method == 'method1':
        # 分别计算每组实验的概率，然后取平均值
        # probabilities = []
        # for i in range(y_pred_f.shape[0]):
        #     mu = np.mean(y_pred_f[i])
        #     sigma = np.mean(sigma_f[i])
        #     p = 1 - stats.norm.cdf((y_best - mu) / sigma)
        #     probabilities.append(p)
        prob_m1, _ = probability_of_prediction(y_pred_f, sigma_f, y_best, phi=0.0)
        prob_m1 = np.mean(prob_m1)
        return prob_m1, _
    
    elif method == 'method2':
        # 计算合并数据的均值和方差，然后计算概率
        # mu_F = np.mean(y_pred_f)
        # print('mu_F is ', mu_F)
        # mu_F = np.mean(np.mean(y_pred_f, axis=0))
        # sigma_F = np.sqrt(np.mean(sigma_f**2) / y_pred_f.shape[1])
        # prob = 1 - stats.norm.cdf((y_best - mu_F) / sigma_F)
        prob_m2, _ = probability_of_prediction_m2(y_pred_f, sigma_f, y_best, phi=0.0)
        prob_m2 = np.mean(prob_m2)
        return prob_m2, _