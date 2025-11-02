#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   plain_sr_pred.py
# Time    :   2025/08/10 18:06:45
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Desc    :   Symbolic Regression for IVPT data to be a function of concentration and time without feature expansion and selection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from plainSRExprLib import Q_pred_pysr_31, Q_pred_pysr_29

# === Step 1: 定义预测函数 ===
def Q_pred(C_pol, C_eth, C_pg, t):
    """
    预测 Q 值，公式来自 Complexity 18:
    Q = x3 * (14.309577 - sqrt(x1)) - (x2 - ((x3 / ((x2 - 0.023182787 - x1))) / x0))
    变量映射:
        x0 -> C_pol
        x1 -> C_eth
        x2 -> C_pg
        x3 -> t
    """
    term1 = t * (14.309577 - np.sqrt(C_eth))
    term2 = C_pg - ((t / ((C_pg - 0.023182787 - C_eth))) / C_pol)
    return term1 - term2

# def Q_pred(C_pol, C_eth, C_pg, t):
#     """
#     预测 Q 值，公式来自 Complexity 17:
#     Q = (t - 1.2988496) * ( (C_pol * (C_pg - 11.340502)) * (0.0020846934 * (13.586026 - C_eth)) + 10.239821 )
#     """
#     # 防护：确保 t 非负，避免负根号之类的问题（此表达式未使用 sqrt(t)）
#     t = np.array(t, dtype=float)
#     main_prod = (C_pol * (C_pg - 11.340502)) * (0.0020846934 * (13.586026 - C_eth))
#     inner = main_prod + 10.239821
#     return (t - 1.2988496) * inner

def Q_pred_best(C_pol, C_eth, C_pg, t):
    # 由SR模型选择的最佳表达式 model.best
    """
    包含所有变量的表达式：
    Q(t,C) = x3*(11.163061 - ((12.226639 - x1)*x2*0.011946725 + 1/(x0*(14.957137 - x1)))**2) - 12.773257)
    变量映射:
        x0 -> C_pol
        x1 -> C_eth
        x2 -> C_pg
        x3 -> t
    """
    # 计算表达式
    inner = (12.226639 - C_eth) * C_pg * 0.011946725 + 1.0 / (C_pol * (14.957137 - C_eth))
    result = t * (11.163061 - inner ** 2) - 12.773257

    return result

def Q_pred_pysr_selected(C_pol, C_eth, C_pg, t):
    """
    由SR模型选择的loss最小的expression: Q(t,C) = (10.967456 - (x1 - 1*15.065539 + 1/(15.199382 - x1))*(x2*0.045788318 - 0.1735339 - 1/(x1 - 1*8.212636)))*(x3 - 1.2924997))
    """
    # 计算表达式
    inner1 = (C_eth - 1 * 15.065539 + 1 / (15.199382 - C_eth))
    inner2 = (C_pg * 0.045788318 - 0.1735339 - 1 / (C_eth - 1 * 8.212636))
    result = 10.967456 - inner1 * inner2 * (t - 1.2924997)

    return result

def Q_pred_pysr_38(C_pol, C_eth, C_pg, t):
    """
    Q(t,C) = ((x3 + inv(x3)) - 1.7105602) * (((20.446016 - (((x0 * (x1 + (((x0 * -0.33102357) + x2) / (x1 + -15.172693)))) * 0.025330327) + x2)) / ((x2 / (x1 + -16.764357)) + (x2 * -0.0822702))) + 11.088998)
    x0: C_pol
    x1: C_eth
    x2: C_pg
    x3: t
    """
    # 计算表达式
    x0 = C_pol
    x1 = C_eth
    x2 = C_pg
    x3 = t
    # inv(x3) = 1/x3
    inv_x3 = 1.0 / x3
    # inner1 = x0 * (x1 + (((x0 * -0.33102357) + x2) / (x1 + -15.172693)))
    inner1 = x0 * (x1 + (((x0 * -0.33102357) + x2) / (x1 + -15.172693)))
    # term_a = (inner1 * 0.025330327) + x2
    term_a = (inner1 * 0.025330327) + x2
    # numerator = 20.446016 - term_a
    numerator = 20.446016 - term_a
    # denominator = (x2 / (x1 + -16.764357)) + (x2 * -0.0822702)
    denominator = (x2 / (x1 + -16.764357)) + (x2 * -0.0822702)
    # main_term = (numerator / denominator) + 11.088998
    main_term = (numerator / denominator) + 11.088998
    # result = ((x3 + inv_x3) - 1.7105602) * main_term
    result = ((x3 + inv_x3) - 1.7105602) * main_term

    return result



# === Step 2: 读取测试数据 ===
file_path = "Symbolic Regression/data/Raw_IVPT_thirty.xlsx"
df_X_test = pd.read_excel(file_path, sheet_name='Formulas-test')
df_Y_test = pd.read_excel(file_path, sheet_name='C-test')

# 去掉首列（如果是RunOrder）
df_X_test = df_X_test.iloc[:, 1:]
df_Y_test = df_Y_test.iloc[:, 1:]

# 假设时间点与Y-test列顺序一致
time_points = df_Y_test.columns.map(lambda x: float(str(x).replace('t', ''))).values

# === Step 3: 生成预测值矩阵 ===

# 只对df_Y_test的每一行做预测，保证样本数一致
num_samples = df_Y_test.shape[0]
pred_matrix = np.zeros((num_samples, df_Y_test.shape[1]), dtype=float)
for i in range(num_samples):
    # 假设df_X_test和df_Y_test一一对应，取前num_samples行
    C_pol = df_X_test.iloc[i, 0]
    C_eth = df_X_test.iloc[i, 1]
    C_pg = df_X_test.iloc[i, 2]
    # pred_matrix[i, :] = Q_pred_best(C_pol, C_eth, C_pg, time_points)   
    # pred_matrix[i, :] = Q_pred_pysr_selected(C_pol, C_eth, C_pg, time_points)
    # pred_matrix[i, :] = Q_pred_pysr_38(C_pol, C_eth, C_pg, time_points)   # maxisize38
    # pred_matrix[i, :] = Q_pred_pysr_31(C_pol, C_eth, C_pg, time_points)   # maxisize31
    pred_matrix[i, :] = Q_pred_pysr_29(C_pol, C_eth, C_pg, time_points)   # maxisize29

# === Step 4: 计算性能指标（基于所有采样点展开） ===
y_true = df_Y_test.values.flatten()
y_pred = pred_matrix.flatten()

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"R² score on test set: {r2:.4f}")
print(f"RMSE on test set: {rmse:.4f}")

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)
print("df_Y_test shape:", df_Y_test.shape)
print("pred_matrix shape:", pred_matrix.shape)

# === Step 5: 可视化对比（每个配方一条曲线） ===
path_visualisation = "Symbolic Regression/srloop/other models/Plain SR/visualisation/"
path_data = "Symbolic Regression/srloop/other models/Plain SR/data/"
# label = "lowerloss"  
# label = "fullvars"
num = 29
label = f"maxsize{num}"  # 使用 maxsize 38 的模型

# plt.figure(figsize=(10, 6))
for i in range(df_Y_test.shape[0]):
    plt.plot(time_points, df_Y_test.iloc[i, :], 'o-', label=f'True F{i+1}', alpha=0.6)
    plt.plot(time_points, pred_matrix[i, :], '--', label=f'Pred F{i+1}', alpha=0.6)

plt.xlabel("Time (h)")
plt.ylabel("Q(t)")
plt.title("Prediction vs True (Test Data)")
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(path_visualisation + f"Q_pred_test_comparison_plain-{label}.png", dpi=300, bbox_inches='tight')
plt.show()

# 单独画出每个配方的对比图
for i in range(df_Y_test.shape[0]):
    # plt.figure(figsize=(10, 6))
    plt.plot(time_points, df_Y_test.iloc[i, :], 'o-', label=f'True F{i+1}', alpha=0.6)
    plt.plot(time_points, pred_matrix[i, :], '--', label=f'Pred F{i+1}', alpha=0.6)
    plt.xlabel("Time (h)")
    plt.ylabel("Q(t)")
    plt.title(f"Prediction vs True (Test Data) - Formula {i+1}")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_visualisation + f"Q_pred_test_plain_comparison_{label}_F{i+1}.png", dpi=300, bbox_inches='tight')
    plt.show()

# 可视化预测值和真实值的散点图对比图
# plt.figure(figsize=(10, 6))
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal Fit')
plt.scatter(y_true, y_pred, alpha=0.6, c='r', label='R² = {:.2f}, RMSE = {:.2f}'.format(r2, rmse))
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot of True vs Predicted Values (Test Data)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(path_visualisation + f"Q_pred_test_scatter_plain-{label}.png", dpi=300, bbox_inches='tight')
plt.show()

# === Step 6: 保存预测结果 ===
df_pred = pd.DataFrame(pred_matrix, columns=df_Y_test.columns)
df_pred.to_excel(path_data + f"Q_pred_test_plain-{label}.xlsx", index=False)
