#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   expr_model.py
# Time    :   2025/08/08 21:12:14
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Dec: Using the selected expressions for prediction and further analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def Q_pred_paracali(C_pol, C_eth, C_pg, t):

    """
        SR filtered expression: 
        ((inv((19.832285 - (C_pol)) * (C_pg)) + sqrt((C_pg * C_eth))) + ((inv(inv(C_pg * C_eth))) * -0.039056)) * ((t - sqrt(t) + exp(-t)) * 2.008584)
    """
    # 防止除零错误的小值
    eps = 0 #e.g., 1e-12, but here we respect the original model  
    
    # 公式部分
    term1 = 1.0 / (((19.868389 - C_pol) * C_pg) + eps)   # inv((19.868389 - C_pol) * C_pg)
    term2 = np.sqrt(C_pg * C_eth)                       # sqrt(C_pg * C_eth)
    term3 = 1.0 / (1.0 / (C_pg * C_eth + eps) + eps)    # inv(inv(C_pg * C_eth))
    term3 = term3 * -0.039275162                           # * -0.039056

    formula_terms = term1 + term2 + term3
    
    # 时间部分
    term_time = (t - np.sqrt(t) + np.exp(-t)) * 2.053

    return formula_terms * term_time

# === Step 2: 读取测试数据 ===
file_path = "Symbolic Regression/data/Raw_IVPT_thirty.xlsx"
df_X_test = pd.read_excel(file_path, sheet_name='Formulas-test')
df_Y_test = pd.read_excel(file_path, sheet_name='C-test')

# 去掉首列（如果是RunOrder）
df_X_test = df_X_test.iloc[:, 1:]
df_Y_test = df_Y_test.iloc[:, 1:]

# 假设时间点与Y-test列顺序一致
time_points = df_Y_test.columns.map(lambda x: float(str(x).replace('t', ''))).values


label = "Para_cali"  # for No.9 model test prara cali

# 只对df_Y_test的每一行做预测，保证样本数一致
num_samples = df_Y_test.shape[0]
pred_matrix = np.zeros((num_samples, df_Y_test.shape[1]), dtype=float)
for i in range(num_samples):
    # 假设df_X_test和df_Y_test一一对应，取前num_samples行
    C_pol = df_X_test.iloc[i, 0]
    C_eth = df_X_test.iloc[i, 1]
    C_pg = df_X_test.iloc[i, 2]
    pred_matrix[i, :] = Q_pred_paracali(C_pol, C_eth, C_pg, time_points)  # No.9 model test prara cali

# === Step 4: 计算性能指标（基于所有采样点展开） ===
y_true = df_Y_test.values.flatten()
y_pred = pred_matrix.flatten()

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = np.mean(np.abs(y_true - y_pred))

print(f"R² score on test set: {r2:.4f}")
print(f"RMSE on test set: {rmse:.4f}")
print(f"MAE on test set: {mae:.4f}")

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)
print("df_Y_test shape:", df_Y_test.shape)
print("pred_matrix shape:", pred_matrix.shape)

# === Step 5: 可视化对比（每个配方一条曲线） ===

# plt.figure(figsize=(10, 6))
for i in range(df_Y_test.shape[0]):
    plt.plot(time_points, df_Y_test.iloc[i, :], 'o-', label=f'True F{i+1}', alpha=0.6)
    plt.plot(time_points, pred_matrix[i, :], '--', label=f'Pred F{i+1}', alpha=0.6)

plt.xlabel("Time (h)")
plt.ylabel("Q(t)")
# plt.title("Prediction vs Measurement")
plt.legend(fontsize=12, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Symbolic Regression/srloop/visualisation/Q_pred_test_comparison_filtered_{label}.png", dpi=300, bbox_inches='tight')
plt.show()

# 单独画出每个配方的对比图
for i in range(df_Y_test.shape[0]):
    # plt.figure(figsize=(10, 6))
    plt.plot(time_points, df_Y_test.iloc[i, :], 'o-', label=f'True R{i+1}', alpha=0.6)
    plt.plot(time_points, pred_matrix[i, :], '--', label=f'Pred R{i+1}', alpha=0.6)
    plt.xlabel("Time (h)", fontsize=14)
    plt.ylabel("Q(t)", fontsize=14)
    # plt.title(f"Prediction vs Measured - R{i+1}")
    plt.legend(fontsize=12, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Symbolic Regression/srloop/visualisation/Q_pred_test_comparison_filtered_R{i+1}_{label}.png", dpi=300, bbox_inches='tight')
    plt.show()

# 可视化预测值和真实值的散点图对比图
# plt.figure(figsize=(10, 6))
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal Fit')
plt.scatter(y_true, y_pred, alpha=0.6, c='r', label='R² = {:.2f}, RMSE = {:.2f}'.format(r2, rmse))
plt.xlabel("Measured Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
# plt.title("Scatter Plot of Measured vs Predicted Values")
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"Symbolic Regression/srloop/visualisation/Q_pred_test_scatter_filtered_{label}.png", dpi=300, bbox_inches='tight')
plt.show()

# === Step 6: 保存预测结果 ===
df_pred = pd.DataFrame(pred_matrix, columns=df_Y_test.columns)
df_pred.to_excel(f"Symbolic Regression/srloop/data/Q_pred_test_filtered_{label}.xlsx", index=False)

