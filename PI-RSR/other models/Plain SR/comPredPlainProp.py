import pandas as pd

#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   comPredPlainProp.py
# Time    :   2025/08/15 16:19:07
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Dec: this script compares the prediction results of plain SR and proposed SR method

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np 

path_plain38 = "Symbolic Regression\srloop\other models\Plain SR\data\Q_pred_test_plain-maxsize38.xlsx"
path_plain29 = "Symbolic Regression\srloop\other models\Plain SR\data\Q_pred_test_plain-maxsize29.xlsx"
path_plain31 = "Symbolic Regression\srloop\other models\Plain SR\data\Q_pred_test_plain-maxsize31.xlsx"
path_proposed = "Symbolic Regression\srloop\other models\Plain SR\data\Q_pred_test_filtered.xlsx"
path_pirsr_para_cali = "Symbolic Regression\srloop\other models\Plain SR\data\Q_pred_test_filtered_Para_cali.xlsx"
path_raw = "Symbolic Regression\srloop\other models\Plain SR\data\Q_true.xlsx"
path_gpr = "Symbolic Regression\srloop\other models\GPR-based\data\Q_pred_gpr.xlsx"

# 读取数据（无配方名列，直接读取所有数据）
df_plain38 = pd.read_excel(path_plain38)
df_plain29 = pd.read_excel(path_plain29)
df_plain31 = pd.read_excel(path_plain31)
df_proposed = pd.read_excel(path_proposed)
df_pirsr_para_cali = pd.read_excel(path_pirsr_para_cali)
df_raw = pd.read_excel(path_raw)
df_gpr = pd.read_excel(path_gpr)

# 每个配方画一张图，x轴为时间点，y轴为Q值
num_time_points = df_raw.shape[1]
num_recipes = df_raw.shape[0]
time_points = df_raw.columns.map(lambda x: float(str(x).replace('t', ''))).values
for i in range(num_recipes):
    y_true_R = df_raw.iloc[i].values
    y_plain38_R = df_plain38.iloc[i].values
    y_plain29_R = df_plain29.iloc[i].values
    y_plain31_R = df_plain31.iloc[i].values
    y_proposed_R = df_proposed.iloc[i].values
    y_gpr_R = df_gpr.iloc[i].values
    y_pirsr_para_cali_R = df_pirsr_para_cali.iloc[i].values
    # plt.figure(figsize=(8, 5))
    plt.plot(time_points, y_true_R, marker='o', color='black', label=f'True')
    plt.plot(time_points, y_plain29_R, alpha=0.6, marker='o', color='green', label=f'PySR-1')
    plt.plot(time_points, y_plain31_R, alpha=0.6, marker='o', color='purple', label=f'PySR-2')
    plt.plot(time_points, y_plain38_R, alpha=0.6, marker='o', color='blue', label=f'PySR-3')
    plt.plot(time_points, y_gpr_R, alpha=0.6, marker='o', color='orange', label=f'GPR')
    plt.plot(time_points, y_proposed_R, alpha=0.6, marker='o', color='red', label=f'PI-RSR')
    plt.plot(time_points, y_pirsr_para_cali_R, alpha=0.6, marker='o', color='brown', label=f'PI-RSR(Cali)')
    plt.xlabel('Time Point (h)')
    # plt.xlim(0, 30)
    plt.xticks(time_points)
    plt.ylabel(f'Q value (R{i+1})')
    # plt.title(f'Prediction Comparison for R{i+1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Symbolic Regression/srloop/other models/Plain SR/visualisation/comparison_formulation_{i+1}_four_gpr_paracali.png", dpi=300, bbox_inches='tight')
    plt.show()

# 所有预测值和真实值的散点图比较
y_true = df_raw.values.flatten()
y_plain38 = df_plain38.values.flatten()
y_plain29 = df_plain29.values.flatten()
y_plain31 = df_plain31.values.flatten()
y_proposed = df_proposed.values.flatten()
y_gpr = df_gpr.values.flatten()
y_pirsr_para_cali = df_pirsr_para_cali.values.flatten()

# 计算R²和RMSE
r2 = metrics.r2_score(y_true, y_proposed)
rmse = np.sqrt(metrics.mean_squared_error(y_true, y_proposed))
print(f"R² score: {r2:.4f}, RMSE: {rmse:.4f}")

plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal Fit')
plt.scatter(y_true, y_plain29, alpha=0.6, c='g', label='PySR-1 R² = {:.3f}, RMSE = {:.3f}'.format(metrics.r2_score(y_true, y_plain29), np.sqrt(metrics.mean_squared_error(y_true, y_plain29)), metrics.mean_absolute_error(y_true, y_plain29)))
plt.scatter(y_true, y_plain31, alpha=0.6, c='purple', label='PySR-2 R² = {:.3f}, RMSE = {:.3f}'.format(metrics.r2_score(y_true, y_plain31), np.sqrt(metrics.mean_squared_error(y_true, y_plain31)), metrics.mean_absolute_error(y_true, y_plain31)))
plt.scatter(y_true, y_plain38, alpha=0.6, c='b', label='PySR-3 R² = {:.3f}, RMSE = {:.3f}'.format(metrics.r2_score(y_true, y_plain38), np.sqrt(metrics.mean_squared_error(y_true, y_plain38)), metrics.mean_absolute_error(y_true, y_plain38)))
plt.scatter(y_true, y_gpr, alpha=0.6, c='orange', label='GPR R² = {:.3f}, RMSE = {:.3f}'.format(metrics.r2_score(y_true, y_gpr), np.sqrt(metrics.mean_squared_error(y_true, y_gpr)), metrics.mean_absolute_error(y_true, y_gpr)))
plt.scatter(y_true, y_proposed, alpha=0.6, c='r', label='PI-RSR R² = {:.3f}, RMSE = {:.3f}'.format(r2, rmse, metrics.mean_absolute_error(y_true, y_proposed)))
plt.scatter(y_true, y_pirsr_para_cali, alpha=0.6, c='brown', label='PI-RSR(Cali) R² = {:.3f}, RMSE = {:.3f}'.format(metrics.r2_score(y_true, y_pirsr_para_cali), np.sqrt(metrics.mean_squared_error(y_true, y_pirsr_para_cali)), metrics.mean_absolute_error(y_true, y_pirsr_para_cali)))
plt.xlabel("Measured Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
# plt.title("Scatter Plot of Measured vs Predicted Values")
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"Symbolic Regression/srloop/other models/Plain SR/visualisation/scatter_comparison_formulation_{i+1}_maxsize38_gpr_paracali.png", dpi=300, bbox_inches='tight')
plt.show()

# residual plot
# plt.figure(figsize=(8, 5))
plt.scatter(y_true, y_true - y_plain29, alpha=0.6, c='g', label='PySR-1')
plt.scatter(y_true, y_true - y_plain31, alpha=0.6, c='purple', label='PySR-2')
plt.scatter(y_true, y_true - y_plain38, alpha=0.6, c='b', label='PySR-3')
plt.scatter(y_true, y_true - y_gpr, alpha=0.6, c='orange', label='GPR')
plt.scatter(y_true, y_true - y_proposed, alpha=0.6, c='r', label='PI-RSR')
plt.scatter(y_true, y_true - y_pirsr_para_cali, alpha=0.6, c='brown', label='PI-RSR(Cali)')
plt.axhline(0, color='black', lw=2, ls='--')
plt.xlabel("Measured Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
# plt.title("Residual Plot")
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"Symbolic Regression/srloop/other models/Plain SR/visualisation/residual_comparison_formulation_{i+1}_maxsize38_gpr_paracali.png", dpi=300, bbox_inches='tight')
plt.show()