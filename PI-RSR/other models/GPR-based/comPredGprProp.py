import pandas as pd

#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   comPredGprProp.py
# Time    :   2025/08/15 16:19:07
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Dec: this script compares the prediction results of plain SR and proposed SR method

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np 

path_gpr = "Symbolic Regression\srloop\other models\GPR-based\data\Q_pred_gpr.xlsx"
path_proposed = "Symbolic Regression\srloop\other models\GPR-based\data\Q_pred_test_filtered.xlsx"
path_raw = "Symbolic Regression\srloop\other models\GPR-based\data\Q_true.xlsx"

# 读取数据（无配方名列，直接读取所有数据）
df_gpr = pd.read_excel(path_gpr)
df_proposed = pd.read_excel(path_proposed)
df_raw = pd.read_excel(path_raw)

# 每个配方画一张图，x轴为时间点，y轴为Q值
num_time_points = df_raw.shape[1]
num_recipes = df_raw.shape[0]
time_points = df_raw.columns.map(lambda x: float(str(x).replace('t', ''))).values
for i in range(num_recipes):
    y_true_R = df_raw.iloc[i].values
    y_gpr_R = df_gpr.iloc[i].values
    y_proposed_R = df_proposed.iloc[i].values
    # plt.figure(figsize=(8, 5))
    plt.plot(time_points, y_true_R, marker='o', color='black', label=f'True R{i+1}')
    plt.plot(time_points, y_gpr_R, marker='o', color='blue', label=f'GPR R{i+1}')
    plt.plot(time_points, y_proposed_R, marker='o', color='red', label=f'Proposed R{i+1}')
    plt.xlabel('Time Point (Index)')
    plt.ylabel('Q value')
    # plt.title(f'Prediction Comparison for R{i+1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Symbolic Regression/srloop/other models/GPR-based/visualisation/comparison_recipe_{i+1}.png", dpi=300, bbox_inches='tight')
    plt.show()

# 所有预测值和真实值的散点图比较
y_true = df_raw.values.flatten()
y_gpr = df_gpr.values.flatten()
y_proposed = df_proposed.values.flatten()

# 计算R²和RMSE
r2 = metrics.r2_score(y_true, y_proposed)
rmse = np.sqrt(metrics.mean_squared_error(y_true, y_proposed))
print(f"R² score: {r2:.4f}, RMSE: {rmse:.4f}")

plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal Fit')
plt.scatter(y_true, y_proposed, alpha=0.6, c='r', label='Proposed R² = {:.2f}, RMSE = {:.2f}'.format(r2, rmse))
plt.scatter(y_true, y_gpr, alpha=0.6, c='b', label='GPR R² = {:.2f}, RMSE = {:.2f}'.format(metrics.r2_score(y_true, y_gpr), np.sqrt(metrics.mean_squared_error(y_true, y_gpr))))
plt.xlabel("Measured Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
# plt.title("Scatter Plot of Measured vs Predicted Values")
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"Symbolic Regression/srloop/other models/GPR-based/visualisation/scatter_comparison_recipe_{i+1}.png", dpi=300, bbox_inches='tight')
plt.show()