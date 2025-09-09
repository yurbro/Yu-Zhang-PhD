#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   mulit_output_gpr.py
# Time    :   2025/08/12 11:13:02
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Dec: This script is a multi-output Gaussian Process Regression implementation for a baseline model, which compares the performance with the proposed SR model.

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error

# ======== 1. 数据读取 ========
file_path = 'Symbolic Regression/data/Raw_IVPT_thirty.xlsx'

# 训练数据
df_X_train = pd.read_excel(file_path, sheet_name='Formulas-train').iloc[:, 1:]  # 去掉RunOrder
df_Y_train = pd.read_excel(file_path, sheet_name='C-train').iloc[:, 1:]  # 去掉Formulas_C

# 测试数据
df_X_test = pd.read_excel(file_path, sheet_name='Formulas-test').iloc[:, 1:]
df_Y_test = pd.read_excel(file_path, sheet_name='C-test').iloc[:, 1:]

# 时间点（假设所有配方的时间点相同，取训练集的列名作为时间）
time_points = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])

# ======== 2. 构造 GPR 输入输出 ========
def build_dataset(df_X, df_Y, time_points):
    X_list = []
    y_list = []
    for i in range(len(df_X)):  # 每个配方
        C_pol, C_eth, C_pg = df_X.iloc[i].values
        Q_values = df_Y.iloc[i].values
        for t, q in zip(time_points, Q_values):
            X_list.append([C_pol, C_eth, C_pg, t])
            y_list.append(q)
    return np.array(X_list), np.array(y_list)

X_train, y_train = build_dataset(df_X_train, df_Y_train, time_points)
X_test, y_test = build_dataset(df_X_test, df_Y_test, time_points)

print(f"训练集样本数: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ======== 3. 定义 GPR 模型 ========
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1, 1, 1, 10], length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0, normalize_y=True, random_state=42)

# ======== 4. 训练模型 ========
print("开始训练 GPR 模型...")
gpr.fit(X_train, y_train)
print("训练完成!")
print("优化后的 kernel:", gpr.kernel_)

# ======== 5. 测试集预测 ========
y_pred, y_std = gpr.predict(X_test, return_std=True)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = 1 - mse / np.var(y_test)
print(f"测试集 R^2: {r2:.4f}")
print(f"测试集 MSE: {mse:.4f}")
print(f"测试集 RMSE: {rmse:.4f}")

# ======== 6. 可选：保存预测结果 ========
df_results = pd.DataFrame({
    'C_pol': X_test[:, 0],
    'C_eth': X_test[:, 1],
    'C_pg': X_test[:, 2],
    't': X_test[:, 3],
    'Q_true': y_test,
    'Q_pred': y_pred,
    'pred_std': y_std
})
df_results.to_excel('Symbolic Regression/srloop/other models/GPR-based/data/GPR_baseline_predictions.xlsx', index=False)
print("预测结果已保存到 GPR_baseline_predictions.xlsx")


# 可视化预测结果（按配方分组）
import matplotlib.pyplot as plt

# 假设每个配方有 len(time_points) 个时间点
num_formulas = int(len(df_results) / len(time_points))

# plt.figure(figsize=(10, 6))
for i in range(num_formulas):
    start = i * len(time_points)
    end = (i + 1) * len(time_points)
    t = df_results['t'].iloc[start:end]
    q_true = df_results['Q_true'].iloc[start:end]
    q_pred = df_results['Q_pred'].iloc[start:end]
    plt.plot(t, q_true, 'o-', label=f'True F{i+1}', alpha=0.6)
    plt.plot(t, q_pred, '--', label=f'Pred F{i+1}', alpha=0.6)

plt.xlabel("Time (h)")
plt.ylabel("Q(t)")
plt.title("Prediction vs True (Test Data)")
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('Symbolic Regression/srloop/other models/GPR-based/visualisation/GPR_baseline_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 单独画出每个配方的预测结果
for i in range(num_formulas):
    # plt.figure(figsize=(10, 6))
    start = i * len(time_points)
    end = (i + 1) * len(time_points)
    t = df_results['t'].iloc[start:end]
    q_true = df_results['Q_true'].iloc[start:end]
    q_pred = df_results['Q_pred'].iloc[start:end]
    plt.plot(t, q_true, 'o-', label=f'True F{i+1}', alpha=0.6)
    plt.plot(t, q_pred, '--', label=f'Pred F{i+1}', alpha=0.6)
    plt.xlabel("Time (h)")
    plt.ylabel("Q(t)")
    plt.title(f"Prediction vs True (Test Data) - Formula {i+1}")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Symbolic Regression/srloop/other models/GPR-based/visualisation/GPR_baseline_predictions_comparison_F{i+1}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 可视化真实值和预测值的散点图
# plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values (Test Data)")
plt.grid(True)
plt.tight_layout()
plt.legend(["y = x", f"R² = {r2:.2f}"])
plt.savefig('Symbolic Regression/srloop/other models/GPR-based/visualisation/GPR_baseline_predictions_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
