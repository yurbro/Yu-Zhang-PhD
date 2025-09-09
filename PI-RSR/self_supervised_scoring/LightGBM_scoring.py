#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   LightGBM_scoring.py
# Time    :   2025/08/08 17:58:14
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Dec: This script implements LightGBM-based scoring starts with constructing the pseudo labels and then training a LightGBM model as a expression scorer.

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === Step 1: 读取表达式结构特征数据 ===
df = pd.read_csv("Symbolic Regression/srloop/self_supervised_scoring/data/structured_expression_features.csv")

# === Step 2: 构建伪标签 ===
epsilon = 1e-6
df["pseudo_score"] = -np.log(df["loss"] + epsilon)  # loss越小，分数越高

# === Step 3: 构建特征矩阵 ===
feature_cols = [
    "complexity", "depth", "num_ops", "exp", "sqrt", "log", "inv", "pow",
    "+", "-", "*", "/", 
    "has_C_pg", "has_C_eth", "has_C_pol", "has_t"
]
X = df[feature_cols]
y = df["pseudo_score"]

# === Step 4: 划分训练集与测试集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: 训练LGBM模型 ===
model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# === Step 6: 模型评估 ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f} | R2: {r2:.3f}")

# === Step 7: 预测全体数据用于排序 ===
df["predicted_score"] = model.predict(X)
df = df.sort_values(by="predicted_score", ascending=False)
df.to_csv("Symbolic Regression/srloop/self_supervised_scoring/data/scored_expressions.csv", index=False)

# === Step 8: 可视化效果 ===
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Pseudo Score")
plt.ylabel("Predicted Score")
plt.title(f"LGBM Self-Supervised Scoring\nRMSE: {rmse:.2f}, R2: {r2:.2f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("Symbolic Regression/srloop/self_supervised_scoring/visualisation/lgbm_score_plot.png")
plt.show()
