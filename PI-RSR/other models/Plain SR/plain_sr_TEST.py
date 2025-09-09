#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   sr_Qt2fct.py
# Time    :   2025/07/31 20:32:21
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Desc    :   Symbolic Regression for IVPT data to be a function of concentration and time

"""
     This model is to explore the potential of symbolic regression to derive a physical model from the IVPT data when some
     people are not familiar with the physics of the problem.
     'Data-driven discovery of governing equations from data' is a common task in scientific computing.
     Concentration of excipients + time t (input) and cumulative amount of ibuprofen (output) are the two main variables in this problem to create the governing equation.
"""

import pandas as pd
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from icecream import ic
import pandas as pd
import seaborn as sns
import sympy
from coverage_penalty import custom_loss

# === Step 1: 读取数据 ===
file_path = 'Symbolic Regression/data/Raw_IVPT_thirty.xlsx'
df_X = pd.read_excel(file_path, sheet_name='Formulas-train')
df_Y = pd.read_excel(file_path, sheet_name='C-train')  # 每列是一个配方在10个时间点的Q(t)
df_X = df_X.iloc[:, 1:]  # 去掉RunOrder列
df_Y = df_Y.iloc[:, 1:]  # 去掉Formulas_C列

# === Step 2: 数据展开（构建训练集） ===
time_points = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
X_all, y_all = [], []

for i, (_, row) in enumerate(df_X.iterrows()):
    c_pol, c_eth, c_pg = row[['Poloxamer 407', 'Ethanol', 'Propylene glycol']]
    Q_series = df_Y.iloc[i, :].values

    for t, q in zip(time_points, Q_series):
        X_all.append([c_pol, c_eth, c_pg, t])     # 每个样本包含三个配方成分和时间点
        y_all.append(q)                           # 每个样本的目标值是Q(t)

X_all = np.array(X_all)
y_all = np.array(y_all)
# ic(X_all, y_all)

# === Step 3: 拟合符号回归模型 ===
model = PySRRegressor(
    model_selection="best",
    niterations=1000,  # same as the proposed method's iterations=8*1000
    parsimony=0.005, # 控制模型复杂度
    adaptive_parsimony_scaling= 500, # 自适应简约性缩放
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "log", "exp", "square", "inv"],
    loss=custom_loss,
    maxsize=38,   # 这里设定为Proposed method中找到的最大复杂度值 ·38
    verbosity=1,
)

model.fit(X_all, y_all)

# === Step 4: 模型评估 ===
y_pred = model.predict(X_all)
print("R2 score:", r2_score(y_all, y_pred))

# === Step 5: 输出公式 ===
print("Best symbolic model:")
print(model)
fame_csv = f'Symbolic Regression/srloop/other models/Plain SR/data/hall_of_fame_plainSR_test.csv'
hall_of_fame = model.equations_
hall_of_fame.to_csv(fame_csv, index=False)

# === 可选可视化 ===
plt.scatter(y_all, y_pred, alpha=0.5)
plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'r--', lw=2)
plt.xlabel("True Q(t)")
plt.ylabel("Predicted Q(t)")
plt.title("Symbolic Regression Prediction vs. True")        
plt.legend(["y = x", "Predictions R2: {:.2f}".format(r2_score(y_all, y_pred))])
# plt.savefig('Symbolic Regression/visualisation/sr_Qt2fct_prediction_raw_train.png')
plt.show()
