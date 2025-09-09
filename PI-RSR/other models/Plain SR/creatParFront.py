#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   creatParFront.py
# Time    :   2025/08/15 12:08:53
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd

# 读取数据

df = pd.read_csv("Symbolic Regression/srloop/other models/Plain SR/data/hall_of_fame_plainSR_maxsize38.csv")
data = {
    "expression_id": df["sympy_format"].tolist(),
    "complexity": df["complexity"].tolist(),
    "loss": df["loss"].tolist()
}

df = pd.DataFrame(data)

# 1. 相同 complexity 只保留 loss 最低的
df_best = df.loc[df.groupby("complexity")["loss"].idxmin()].reset_index(drop=True)

# 2. 按 complexity 升序排序
df_best = df_best.sort_values(by="complexity").reset_index(drop=True)

# 3. 计算 Pareto front（minimize complexity, minimize loss）
pareto_front = []
current_best_loss = float('inf')
for _, row in df_best.iterrows():
    if row["loss"] < current_best_loss:
        pareto_front.append(row)
        current_best_loss = row["loss"]

df_pareto = pd.DataFrame(pareto_front).reset_index(drop=True)

# 把表达式中的变量换成原始变量

def replace_variables(expr, original_vars):
    for var, original in original_vars.items():
        expr = expr.replace(var, original)
    return expr

# 应用到 Pareto front
original_vars = {
    "x0": "c_pol",
    "x1": "c_eth",
    "x2": "c_pg",
    "x3": "t"
}

df_pareto["expression_id"] = df_pareto["expression_id"].apply(lambda x: replace_variables(x, original_vars))

# 保存结果
df_pareto.to_csv("Symbolic Regression/srloop/other models/Plain SR/data/pareto_front_run-plainSR_maxsize38.csv", index=False)

print("精简后数据（相同 complexity 取最优）：")
print(df_best)

print("\nPareto front 数据集：")
print(df_pareto)


