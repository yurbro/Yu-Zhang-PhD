#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   creatParFront.py
# Time    :   2025/08/14 19:29:16
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd

# 读取数据

df = pd.read_csv("Symbolic Regression/srloop/data/all_expr_recomputed_complexity.csv")
data = {
    "expression_id": df["restored_equation"].tolist(),
    "complexity": df["complexity_recomputed"].tolist(),
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
# 保存结果
df_pareto.to_csv("Symbolic Regression/srloop/data/restored_pareto_front_run-8.csv", index=False)

print("精简后数据（相同 complexity 取最优）：")
print(df_best)

print("\nPareto front 数据集：")
print(df_pareto)

