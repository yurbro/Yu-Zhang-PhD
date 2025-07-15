#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   lhs_sample.py
# Time    :   2025/06/02 20:20:47
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk


import numpy as np
from ackley_func import ackley, ackley_max
# from zakharov_func import zakharov_max

# 1. 定义采样区间：三个维度都在 [-32.768, 32.768] 范围内
# lb = np.array([-32.768, -32.768, -32.768])
# ub = np.array([32.768, 32.768, 32.768])

lb = np.array([-5, -5, -5])  # 下界
ub = np.array([10, 10, 10])   # 上界
d  = 3

def lhs_samples(n: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """ 在每个维度范围 [lb[i], ub[i]] 上做 LHS 采样，返回 shape (n, d) """
    rng = np.random.default_rng(seed=42)  # 设置随机种子以确保可重复性
    d = len(lb)
    result = np.zeros((n, d))
    for j in range(d):
        # 生成 n 个分区的随机点
        cut = np.linspace(0, 1, n + 1)
        u = rng.random(n)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(points)
        result[:, j] = points
    return lb + result * (ub - lb)


if __name__ == "__main__":
    n_init = 20         # 初始采样点数量 20
    X_init = lhs_samples(n_init, lb, ub)      # shape (20, 3)
    print(f"Initial LHS samples (shape {X_init.shape}):")
    print(X_init.round(3))

    # Ackley function
    # Y_init = ackley(X_init)                   # shape (20,)
    Y_init = ackley_max(X_init)                # shape (20,)
    for xi, yi in zip(X_init, Y_init):
        print(f"x={xi.round(3)} → Ackley={yi:.4f}")


    # # zakharov func
    # Y_init = zakharov_max(X_init)  # shape (20,)
    # for xi, yi in zip(X_init, Y_init):
    #     print(f"x={xi.round(3)} → Zakharov={yi:.4f}")

    # save to excel file
    import pandas as pd
    # keep three decimal places for better readability
    X_init = X_init.round(3)
    Y_init = Y_init.round(3)
    df = pd.DataFrame(X_init, columns=[f"x{i+1}" for i in range(d)])
    df['Ackley'] = Y_init
    df.to_excel("Multi-Objective Optimisation\Benchmark\Ackley_Function\Dataset\lhs_samples_ackley.xlsx", index=False, sheet_name='INITIAL')
    print("LHS samples and Ackley values saved to 'lhs_samples_ackley.xlsx'.")
