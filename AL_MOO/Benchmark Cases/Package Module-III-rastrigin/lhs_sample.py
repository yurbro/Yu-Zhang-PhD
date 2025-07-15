#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   lhs_sample.py
# Time    :   2025/06/02 20:20:47
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk


import numpy as np
from ackley_func import ackley, ackley_max

lb = np.array([-5, -5, -5])  # lower bound
ub = np.array([10, 10, 10])   # upper bound
d  = 3

def lhs_samples(n: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """  Generate Latin Hypercube Sampling (LHS) points within the given bounds."""
    rng = np.random.default_rng(seed=42)  # Set random seed for reproducibility
    d = len(lb)
    result = np.zeros((n, d))
    for j in range(d):
        # Generate random points in n partitions
        cut = np.linspace(0, 1, n + 1)
        u = rng.random(n)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(points)
        result[:, j] = points
    return lb + result * (ub - lb)


if __name__ == "__main__":
    n_init = 20         # Initial number of sampling points
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
