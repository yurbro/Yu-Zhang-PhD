#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   adptive_af_weight.py
# Time    :   2025/06/04 18:26:29
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import numpy as np

def update_normalised_weights_and_allocate(
    prev_weights: dict,
    accuracies: dict,
    k: int,
    alpha: float = 0.0,
    rounding: str = 'floor',
    epsilon: float = 1e-9
    ):
    """
    使用归一化精度更新 EI 和 HV 的权重，并分配实验点数。

    Parameters:
    -----------
    prev_weights : dict
        {"ei": float, "hv": float}，上一轮的归一化权重
    accuracies : dict
        {"ei": float, "hv": float}，当前轮的准确率
    k : int
        当前轮的实验点总数
    alpha : float
        平滑系数（0 = 不平滑，1 = 完全保留旧权重）
    rounding : str
        点数分配时的舍入方法：'floor' or 'round'
    epsilon : float
        避免除以 0 的极小值（用于归一化）

    Returns:
    --------
    new_weights : dict
        {"ei": float, "hv": float}，归一化后的新权重
    allocation : dict
        {"ei": int, "hv": int}，实验点分配数量
    """

    a_ei = accuracies.get("ei", 0.0)
    a_hv = accuracies.get("hv", 0.0)
    total_acc = a_ei + a_hv

    # ===== Step 1: 特殊情况处理（全部为 0） =====
    if total_acc == 0:
        norm_a_ei = 0.5
        norm_a_hv = 0.5
    else:
        norm_a_ei = a_ei / (total_acc + epsilon)
        norm_a_hv = a_hv / (total_acc + epsilon)

    # ===== Step 2: 使用指数平滑更新权重 =====
    w_old_ei = prev_weights.get("ei", 0.5)
    w_new_ei = alpha * w_old_ei + (1 - alpha) * norm_a_ei
    w_new_hv = 1.0 - w_new_ei  # 保证归一化

    # ===== Step 3: 分配实验点 =====
    if rounding == 'floor':
        n_ei = max(1, int(w_new_ei * k))
    elif rounding == 'round':
        n_ei = max(1, round(w_new_ei * k))
    else:
        raise ValueError("Invalid rounding method: choose 'floor' or 'round'.")

    n_hv = k - n_ei

    # ===== Output =====
    new_weights = {"ei": w_new_ei, "hv": w_new_hv}
    allocation = {"ei": n_ei, "hv": n_hv}

    return new_weights, allocation

def calculate_accuracy(targets, y_best):
    """
    计算Pareto点对应的目标值超过最大值y_best的概率。
    Parameters:
    -----------
    targets : list or np.array
        实际目标值
    y_best : float
        当前最优目标值

    Returns:
    --------
    accuracy : float
        准确率，范围在 [0, 1]
    """
    targets = np.array(targets)
    y_best = np.array(y_best)

    # 计算超过 y_best 的目标值数量
    count_exceeding = np.sum(targets >= y_best)

    # 计算准确率
    accuracy = count_exceeding / len(targets) if len(targets) > 0 else 0.0

    return accuracy

if __name__ == "__main__":

    # 初始条件
    """
        只需把 new_weights 存为 prev_weights, 即可迭代使用每一轮。
    """
    prev_weights = {"ei": 0.5, "hv": 0.5}       # Insert your previous weights here
    accuracies = {"ei": 0.0, "hv": 0.0}     # Insert your accuracies here
    k = 6
    alpha = 0.5                                 # 平滑系数，0 = 不平滑，1 = 保留旧权重, < 1 = 新旧权重混合

    # 更新权重 + 分配点数
    new_weights, allocation = update_normalised_weights_and_allocate(
        prev_weights, accuracies, k, alpha, rounding='floor', epsilon=1e-9
    )

    print("New Weights:", new_weights)
    print("Point Allocation:", allocation)

    # 示例：计算准确率
    targets = [0.8, 0.9, 0.85, 0.95, 0.7]  # 示例目标值
    y_best = 0.9  # 当前最优目标值
    accuracy = calculate_accuracy(targets, y_best)
    print("Accuracy:", accuracy)