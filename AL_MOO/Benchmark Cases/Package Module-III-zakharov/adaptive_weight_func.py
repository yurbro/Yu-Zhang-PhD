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
    Update the weights of EI and HV using normalized accuracy and allocate experiment points.

    Parameters:
    -----------
    prev_weights : dict
        {"ei": float, "hv": float}, normalized weights from previous round
    accuracies : dict
        {"ei": float, "hv": float}, accuracies for current round
    k : int
        Total number of experiment points for current round
    alpha : float
        Smoothing coefficient (0 = no smoothing, 1 = keep old weights)
    rounding : str
        Rounding method for point allocation: 'floor' or 'round'
    epsilon : float
        Small value to avoid division by zero (for normalization)

    Returns:
    --------
    new_weights : dict
        {"ei": float, "hv": float}, new normalized weights
    allocation : dict
        {"ei": int, "hv": int}, number of experiment points allocated
    """

    a_ei = accuracies.get("ei", 0.0)
    a_hv = accuracies.get("hv", 0.0)
    total_acc = a_ei + a_hv

    # ===== Step 1: Special case handling (all zero) =====
    if total_acc == 0:
        norm_a_ei = 0.5
        norm_a_hv = 0.5
    else:
        norm_a_ei = a_ei / (total_acc + epsilon)
        norm_a_hv = a_hv / (total_acc + epsilon)

    # ===== Step 2: Update weights using exponential smoothing =====
    w_old_ei = prev_weights.get("ei", 0.5)
    w_new_ei = alpha * w_old_ei + (1 - alpha) * norm_a_ei
    w_new_hv = 1.0 - w_new_ei  # Ensure normalization

    # ===== Step 3: Allocate experiment points =====
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
    Calculate the probability that the objective values of Pareto points exceed the maximum value y_best.
    Parameters:
    -----------
    targets : list or np.array
        Actual objective values
    y_best : float
        Current best objective value

    Returns:
    --------
    accuracy : float
        Accuracy, range [0, 1]
    """
    targets = np.array(targets)
    y_best = np.array(y_best)

    # Count the number of objective values exceeding y_best
    count_exceeding = np.sum(targets >= y_best)

    # Calculate accuracy
    accuracy = count_exceeding / len(targets) if len(targets) > 0 else 0.0

    return accuracy

if __name__ == "__main__":

    # Initial conditions
    """
        Just save new_weights as prev_weights to use iteratively in each round.
    """
    prev_weights = {"ei": 0.5, "hv": 0.5}       # Insert your previous weights here
    accuracies = {"ei": 0.0, "hv": 0.0}     # Insert your accuracies here
    k = 6
    alpha = 0.5                                 # Smoothing coefficient, 0 = no smoothing, 1 = keep old weights, < 1 = mix old and new weights

    # Update weights + allocate points
    new_weights, allocation = update_normalised_weights_and_allocate(
        prev_weights, accuracies, k, alpha, rounding='floor', epsilon=1e-9
    )

    print("New Weights:", new_weights)
    print("Point Allocation:", allocation)

    # Example: calculate accuracy
    targets = [0.8, 0.9, 0.85, 0.95, 0.7]  # Example objective values
    y_best = 0.9  # Current best objective value
    accuracy = calculate_accuracy(targets, y_best)
    print("Accuracy:", accuracy)