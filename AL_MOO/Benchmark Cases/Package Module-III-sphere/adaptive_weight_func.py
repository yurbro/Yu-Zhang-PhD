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
    

    a_ei = accuracies.get("ei", 0.0)
    a_hv = accuracies.get("hv", 0.0)
    total_acc = a_ei + a_hv

    if total_acc == 0:
        norm_a_ei = 0.5
        norm_a_hv = 0.5
    else:
        norm_a_ei = a_ei / (total_acc + epsilon)
        norm_a_hv = a_hv / (total_acc + epsilon)

    w_old_ei = prev_weights.get("ei", 0.5)
    w_new_ei = alpha * w_old_ei + (1 - alpha) * norm_a_ei
    w_new_hv = 1.0 - w_new_ei  # Ensure normalization

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
   
    targets = np.array(targets)
    y_best = np.array(y_best)

    count_exceeding = np.sum(targets >= y_best)

    accuracy = count_exceeding / len(targets) if len(targets) > 0 else 0.0

    return accuracy

if __name__ == "__main__":

  
    prev_weights = {"ei": 0.5, "hv": 0.5}       # Insert your previous weights here
    accuracies = {"ei": 0.0, "hv": 0.0}     # Insert your accuracies here
    k = 6
    alpha = 0.5                                

    new_weights, allocation = update_normalised_weights_and_allocate(
        prev_weights, accuracies, k, alpha, rounding='floor', epsilon=1e-9
    )

    print("New Weights:", new_weights)
    print("Point Allocation:", allocation)
    # Example usage of calculate_accuracy
    targets = [0.8, 0.9, 0.85, 0.95, 0.7]  
    y_best = 0.9  
    accuracy = calculate_accuracy(targets, y_best)
    print("Accuracy:", accuracy)