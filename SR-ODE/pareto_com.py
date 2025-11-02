#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   pareto_com.py
# Time    :   2025/08/07 16:46:22
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Desc    :   Compare Pareto front results from different runs

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def compare_pareto_fronts(run_numbers, output_dir='Symbolic Regression/srloop/data/'):
    """
    Compare Pareto fronts from different runs.
    :param run_numbers: List of run numbers to compare.
    :param output_dir: Directory where the results are stored.
    """
    plt.figure(figsize=(10, 6))
    for run in range(1, run_numbers + 1):
        pareto_data = pd.read_csv(f"{output_dir}/hall_of_fame_run-{run}.csv")
        plt.plot(pareto_data['complexity'], pareto_data['loss'], label=f"Run-{run}")
    plt.xlabel("Complexity")
    plt.ylabel("Loss")
    plt.title("Pareto Front Comparison for Each Run")
    plt.legend()
    plt.show()

    # 新增：只画complexity在5到20之间的对比图
def compare_pareto_fronts_complexity_range(run_numbers, output_dir='Symbolic Regression/srloop/data/', min_complexity=5, max_complexity=20):
    """
    Compare Pareto fronts from different runs, only for complexity in [min_complexity, max_complexity].
    :param run_numbers: Number of runs to compare.
    :param output_dir: Directory where the results are stored.
    :param min_complexity: Minimum complexity to include.
    :param max_complexity: Maximum complexity to include.
    """
    plt.figure(figsize=(10, 6))
    for run in range(1, run_numbers + 1):
        pareto_data = pd.read_csv(f"{output_dir}/hall_of_fame_run-{run}.csv")
        filtered = pareto_data[(pareto_data['complexity'] >= min_complexity) & (pareto_data['complexity'] <= max_complexity)]
        plt.plot(filtered['complexity'], filtered['loss'], label=f"Run-{run}")
    plt.xlabel("Complexity")
    plt.ylabel("Loss")
    plt.title(f"Pareto Front Comparison (Complexity {min_complexity}-{max_complexity})")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Example usage
    run_numbers = 5  # Replace with actual run numbers
    compare_pareto_fronts(run_numbers)

    # Just show the complexity from 5 to 20
    compare_pareto_fronts_complexity_range(run_numbers, min_complexity=5, max_complexity=20)



