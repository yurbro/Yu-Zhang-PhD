#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   statistical_significance_test.py
# Time    :   2025/06/12 20:38:21
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

"""
    This module to validate the proposed method better than the baseline methods, rather than the occasionally better performance.
"""

from scipy.stats import ttest_rel, ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the best series from your results and the baseline results

path = "Multi-Objective Optimisation\Benchmark\Package Module-III\BO-RE\single_bo_results.xlsx"

df = pd.read_excel(path, sheet_name='Alpha-Comparison-3D')

proposed = df['Alpha-0.5'].values  # Replace with your proposed method column
bo_ei = df['BO_EI'].values  # Replace with your baseline method column

# stat, p_value = ttest_rel(proposed, bo_ei)
# print(f"P-value (Proposed vs BO_EI) = {p_value:.4f}")

# 假设你的DataFrame叫df，列名分别为proposed, BO_EI, BO_UCB, BO_POI

p_ei = ttest_ind(proposed, bo_ei, equal_var=False).pvalue
p_ucb = ttest_ind(proposed, df['BO_UCB'], equal_var=False).pvalue
p_poi = ttest_ind(proposed, df['BO_POI'], equal_var=False).pvalue

print(f"P-value (Proposed vs BO_EI): {p_ei:.4f}")
print(f"P-value (Proposed vs BO_UCB): {p_ucb:.4f}")
print(f"P-value (Proposed vs BO_POI): {p_poi:.4f}")