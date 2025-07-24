#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   analysis_alpha.py
# Time    :   2025/06/10 10:16:13
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import matplotlib.pyplot as plt


path = 'Multi-Objective Optimisation/Benchmark/Package Module-III/PLOT'
filename = 'PROPOSED.xlsx'  # change to your filename
sheetname = 'Alpha-Fixed'  # change to your sheet name
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  # change to your file path

iterations = df['Iteration']
alpha_columns = [col for col in df.columns if col.startswith('Alpha')]

# Plot the performance of different Alpha values
plt.figure(figsize=(12, 6))
for col in alpha_columns:
    plt.plot(iterations, df[col], label=col)

# Plot settings
plt.xlabel('Iteration')
plt.ylabel('Ackley Best Value')
plt.title('Performance across different Alpha values')
plt.legend(loc='best', fontsize='small', ncol=2)
# plt.grid(True)
# plt.tight_layout()
plt.xticks(range(0, len(iterations) + 1))
plt.savefig(path + '\\alpha_fixed_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the comparison of EI, HV, and Random
sheetname_2 = 'Alpha-Fixed-Comparison'  # change to your sheet name
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2)  # change to your file path
# Define column groups
alpha_cols = [col for col in df.columns if col.startswith('Alpha')]
special_cols = ['EI', 'HV', 'Random']

# Plot the performance of Alpha and special strategies
plt.figure(figsize=(14, 7))

# Plot Alpha series (uniform style)
for col in alpha_cols:
    plt.plot(iterations, df[col], label=col, linestyle='-', linewidth=1, alpha=0.8)

# Plot special strategies with different line styles and colors
special_styles = {
    'EI': {'linestyle': '--', 'color': 'red', 'linewidth': 2},
    'HV': {'linestyle': '-.', 'color': 'blue', 'linewidth': 2},
    'Random': {'linestyle': ':', 'color': 'green', 'linewidth': 2},
}

for col in special_cols:
    if col in df.columns:
        plt.plot(iterations, df[col], label=col, **special_styles[col])

# Plot settings
plt.xlabel('Iteration')
plt.ylabel('Ackley Best Value')
plt.title('Strategy Performance Comparison: Alpha vs EI/HV/Random')
plt.legend(loc='best', fontsize='small', ncol=2)
plt.xticks(range(0, len(df['Iteration']) + 1))
plt.savefig(path + '\\alpha_fixed_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


