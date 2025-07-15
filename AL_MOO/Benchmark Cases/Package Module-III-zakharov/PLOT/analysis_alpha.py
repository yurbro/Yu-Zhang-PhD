#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   analysis_alpha.py
# Time    :   2025/06/10 10:16:13
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import matplotlib.pyplot as plt

filename = 'PROPOSED.xlsx'  # change to your file name
# Read the Excel file
sheetname = 'Alpha-Fixed'  # change to your sheet name
path = 'Multi-Objective Optimisation/Benchmark/Package Module-III-zakharov/PLOT'
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  # change to your file path

# Extract iteration and all alpha columns
iterations = df['Iteration']
alpha_columns = [col for col in df.columns if col.startswith('Proposed')]

# Plot
plt.figure(figsize=(12, 6))
for col in alpha_columns:
    plt.plot(iterations, df[col], label=col)

# Plot settings
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Ackley Best Value', fontsize=14)
plt.title('Performance across different Proposed', fontsize=16)
plt.legend(loc='best', fontsize='small', ncol=2)
# plt.grid(True)
# plt.tight_layout()
plt.xticks(range(0, len(iterations) + 1))
plt.savefig(path + '\\alpha_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the comparison of EI, HV, and Random
sheetname_2 = 'Alpha-Fixed-Comparison'  # change to your comparison sheet name
# Read the comparison sheet
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2)  # change to your file path
# Define column groups
alpha_cols = [col for col in df.columns if col.startswith('Proposed')]
special_cols = ['EI', 'HV', 'Random']

# Plot
plt.figure(figsize=(14, 7))

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
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Zakharov Best Value', fontsize=14)
plt.title('Strategy Performance Comparison: Proposed vs EI/HV/Random', fontsize=16)
plt.legend(loc='best', fontsize='small', ncol=2)
plt.xticks(range(0, len(df['Iteration']) + 1))
plt.savefig(path + '\\alpha_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


