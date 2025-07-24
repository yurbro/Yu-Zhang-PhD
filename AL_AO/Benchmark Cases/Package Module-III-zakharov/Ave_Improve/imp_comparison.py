#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   imp_comparison.py
# Time    :   2025/06/13 21:31:35
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk



import pandas as pd
import matplotlib.pyplot as plt

# ========= 1. data loading =========
# file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III\Ave_Improve\improvement_comparison - Zakharov.xlsx"  # compare af strategies
path = 'Multi-Objective Optimisation\Benchmark\Package Module-III-zakharov\Ave_Improve'
file_path = path + r"\improvement_comparison - bo - Zakharov.xlsx"     # compare bo

df = pd.read_excel(file_path, sheet_name='Sheet1')
x = df['Iteration']

# ========= 2. canvas & style =========
plt.style.use('seaborn-v0_8-colorblind')   # Friendly color
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# ========= 3. color & marker =========
series_specs = {
    'Alpha-0.5 Imp': dict(color='#1f77b4', marker='o'),
    # 'EI Imp':        dict(color='#d62728', marker='P'),
    # 'HV Imp':        dict(color='#ff7f0e', marker='X'),
    # 'RANDOM Imp':    dict(color='#7f7f7f', marker='D'),
    'BO_EI Imp':        dict(color='#d62728', marker='P'),
    'BO_UCB Imp':        dict(color='#ff7f0e', marker='X'),
    'BO_POI Imp':    dict(color='#7f7f7f', marker='D'),
}

# ========= 4. plot =========
for label, style in series_specs.items():
    y = df[label]
    ax.plot(x, y, label=label,
            linewidth=2.2, markersize=7,
            marker=style['marker'], color=style['color'], alpha=0.9)
    # Scatter points for non-zero values
    # This avoids cluttering the plot with too many points
    nz = y != 0
    ax.scatter(x[nz], y[nz], color=style['color'], s=45, zorder=3)

# ========= 5. axis & grid =========
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Improvement', fontsize=14)
ax.set_title('Improvement Comparison', fontsize=16, weight='bold')
# ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.set_ylim(bottom=0)        # If you want to truncate/log, you can change it here
# ax.set_yscale('log')       # Optional: Enable this if the gap is too large

# ========= 6. legend =========
ax.legend(fontsize=11, loc='best')

# ========= 7. Show/Save =========
plt.tick_params(axis='both', labelsize=12)  # Set the size of the x and y axis numbers to 12
plt.tight_layout()
plt.savefig(path + r'\improvement_comparison_bo_Zakharov.png', dpi=300, bbox_inches='tight')
plt.show()
