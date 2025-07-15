#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   imp_comparison.py
# Time    :   2025/06/13 21:31:35
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk


import pandas as pd
import matplotlib.pyplot as plt

# ========= 1. read data =========
# file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III-ackley\Ave_Improve\improvement_comparison.xlsx"  # compare af strategies
file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III-ackley\Ave_Improve\improvement_comparison - bo.xlsx"     # compare bo

df = pd.read_excel(file_path, sheet_name='Sheet1')
x = df['Iteration']

# ========= 2. canvas & style =========
plt.style.use('seaborn-v0_8-colorblind')   # friendly color
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# ========= 3. color & marker =========
series_specs = {
    'Alpha-0.5 Imp': dict(color='#1f77b4', marker='o'),
    'Alpha-0.6 Imp': dict(color='#9467bd', marker='o'),
    'Alpha-0.7 Imp': dict(color='#17becf', marker='o'),
    'Alpha-0.8 Imp': dict(color='#2ca02c', marker='o'),
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
    
    nz = y != 0
    ax.scatter(x[nz], y[nz], color=style['color'], s=45, zorder=3)

# ========= 5. ax & grid =========
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Improvement', fontsize=14)
ax.set_title('Improvement Comparison', fontsize=16, weight='bold')
# ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.set_ylim(bottom=0)        # If you want to truncate/log scale, you can change it here
# ax.set_yscale('log')       # Optional: Open it when the gap is too large

# ========= 6. legend =========
ax.legend(fontsize=11, loc='best')

# ========= 7. show/save =========
plt.tick_params(axis='both', labelsize=12)  # Set x and y axis tick label size to 12
plt.tight_layout()
plt.savefig('Multi-Objective Optimisation\Benchmark\Package Module-III-ackley\Ave_Improve\improvement_comparison_bo.png', dpi=300, bbox_inches='tight')
plt.show()
