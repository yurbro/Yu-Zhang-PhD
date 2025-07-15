#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   loocv_analysis.py
# Time    :   2025/06/13 16:51:18
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III-rosenbrock\Loocv_RE\LOOCV_Comparison.xlsx"
df = pd.read_excel(file_path, sheet_name='R2')

# 绘图
# plt.figure(figsize=(8, 6))
plt.plot(df['Iteration'], df['PROPOSED'], marker='o', label='PROPOSED')
plt.plot(df['Iteration'], df['EI'], marker='s', label='EI')
plt.plot(df['Iteration'], df['HV'], marker='^', label='HV')
plt.plot(df['Iteration'], df['RANDOM'], marker='d', label='RANDOM')

plt.xlabel('Iteration', fontsize=10)
plt.ylabel('R$^2$ Value')
plt.title('R$^2$ Value vs Iteration for Four Methods')
plt.legend()
# plt.xticks(df['Iteration'], rotation=45)
# plt.xticks(range(0, len(df['Iteration']) + 1))
plt.tight_layout()
plt.savefig(r"Multi-Objective Optimisation\Benchmark\\Package Module-III-rosenbrock\\Loocv_RE\\loocv_r2_comparison_Rosenbrock.png", dpi=300, bbox_inches='tight')
plt.show()