#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   imp_comparison.py
# Time    :   2025/06/13 21:31:35
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取Excel
# file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III\Ave_Improve\improvement_comparison.xlsx"
# df = pd.read_excel(file_path)

# # 设置Iteration为横坐标
# x = df['Iteration']

# # Alpha类用不同颜色
# alpha_cols = ['Alpha-0.5 Imp', 'Alpha-0.6 Imp', 'Alpha-0.7 Imp', 'Alpha-0.8 Imp']
# alpha_colors = ['blue', 'purple', 'cyan', 'green']
# for col, color in zip(alpha_cols, alpha_colors):
#     plt.plot(x, df[col], color=color, marker='o', linestyle='-', label=col)

# # 其他三类
# plt.plot(x, df['EI Imp'], color='red', marker='s', linestyle='-', label='EI Imp')
# plt.plot(x, df['HV Imp'], color='orange', marker='^', linestyle='-', label='HV Imp')
# plt.plot(x, df['RANDOM Imp'], color='grey', marker='d', linestyle='-', label='RANDOM Imp')

# plt.xlabel('Iteration')
# plt.ylabel('Improvement')
# plt.title('Improvement Comparison')
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# ========= 1. 数据读取 =========
# file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III-ackley\Ave_Improve\improvement_comparison.xlsx"  # compare af strategies
file_path = r"Multi-Objective Optimisation\Benchmark\Package Module-III-ackley\Ave_Improve\improvement_comparison - bo.xlsx"     # compare bo

df = pd.read_excel(file_path, sheet_name='Sheet1')
x = df['Iteration']

# ========= 2. 画布 & 风格 =========
plt.style.use('seaborn-v0_8-colorblind')   # 友好配色
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# ========= 3. 颜色 & 标记 =========
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

# ========= 4. 绘图 =========
for label, style in series_specs.items():
    y = df[label]
    ax.plot(x, y, label=label,
            linewidth=2.2, markersize=7,
            marker=style['marker'], color=style['color'], alpha=0.9)
    # 让非零点再叠一层散点，凸显极值
    nz = y != 0
    ax.scatter(x[nz], y[nz], color=style['color'], s=45, zorder=3)

# ========= 5. 轴 & 网格 =========
ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Improvement', fontsize=14)
ax.set_title('Improvement Comparison', fontsize=16, weight='bold')
# ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.set_ylim(bottom=0)        # 如果想截断/对数，可在这里改
# ax.set_yscale('log')       # 可选：差距过大时打开

# ========= 6. 图例 =========
ax.legend(fontsize=11, loc='best')

# ========= 7. 显示/保存 =========
plt.tick_params(axis='both', labelsize=12)  # 设置横纵坐标数字大小为12
plt.tight_layout()
plt.savefig('Multi-Objective Optimisation\Benchmark\Package Module-III-ackley\Ave_Improve\improvement_comparison_bo.png', dpi=300, bbox_inches='tight')
plt.show()

# plt.tight_layout()
# plt.show()

# plt.plot(x, y, label='曲线')
# # plt.legend(loc='best')  # 图例放在右上角
# plt.show()
