

import pandas as pd
import matplotlib.pyplot as plt

# 设置路径和文件名
path = 'Multi-Objective Optimisation/Benchmark/Package Module-III/GPT-PLOT'

filename = 'PROPOSED.xlsx'  # 修改为你的文件名
sheetname = 'Alpha-Iter'  # 修改为你的工作表名称
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  # 修改为你的文件路径

# 提取 iteration 和所有 alpha 列
iterations = df['Alpha']
alpha_columns = [col for col in df.columns if col.startswith('Iter')]

# 画图
plt.figure(figsize=(12, 6))
for col in alpha_columns:
    plt.plot(iterations, df[col], label=col)

# Plot settings
plt.xlabel('Alpha Value')
plt.ylabel('Ackley Best Value')
plt.title('Performance across different Alpha values')
plt.legend(loc='best', fontsize='small', ncol=2)
# plt.xticks(range(0, len(iterations), 1))
plt.savefig(path + '\\alpha_iter_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# # Plot the comparison of EI, HV, and Random
# sheetname_2 = 'Alpha-Comparison'  # 修改为你的工作表名称
# df = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2)  # 修改为你的文件路径
# # 定义列分组
# alpha_cols = [col for col in df.columns if col.startswith('Alpha')]
# special_cols = ['EI', 'HV', 'Random']

# # 画图
# plt.figure(figsize=(14, 7))

# # 画 Alpha 系列（统一样式）
# for col in alpha_cols:
#     plt.plot(iterations, df[col], label=col, linestyle='-', linewidth=1, alpha=0.8)

# # 画特殊策略，使用不同的线型和颜色
# special_styles = {
#     'EI': {'linestyle': '--', 'color': 'red', 'linewidth': 2},
#     'HV': {'linestyle': '-.', 'color': 'blue', 'linewidth': 2},
#     'Random': {'linestyle': ':', 'color': 'green', 'linewidth': 2},
# }

# for col in special_cols:
#     if col in df.columns:
#         plt.plot(iterations, df[col], label=col, **special_styles[col])

# # 图形设置
# plt.xlabel('Iteration')
# plt.ylabel('Ackley Best Value')
# plt.title('Strategy Performance Comparison: Alpha vs EI/HV/Random')
# plt.legend(loc='best', fontsize='small', ncol=2)
# plt.xticks(range(0, len(df['Iteration']) + 1))
# plt.savefig(path + '\\alpha_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()
