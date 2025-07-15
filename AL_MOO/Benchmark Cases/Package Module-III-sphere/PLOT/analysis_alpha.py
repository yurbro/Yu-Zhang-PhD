#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   analysis_alpha.py
# Time    :   2025/06/10 10:16:13
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import matplotlib.pyplot as plt

# def set_plot_style(dim, alpha_columns_5d, iterations_5d, path, df_5d, alpha_cols_5d, special_cols_5d):

#     plt.figure(figsize=(12, 6))
#     for col in alpha_columns_5d:
#         plt.plot(iterations_5d, df_5d[col], label=col)
#     # Plot settings
#     plt.xlabel('Iteration')
#     plt.ylabel(f'Ackley Best Value ({dim}D)')
#     plt.title(f'Performance across different Alpha values ({dim}D)')
#     plt.legend(loc='best', fontsize='small', ncol=2)
#     # plt.grid(True)
#     plt.xticks(range(0, len(iterations_5d) + 1))
#     plt.savefig(path + f'\\alpha_performance_{dim}D.png', dpi=300, bbox_inches='tight')
#     plt.show()


#     plt.figure(figsize=(14, 7))

#     for col in alpha_cols_5d:
#         plt.plot(iterations_5d, df_5d[col], label=col, linestyle='-', linewidth=1, alpha=0.8)

#     special_styles_5d = {
#         'EI': {'linestyle': '--', 'color': 'red', 'linewidth': 2},
#         'HV': {'linestyle': '-.', 'color': 'blue', 'linewidth': 2},
#         'Random': {'linestyle': ':', 'color': 'green', 'linewidth': 2},
#     }
#     for col in special_cols_5d:
#         if col in df_5d.columns:
#             plt.plot(iterations_5d, df_5d[col], label=col, **special_styles_5d[col])

#     plt.xlabel('Iteration')
#     plt.ylabel(f'Ackley Best Value ({dim}D)')
#     plt.title(f'Strategy Performance Comparison: Alpha vs EI/HV/Random ({dim}D)')
#     plt.legend(loc='best', fontsize='small', ncol=2)
#     plt.xticks(range(0, len(df_5d['Iteration']) + 1))
#     plt.savefig(path + f'\\alpha_comparison_{dim}D.png', dpi=300, bbox_inches='tight')
#     plt.show()



# if __name__ == "__main__":
    

#     path = 'Multi-Objective Optimisation/Benchmark/Package Module-III/PLOT'
#     # Test PROPOSED-xD
#     dim = 3  
#     filename = f'PROPOSED - {dim}D.xlsx'  
#     sheetname = f'Alpha-Ackley-{dim}D'  
#     df_5d = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  

#     iterations_5d = df_5d['Iteration']
#     alpha_columns_5d = [col for col in df_5d.columns if col.startswith('Alpha')]
#     # Plot the comparison of EI, HV, and Random for 5D
#     sheetname_2_5d = f'Alpha-Comparison-{dim}D'  
#     df_5d = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2_5d)  

#     alpha_cols_5d = [col for col in df_5d.columns if col.startswith('Alpha')]
#     special_cols_5d = ['EI', 'HV', 'Random']
    

#     set_plot_style(dim, alpha_columns_5d, iterations_5d, path, df_5d, alpha_cols_5d, special_cols_5d)



filename = 'PROPOSED.xlsx'  
sheetname = 'Alpha-Fixed' 
path = 'Multi-Objective Optimisation/Benchmark/Package Module-III-sphere/PLOT'
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  


iterations = df['Iteration']
alpha_columns = [col for col in df.columns if col.startswith('Proposed')]


plt.figure(figsize=(12, 6))
for col in alpha_columns:
    plt.plot(iterations, df[col], label=col)

# Plot settings
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Ackley Best Value', fontsize=14)
# plt.title('Performance across different Proposed values')
plt.legend(loc='best', fontsize='large', ncol=2)
# plt.grid(True)
# plt.tight_layout()
plt.xticks(range(0, len(iterations) + 1))
plt.savefig(path + '\\alpha_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the comparison of EI, HV, and Random
sheetname_2 = 'Alpha-Fixed-Comparison'  
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2)  

alpha_cols = [col for col in df.columns if col.startswith('Proposed')]
special_cols = ['EI', 'HV', 'Random']


plt.figure(figsize=(14, 7))


for col in alpha_cols:
    plt.plot(iterations, df[col], label=col, linestyle='-', linewidth=1, alpha=0.8)


special_styles = {
    'EI': {'linestyle': '--', 'color': 'red', 'linewidth': 2},
    'HV': {'linestyle': '-.', 'color': 'blue', 'linewidth': 2},
    'Random': {'linestyle': ':', 'color': 'green', 'linewidth': 2},
}

for col in special_cols:
    if col in df.columns:
        plt.plot(iterations, df[col], label=col, **special_styles[col])


plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Sphere Best Value', fontsize=14)
plt.title('Strategy Performance Comparison: Proposed vs EI/HV/Random', fontsize=16)
plt.legend(loc='best', fontsize='large', ncol=2)
plt.xticks(range(0, len(df['Iteration']) + 1))
plt.savefig(path + '\\alpha_comparison-sphere.png', dpi=300, bbox_inches='tight')
plt.show()


