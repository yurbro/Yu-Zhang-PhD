#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   analysis_bo.py
# Time    :   2025/06/11 20:28:22
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import matplotlib.pyplot as plt

def set_plot_style(dim, alpha_columns_5d, iterations_5d, path, df_5d, alpha_cols_5d, special_cols_5d):
    # 画图
    # plt.figure(figsize=(12, 6))
    # for col in alpha_columns_5d:
    #     plt.plot(iterations_5d, df_5d[col], label=col)
    # # Plot settings
    # plt.xlabel('Iteration')
    # plt.ylabel(f'Ackley Best Value ({dim}D)')
    # plt.title(f'Performance across different Alpha values ({dim}D)')
    # plt.legend(loc='best', fontsize='small', ncol=2)
    # # plt.grid(True)
    # plt.xticks(range(0, len(iterations_5d) + 1))
    # plt.savefig(path + f'\\alpha_performance_{dim}D.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 画图
    plt.figure(figsize=(14, 7))
    # 画 Alpha 系列（统一样式）
    for col in alpha_cols_5d:
        plt.plot(iterations_5d, df_5d[col], label=col, linestyle='-', linewidth=1, alpha=0.8)
    # 画特殊策略，使用不同的线型和颜色
    special_styles_5d = {
        'BO_EI': {'linestyle': '--', 'color': 'red', 'linewidth': 2},
        'BO_UCB': {'linestyle': '-.', 'color': 'blue', 'linewidth': 2},
        'BO_POI': {'linestyle': ':', 'color': 'green', 'linewidth': 2},
    }
    for col in special_cols_5d:
        if col in df_5d.columns:
            plt.plot(iterations_5d, df_5d[col], label=col, **special_styles_5d[col])
    # 图形设置
    plt.xlabel('Iteration', fontsize=12)
    # plt.xticks(fontsize=10)
    plt.ylabel(f'Ackley Best Value ({dim}D)', fontsize=12)
    plt.title(f'Strategy Performance Comparison: Alpha vs EI/UCB/POI ({dim}D)', fontsize=12)
    plt.legend(loc='best', fontsize='medium', ncol=2)
    plt.xticks(range(0, len(df_5d['Iteration']) + 1), fontsize=10)
    plt.savefig(path + f'\\alpha_comparison_{dim}D.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # 读取数据文件
    path = 'Multi-Objective Optimisation\\Benchmark\\Package Module-III\\BO-RE'
    # Test PROPOSED-xD
    dim = 3  # 维度
    filename = 'single_bo_results.xlsx'  # 修改为你的文件名
    sheetname = f'Alpha-Comparison-{dim}D'  # 修改为你的工作表名称
    df_5d = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  # 修改为你的文件路径
    # 提取 iteration 和所有 alpha 列
    iterations_5d = df_5d['Iteration']
    alpha_columns_5d = [col for col in df_5d.columns if col.startswith('Alpha')]
    # Plot the comparison of EI, HV, and Random for 5D
    sheetname_2_5d = f'Alpha-Comparison-{dim}D'  # 修改为你的工作表名称
    df_5d = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2_5d)  # 修改为你的文件路径
    # 定义列分组
    alpha_cols_5d = [col for col in df_5d.columns if col.startswith('Alpha')]
    special_cols_5d = ['BO_EI', 'BO_UCB', 'BO_POI']

    # 调用函数绘图
    set_plot_style(dim, alpha_columns_5d, iterations_5d, path, df_5d, alpha_cols_5d, special_cols_5d)
