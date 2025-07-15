#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   analysis_bo.py
# Time    :   2025/06/11 20:28:22
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import matplotlib.pyplot as plt

def set_plot_style(dim, alpha_columns_5d, iterations_5d, path, df_5d, alpha_cols_5d, special_cols_5d):
    # Plotting
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
    # plt.savefig(path + f'\alpha_performance_{dim}D.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Plotting
    plt.figure(figsize=(14, 7))
    # Plot Alpha series (uniform style)
    for col in alpha_cols_5d:
        plt.plot(iterations_5d, df_5d[col], label=col, linestyle='-', linewidth=1, alpha=0.8)
    # Plot special strategies with different line styles and colors
    special_styles_5d = {
        'BO_EI': {'linestyle': '--', 'color': 'red', 'linewidth': 2},
        'BO_UCB': {'linestyle': '-.', 'color': 'blue', 'linewidth': 2},
        'BO_POI': {'linestyle': ':', 'color': 'green', 'linewidth': 2},
    }
    for col in special_cols_5d:
        if col in df_5d.columns:
            plt.plot(iterations_5d, df_5d[col], label=col, **special_styles_5d[col])
    # Figure settings
    plt.xlabel('Iteration', fontsize=12)
    # plt.xticks(fontsize=10)
    plt.ylabel(f'Ackley Best Value ({dim}D)', fontsize=12)
    plt.title(f'Strategy Performance Comparison: Alpha vs EI/UCB/POI ({dim}D)', fontsize=12)
    plt.legend(loc='best', fontsize='medium', ncol=2)
    plt.xticks(range(0, len(df_5d['Iteration']) + 1), fontsize=10)
    plt.savefig(path + f'\alpha_comparison_{dim}D.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # Read data file
    path = 'Multi-Objective Optimisation\\Benchmark\\Package Module-III\\BO-RE'
    # Test PROPOSED-xD
    dim = 3  # Dimension
    filename = 'single_bo_results.xlsx'  # Change to your filename
    sheetname = f'Alpha-Comparison-{dim}D'  # Change to your sheet name
    df_5d = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  # Change to your file path
    # Extract iteration and all alpha columns
    iterations_5d = df_5d['Iteration']
    alpha_columns_5d = [col for col in df_5d.columns if col.startswith('Alpha')]
    # Plot the comparison of EI, HV, and Random for 5D
    sheetname_2_5d = f'Alpha-Comparison-{dim}D'  # Change to your sheet name
    df_5d = pd.read_excel(path + '/' + filename, sheet_name=sheetname_2_5d)  # Change to your file path
    # Define column groups
    alpha_cols_5d = [col for col in df_5d.columns if col.startswith('Alpha')]
    special_cols_5d = ['BO_EI', 'BO_UCB', 'BO_POI']

    # Call function to plot
    set_plot_style(dim, alpha_columns_5d, iterations_5d, path, df_5d, alpha_cols_5d, special_cols_5d)
