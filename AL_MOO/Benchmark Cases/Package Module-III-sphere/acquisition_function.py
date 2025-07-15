#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   acquisitionfunction_RUN3-PROPOSED.py
# Time    :   2025/06/04 19:37:56
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from adaptive_weight_func import update_normalised_weights_and_allocate
import os


def expected_improvement(mu_cand, var_cand, mu_plus, jitter=1e-9):
    xi = 0.1
    sigma = np.sqrt(np.maximum(var_cand, 0.0)) + jitter
    u = mu_cand - mu_plus - xi
    z = u / sigma
    return sigma * norm.pdf(z) + u * norm.cdf(z)

def parse_x(xstr):
    return np.fromstring(xstr.replace('[', '').replace(']', ''), sep=' ')

def plot_ei(ei_vals, path, savefig):
    plt.plot(range(1, len(ei_vals) + 1), ei_vals, marker='o', label='EI')
    plt.xlabel('Pareto Points')
    plt.ylabel('EI Value')
    plt.legend()
    plt.tight_layout()
    plt.xticks(range(1, len(ei_vals) + 1))
    if savefig:
        plt.savefig(f'{path}\\EI_Pareto_Points.png', dpi=300, bbox_inches='tight')  # epoch 1 TODO: code to choose save or not
    # plt.show()

def hv_2d(points, ref):
    sorted_pts = sorted(points, key=lambda pt: pt[0], reverse=True)
    hv, prev_x = 0.0, ref[0]
    for x, y in sorted_pts:
        hv += (prev_x - x) * (ref[1] - y)
        prev_x = x
    return hv

def plot_hv_contrib(x_vals, y_vals, path, savefig):
    x_vals = range(1, len(x_vals) + 1)

    if any(y < 0 for y in y_vals):
        y_vals = [0] * len(y_vals)

    plt.plot(x_vals, y_vals, marker='o', label='HV Contribution')
    plt.xlabel('Pareto Point Order (sorted by HV contribution)')
    plt.ylabel('Hypervolume Contribution')
    plt.legend()
    plt.tight_layout()
    plt.xticks(x_vals)
    if savefig:
        plt.savefig(f'{path}\\HV_contribution.png', dpi=300)  # epoch 1 TODO: code to choose save or not
    # plt.show()

def plot_ei_and_hv(ei_vals, hv_vals, path, savefig):
    fig, ax1 = plt.subplots()
    color1 = 'tab:blue'
    color2 = 'tab:red'
    x = np.arange(1, len(ei_vals) + 1)

    ax1.set_xlabel('Pareto Points')
    ax1.set_ylabel('EI', color=color1)
    ax1.plot(x, ei_vals, marker='o', color=color1, label='EI')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('HV Contribution', color=color2)
    ax2.plot(x, hv_vals, marker='s', color=color2, label='HV Contribution')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.title('EI and HV Contribution')
    plt.xticks(x)
    if savefig:
        plt.savefig(f'{path}\\EI_HV_Combined.png', dpi=300, bbox_inches='tight')    # epoch 1 TODO: code to choose save or not
    # plt.show()

def save_top_candidates(dim, pareto_front, ei_vals, df, path_data, run_num, method, total, k):
    """
    保存候选点到Excel，根据method选择不同的策略:
    - method='EI': 只保存EI最大的total个点
    - method='HV': 只保存HV贡献最大的total个点
    - method='Random': 随机选择total个点
    - method='Hybrid'（默认）: 保存EI最大的k个点和HV贡献最大的(total-k)个点
    """
    save_path = f'{path_data}\\Top-RUN{run_num}-{method}.xlsx'

    if method == f'EI-{dim}D':
        # 只用EI，保存EI最大的total个点
        if np.all(ei_vals <= 0):
            print(f"All EI values are non-positive. Selecting top {total} candidates by Mean instead.")
            top_idx = np.argsort(pareto_front['Mean'].values)[-total:][::-1]
        else:
            top_idx = np.argsort(ei_vals)[-total:][::-1]
            print(f"Selecting top {total} candidates by EI.")
        top_ei = pareto_front.iloc[top_idx][['X', 'Mean', 'Std']]
        top_ei = top_ei.copy()
        top_ei['EI'] = ei_vals[top_idx]
        top_ei.insert(0, 'Num', top_idx + 1)
        # 自动根据输入的维度确定X的个数
        first_x = top_ei['X'].iloc[0]
        x_dim = len(parse_x(first_x))
        x_col_names = [f'X{i+1}' for i in range(x_dim)]
        X_split = top_ei['X'].apply(lambda x: pd.Series(parse_x(x), index=x_col_names))
        top_ei = pd.concat([top_ei[['Num']], X_split, top_ei.drop(columns=['X'])], axis=1)
        sheetname_ei = 'Top_EI'
        with pd.ExcelWriter(save_path) as writer:
            top_ei.to_excel(writer, sheet_name=sheetname_ei, index=False)
        print(f"Top {total} candidates by EI saved to '{os.path.basename(save_path)}'.")
        return save_path, sheetname_ei

    elif method == f'HV-{dim}D':
        # 只用HV，保存HV贡献最大的total个点
        top_hv = df.sort_values("HV_Contribution", ascending=False).head(total)
        top_hv = top_hv[['Num', 'X', 'Mean', 'Std', 'HV_Contribution']]

        # 自动根据输入的维度确定X的个数
        first_x = top_hv['X'].iloc[0]
        x_dim = len(parse_x(first_x))
        x_col_names = [f'X{i+1}' for i in range(x_dim)]
        X_split = top_hv['X'].apply(lambda x: pd.Series(parse_x(x), index=x_col_names))
        top_hv = pd.concat([top_hv[['Num']], X_split, top_hv.drop(columns=['X'])], axis=1)
        sheetname_hv = 'Top_HV'
        with pd.ExcelWriter(save_path) as writer:
            top_hv.to_excel(writer, sheet_name=sheetname_hv, index=False)
        print(f"Top {total} candidates by HV contribution saved to '{os.path.basename(save_path)}'.")
        return save_path, sheetname_hv

    elif method == f'RANDOM-{dim}D':
        # 随机选择total个点
        # np.random.seed(5)  # 3D
        np.random.seed(9)  # 5D
        total = min(total, len(pareto_front))  # 确保不超过pareto_front的长度
        idx = np.random.choice(len(pareto_front), size=total, replace=False)
        top_rand = pareto_front.iloc[idx][['X', 'Mean', 'Std']]
        top_rand = top_rand.copy()
        top_rand['EI'] = ei_vals[idx]
        top_rand.insert(0, 'Num', idx + 1)
        # 自动根据输入的维度确定X的个数
        first_x = top_rand['X'].iloc[0]
        x_dim = len(parse_x(first_x))
        x_col_names = [f'X{i+1}' for i in range(x_dim)]
        X_split = top_rand['X'].apply(lambda x: pd.Series(parse_x(x), index=x_col_names))
        top_rand = pd.concat([top_rand[['Num']], X_split, top_rand.drop(columns=['X'])], axis=1)
        sheetname_random = 'Top_Random'
        with pd.ExcelWriter(save_path) as writer:
            top_rand.to_excel(writer, sheet_name=sheetname_random, index=False)
        print(f"{total} random candidates saved to '{os.path.basename(save_path)}'.")
        return save_path, sheetname_random

    else:
        # 默认Hybrid: EI最大的k个点 + HV最大的(total-k)个点
        k_EI = k
        k_HV = total - k

        if np.all(ei_vals <= 0):
            print(f"All EI values are non-positive. Selecting top {k_EI} candidates by Mean instead.")
            top_ei_idx = np.argsort(pareto_front['Mean'].values)[-k_EI:][::-1]
        else:
            top_ei_idx = np.argsort(ei_vals)[-k_EI:][::-1]
            print(f"Selecting top {k_EI} candidates by EI.")

        top_ei = pareto_front.iloc[top_ei_idx][['X', 'Mean', 'Std']]
        top_ei = top_ei.copy()
        top_ei['EI'] = ei_vals[top_ei_idx]
        top_ei.insert(0, 'Num', top_ei_idx + 1)
        # 自动根据输入的维度确定X的个数
        first_x = top_ei['X'].iloc[0]
        x_dim = len(parse_x(first_x))
        x_col_names = [f'X{i+1}' for i in range(x_dim)]
        X_split_ei = top_ei['X'].apply(lambda x: pd.Series(parse_x(x), index=x_col_names))
        top_ei = pd.concat([top_ei[['Num']], X_split_ei, top_ei.drop(columns=['X'])], axis=1)

        top_hv = df.sort_values("HV_Contribution", ascending=False).head(k_HV)
        top_hv = top_hv[['Num', 'X', 'Mean', 'Std', 'HV_Contribution']]
        # 自动根据输入的维度确定X的个数
        first_x = top_hv['X'].iloc[0]
        x_dim = len(parse_x(first_x))
        x_col_names = [f'X{i+1}' for i in range(x_dim)]
        X_split_hv = top_hv['X'].apply(lambda x: pd.Series(parse_x(x), index=x_col_names))
        top_hv = pd.concat([top_hv[['Num']], X_split_hv, top_hv.drop(columns=['X'])], axis=1)
        sheetname_eihv = ['Top3_EI', 'Top3_HV']

        with pd.ExcelWriter(save_path) as writer:
            top_ei.to_excel(writer, sheet_name=sheetname_eihv[0], index=False)
            top_hv.to_excel(writer, sheet_name=sheetname_eihv[1], index=False)

        print(f"Top {k_EI} candidates by EI and {k_HV} candidates by HV contribution saved to '{os.path.basename(save_path)}'.")
        return save_path, sheetname_eihv

def run_acquisition_function(dim, run_num, path_af, path_data, path_data_run, method, total, k, benchmark, savefig):
    # Load Pareto front and hyperparameters
    # run_num = 10
    pareto_front = pd.read_excel(f'{path_data}\RUN-{run_num}-{method}\pareto_front.xlsx')     # Use this for epoch 1
    mu_cand, std_cand = pareto_front['Mean'].values, pareto_front['Std'].values
    var_cand = std_cand ** 2

    # RUN-n-PROPOSED
    run_num_hist = run_num - 1
    file_path = f'{path_data}\\lhs_samples_{benchmark.lower()}_{method}.xlsx'     # Update the file path
    sheetname = f"RUN-{run_num_hist}-{method}"
    df = pd.read_excel(file_path, sheet_name=sheetname)
    mu_total = df.iloc[:, -1].values
    mu_plus = np.max(mu_total) # Replace with the actual mu_plus value for RUN-n-PROPOSED
    print(f"Using mu_plus = {mu_plus} for RUN-{run_num_hist}-{method}")

    ei_vals = expected_improvement(mu_cand, var_cand, mu_plus, jitter=1e-9)
    sorted_indices = np.argsort(ei_vals)[::-1]

    print("Candidates sorted by EI (descending):")
    for i in sorted_indices:
        print(f"Candidate {i+1}: EI = {ei_vals[i]:.4f}")

    plot_ei(ei_vals, path_af, savefig)

    # Hypervolume Contribution
    excel_path = f"{path_data}\\RUN-{run_num}-{method}\\pareto_front.xlsx"
    df = pd.read_excel(excel_path, sheet_name="Sheet1")      # Choose the correct epoch num. R-Epoch-1, R-Epoch-1-2, R-Epoch-2-2, R-Epoch-3

    # 强制转换为float，并去除NaN行
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
    df["Std"] = pd.to_numeric(df["Std"], errors="coerce")
    df = df.dropna(subset=["Mean", "Std"])

    df["Num"] = df.index + 1  # 添加一个序号列

    ref_point = (df["Mean"].min() - 1.0, df["Std"].min() - 1.0)
    points = df[["Mean", "Std"]].values
    total_hv = hv_2d(points, ref_point)
    contribs = [max(total_hv - hv_2d(points[np.arange(len(df)) != i], ref_point), 0) for i in range(len(df))]
    df["HV_Contribution"] = contribs
    df_sorted = df.sort_values("HV_Contribution", ascending=False).reset_index(drop=True)
    print(df_sorted[["Num", "Mean", "Std", "HV_Contribution"]].head(5).to_string(index=False))
    plot_hv_contrib(df['Num'].values, df["HV_Contribution"].values, path_af, savefig)
    plot_ei_and_hv(ei_vals, df["HV_Contribution"].values, path_af, savefig)

    # 根据分配结果，来保存EI和HV, {这是我们提出的Adaptive acquisition function的方法}
    savepath, sheetname = save_top_candidates(dim, pareto_front, ei_vals, df, path_data_run, run_num, method, total, k)

    return savepath, sheetname

if __name__ == "__main__":
    run_acquisition_function()