#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   rastrigin_func.py
# Time    :   2025/06/20 11:07:55
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# rastrigin_func.py
import numpy as np
from typing import Union
import pandas as pd
import os

def ackley(x: Union[np.ndarray, list]) -> np.ndarray:
    
    x_arr = np.atleast_2d(x).astype(float)
    n, d = x_arr.shape
    A = 10  # 常数
    return A * d + np.sum(x_arr**2 - A * np.cos(2 * np.pi * x_arr), axis=1)

def ackley_max(x: Union[np.ndarray, list]) -> np.ndarray:
    
    return -ackley(x)  # maximization objective, negative of minimization objective

def read_pts_from_excel(dim, method, filepath, selected_sheetname):
    
    # check the method
    cols = [f"X{i+1}" for i in range(dim)]
    if method not in [f'EI-{dim}D', f'HV-{dim}D', f'RANDOM-{dim}D']:
        sheets = selected_sheetname
        pts_list = []
        for sheet in sheets:
            df = pd.read_excel(filepath, sheet_name=sheet)
            # Select X columns matching the dimension
            pts_sheet = df[cols].to_numpy(dtype=float)
            pts_list.append(pts_sheet)
        pts = np.vstack(pts_list)
        return pts

    elif method == f'EI-{dim}D':
        sheetname_ei = selected_sheetname
        df_ei = pd.read_excel(filepath, sheet_name=sheetname_ei)
        pts_ei = df_ei[cols].to_numpy(dtype=float)
        return pts_ei

    elif method == f'HV-{dim}D':
        sheetname_hv = selected_sheetname
        df_hv = pd.read_excel(filepath, sheet_name=sheetname_hv)
        pts_hv = df_hv[cols].to_numpy(dtype=float)
        return pts_hv

    elif method == f'RANDOM-{dim}D':
        sheetname_random = selected_sheetname
        df_random = pd.read_excel(filepath, sheet_name=sheetname_random)
        pts_random = df_random[cols].to_numpy(dtype=float)
        return pts_random

def read_pts_separate_from_excel(dim, method, filepath, selected_sheetname):
    """
    read_pts_separate_from_excel(dim, method, filepath, selected_sheetname) -> tuple:
    Read points from Excel file based on method and dimension.
        pts_ei: np.ndarray, Top3_EI sheet的点
        pts_hv: np.ndarray, Top3_HV sheet的点
    """
    # check the method
    cols = [f"X{i+1}" for i in range(dim)]
    if method not in [f'EI-{dim}D', f'HV-{dim}D', f'RANDOM-{dim}D']:
        sheetname_ei = selected_sheetname[0]
        sheetname_hv = selected_sheetname[1]
        df_ei = pd.read_excel(filepath, sheet_name=sheetname_ei)
        df_hv = pd.read_excel(filepath, sheet_name=sheetname_hv)
        # Select X columns matching the dimension
        pts_ei = df_ei[cols].to_numpy(dtype=float)
        pts_hv = df_hv[cols].to_numpy(dtype=float)
        return pts_ei, pts_hv

    elif method == f'EI-{dim}D':
        sheetname_ei = selected_sheetname
        df_ei = pd.read_excel(filepath, sheet_name=sheetname_ei)
        pts_ei = df_ei[cols].to_numpy(dtype=float)
        return pts_ei

    elif method == f'HV-{dim}D':
        sheetname_hv = selected_sheetname
        df_hv = pd.read_excel(filepath, sheet_name=sheetname_hv)
        pts_hv = df_hv[cols].to_numpy(dtype=float)
        return pts_hv

    elif method == f'RANDOM-{dim}D':
        sheetname_random = selected_sheetname
        df_random = pd.read_excel(filepath, sheet_name=sheetname_random)
        pts_random = df_random[cols].to_numpy(dtype=float)
        return pts_random


def run_ackley(dim, run_num, path, path_data_run, output_path, method, selected_sheetname):
    """
    Run Ackley function optimization.
    input:
        dim: int, dimension of the problem
        run_num: int, run number for saving results
    """
    # generate column names for the DataFrame
    col_names = [f'x{i+1}' for i in range(dim)]

    # Check the method
    if method not in [f'EI-{dim}D', f'HV-{dim}D', f'RANDOM-{dim}D']:
        pts_ei, pts_hv = read_pts_separate_from_excel(dim, method, path, selected_sheetname)
        vals_ei = ackley_max(pts_ei)
        vals_hv = ackley_max(pts_hv)

        df_ei = pd.DataFrame(pts_ei, columns=col_names)
        df_ei['Ackley'] = vals_ei
        df_hv = pd.DataFrame(pts_hv, columns=col_names)
        df_hv['Ackley'] = vals_hv

        output_path_single = os.path.join(path_data_run, f'Top-RUN{run_num}-{method}_Ackley_result.xlsx')
        sheetname_ei = selected_sheetname[0]
        sheetname_hv = selected_sheetname[1]
        with pd.ExcelWriter(output_path_single, engine='openpyxl') as writer:
            df_ei.to_excel(writer, index=False, sheet_name=sheetname_ei)
            df_hv.to_excel(writer, index=False, sheet_name=sheetname_hv)

    elif method == f'EI-{dim}D':
        pts_ei = read_pts_separate_from_excel(dim, method, path, selected_sheetname)
        vals_ei = ackley_max(pts_ei)
        df_method = pd.DataFrame(pts_ei, columns=col_names)
        df_method['Ackley'] = vals_ei
        output_path_single = os.path.join(path_data_run, f'Top-RUN{run_num}-{method}_Ackley_result.xlsx')
        with pd.ExcelWriter(output_path_single, engine='openpyxl') as writer:
            df_method.to_excel(writer, index=False, sheet_name=selected_sheetname)
    elif method == f'HV-{dim}D':
        pts_hv = read_pts_separate_from_excel(dim, method, path, selected_sheetname)
        vals_hv = ackley_max(pts_hv)
        df_method = pd.DataFrame(pts_hv, columns=col_names)
        df_method['Ackley'] = vals_hv
        output_path_single = os.path.join(path_data_run, f'Top-RUN{run_num}-{method}_Ackley_result.xlsx')
        with pd.ExcelWriter(output_path_single, engine='openpyxl') as writer:
            df_method.to_excel(writer, index=False, sheet_name=selected_sheetname)
    elif method == f'RANDOM-{dim}D':
        pts_random = read_pts_separate_from_excel(dim, method, path, selected_sheetname)
        vals_random = ackley_max(pts_random)
        df_method = pd.DataFrame(pts_random, columns=col_names)
        df_method['Ackley'] = vals_random
        output_path_single = os.path.join(path_data_run, f'Top-RUN{run_num}-{method}_Ackley_result.xlsx')
        with pd.ExcelWriter(output_path_single, engine='openpyxl') as writer:
            df_method.to_excel(writer, index=False, sheet_name=selected_sheetname)
    print(f"Results saved to {output_path_single}")

    # Save to the historical dataset
    pts = read_pts_from_excel(dim, method, path, selected_sheetname)
    vals = ackley_max(pts)
    for i, v in enumerate(vals):
        print(f"x={pts[i]} → Ackley={v:.4f}")

    df_result = pd.DataFrame(pts, columns=col_names)
    df_result['Ackley'] = vals

    sheet_name = f'RUN-{run_num}-{method}'
    if not os.path.exists(output_path):
        df_result.to_excel(output_path, index=False, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_result.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"Results saved to {output_path}")