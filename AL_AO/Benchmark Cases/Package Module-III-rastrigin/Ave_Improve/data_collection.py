#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   data_collection.py
# Time    :   2025/06/13 19:57:16
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# This code is used to collect data from multiple Excel files and calculate the average improvement of a benchmark function across multiple iterations.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# load the data from the file
def load_data(file_path, benchmark):
    # load the sheet names of the Excel file
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    df_list = []
    print(f"Available sheets: {sheet_names}")
    # load the data from the specified sheet and save to a DataFrame
    for sheet in sheet_names:
        print(f"Loading data from sheet: {sheet}")
        df = pd.read_excel(file_path, sheet_name=sheet)
        # Get the columns that match the benchmark function
        df_list.append(df[benchmark].values)

    return df_list


def run_loop(iter, benchmark, method, dim, file_path_save, path):
    all_dfs = []
    for i in range(1, iter + 1):
        file_path = f"{path}\\RUN-{i}-{method}-{dim}D\\Top-RUN{i}-{method}-{dim}D_{benchmark}_result.xlsx"   # PROPOSED
        # file_path = f"{path}\\RUN-{i}-{method}\\Top-RUN{i}-{method}_{benchmark}_result.xlsx"

        data = load_data(file_path, benchmark)
        data = np.concatenate(data).reshape(-1, 1)
        df = pd.DataFrame(data, columns=[f"iter-{i}"])
        all_dfs.append(df)

    # Merge all df into a single DataFrame, with different columns for different iterations
    result_df = pd.concat(all_dfs, axis=1)
    # Save to the same sheet
    with pd.ExcelWriter(file_path_save, engine='openpyxl', mode='w') as writer:
        result_df.to_excel(writer, sheet_name=method, index=False)
    print(f"All iterations saved to {file_path_save} in sheet '{method}'.")


if __name__ == "__main__":

    path = "Multi-Objective Optimisation\Benchmark\Package Module-III-rastrigin\Dataset"
    method = "HV"  # or "PRO", depending on the method
    iter = 30
    dim = 5  # DIMensions, can be 2, 3, 5, or 10
    alpha = 0.5
    benchmark = "Ackley"  # or "Zakharov", depending on the benchmark function
    file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-rastrigin\Ave_Improve\\{method}_improvement_analysis_a-{alpha}-{dim}D.xlsx"  # 5D
    # file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-rastrigin\Ave_Improve\\{method}_improvement_analysis_a-{alpha}.xlsx"
    # file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-rastrigin\Ave_Improve\\{method}_improvement_analysis.xlsx"


    # Run the loop to process data
    run_loop(iter, benchmark, method, dim, file_path_save, path)