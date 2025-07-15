#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   ave_imp_calc.py
# Time    :   2025/06/13 19:56:47
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# This code is used to calculate the average improvement of a benchmark function across multiple iterations

import pandas as pd
import numpy as np

def calculate_improvement(file_path_save, benchmark, method, iter):
    df = pd.read_excel(file_path_save, sheet_name=method)

    results = []
    for i in range(1, iter + 1):
        df_iter = df[f'iter-{i}'].values
        # print(f"Processing iteration {i} for benchmark {benchmark}...: {df_iter}")
        df_iter = pd.DataFrame(df_iter, columns=[f'iter-{i}'])
        best_value = pd.read_excel(file_path_save, sheet_name='IncumbentBest')
        best_value = best_value['Inc_best'].values[i-1]
        # df_imp = (df_iter - best_value) / abs(best_value) * 100
        
        # 选择df_iter中最大的值
        df_iter_max = df_iter.max().values[0]
        # print(f"Best value for iteration {i}: {best_value}, Max value in df_iter: {df_iter_max}")
        df_imp = df_iter_max - best_value  # 计算改进值

        # # average improvement calculation
        # df_imp = df_iter.values - best_value  # 计算改进值
        # df_imp = pd.DataFrame(df_imp, columns=[f'iter-{i}'])


        if df_imp < 0:
            df_imp = 0  # 如果改进值小于0，则设置为0

        # print(f"Iteration {i} improvement: {df_imp}")
        avg_improvement = np.mean(df_imp)
        print(f"Average improvement for iteration {i}: {avg_improvement}")
        results.append({'Iteration': i, 'Average Improvement': avg_improvement})

    improvement_df = pd.DataFrame(results)
    with pd.ExcelWriter(file_path_save, mode='a', if_sheet_exists='replace') as writer:
        improvement_df.to_excel(writer, sheet_name=f'{benchmark}_improvement', index=False)

    print(f"Average improvement for {benchmark} saved to {file_path_save} in sheet '{benchmark}_improvement'.")

if __name__ == "__main__":

    path = "Multi-Objective Optimisation\Benchmark\Package Module-III-sphere\Dataset"
    # method = "PROPOSED"  # or "PROPOSED", depending on the method
    iter = 30
    dim = 5  # 维度
    alpha = 0.5
    benchmark = "Ackley"  # or "Zakharov", depending on the benchmark function
    # file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-sphere\Ave_Improve\\{method}_improvement_analysis_a-{alpha}.xlsx"
    # file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-sphere\Ave_Improve\\{method}_improvement_analysis_a-{alpha}-{dim}D.xlsx"
    # file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-sphere\Ave_Improve\\{method}_improvement_analysis.xlsx"

    # calc BO_3D results

    method = f"BO_POI_{dim}D"  # Specify the method for which you want to calculate the improvement
    file_path_save = f"Multi-Objective Optimisation\Benchmark\Package Module-III-sphere\Ave_Improve\\{method}_improvement_analysis.xlsx"   # calc BO_3D results

    # Run the improvement calculation
    calculate_improvement(file_path_save, benchmark, method, iter)