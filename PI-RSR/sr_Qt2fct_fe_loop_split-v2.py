#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   sr_Qt2fct_fe_loop.py
# Time    :   2025/08/04 18:57:18
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Desc    :   Symbolic Regression for IVPT data to be a function of concentration and time with feature expansion and selection Loop

"""
    This code splits the dataset into training and testing sets.
    Update: The physics-informed constraints added into the filtering process of Pareto front.
"""
import pandas as pd
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from icecream import ic
import pandas as pd
from sr_FeEng import build_feature_matrix, feature_selection, feature_names_default
from sub_StruFilter import build_new_feature, statistics_for_structure_frequency
from restore_expr import restore_equations
from physics_score import physics_check_expr, filter_and_save_physics_valid_models

def feature_bank(df_X, y_all, time_points, run, feature_names, feature_matrix_updated, n_febank):
    # === Feature expansion and selection ===
    feature_matrix_updated, X_repeat, T, feature_names_updated = build_feature_matrix(df_X, time_points, feature_names, feature_matrix_updated, run, n_febank)
    # Run feature selection & retain only meaningful ones by using mutual information and LASSO
    mi_series, lasso_series, selected_features = feature_selection(feature_matrix_updated, y_all, run, feature_names_updated, plot=True)

    # 可用于后续 PySR 的列索引
    selected_indices = [feature_names_updated.index(feat) for feat in selected_features]
    final_matrix_for_pysr = feature_matrix_updated[:, selected_indices]  # 仅保留选中的特征

    # 保存特征重要性
    pd.DataFrame({
        "MI": mi_series,
        "Lasso": lasso_series
    }).to_csv(f"Symbolic Regression\\srloop\\data\\feature_scores_run-{run}.csv")

    # 保存最终筛选的变量名
    pd.Series(selected_features).to_csv(f"Symbolic Regression\\srloop\\data\\selected_features_run-{run}.csv", index=False)

    return final_matrix_for_pysr, feature_matrix_updated, selected_features, selected_indices, feature_names_updated

def plot_performance(y_raw, y_pred, run):
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_raw, y_pred, alpha=0.5)
    plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'r--', lw=2)
    plt.xlabel("True Q(t)")
    plt.ylabel("Predicted Q(t)")
    plt.title("Symbolic Regression Prediction vs. True")
    plt.legend(["y = x", "Predictions, R2: {:.2f}".format(r2_score(y_all, y_pred))])
    plt.savefig(fr'Symbolic Regression/srloop/visualisation/sr_Qt2fct_fe_prediction_run-{run}.png')
    # plt.show()

if __name__ == "__main__":

    # === Step 1: 读取数据 ===
    file_path = 'Symbolic Regression\data\Raw_IVPT_thirty.xlsx'
    df_X = pd.read_excel(file_path, sheet_name='Formulas-train')
    df_Y = pd.read_excel(file_path, sheet_name='C-train')  # 每列是一个配方在10个时间点的Q(t)
    df_X = df_X.iloc[:, 1:]  # 去掉RunOrder列
    df_Y = df_Y.iloc[:, 1:]  # 去掉Formulas_C列

    # === Step 2: 数据展开（构建训练集） ===
    time_points = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])
    X_all, y_all = [], []

    for i, (_, row) in enumerate(df_X.iterrows()):
        c_pol, c_eth, c_pg = row[['Poloxamer 407', 'Ethanol', 'Propylene glycol']]
        Q_series = df_Y.iloc[i, :].values

        for t, q in zip(time_points, Q_series):
            X_all.append([c_pol, c_eth, c_pg, t])     # 每个样本包含三个配方成分和时间点
            y_all.append(q)                           # 每个样本的目标值是Q(t)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    param_ranges = {
            'C_pol': (20.0, 30.0),   # % w/w typical range — 请视实际数据调整
            'C_eth': (10.0, 20.0),
            'C_pg' : (10.0, 20.0)
        }

    # === Loop for multiple runs ===
    # Parameters settings
    iter = 10 # 迭代次数
    ir = 1 # count of runs
    feature_names_update = feature_names_default  # 默认特征名称
    n_febank = False  # 标识符来判断是否需要构建新的特征库

    # Initialize the feature bank
    # 原始变量
    X_raw = df_X[['Poloxamer 407', 'Ethanol', 'Propylene glycol']].values
    X_repeat = np.repeat(X_raw, len(time_points), axis=0)
    T = np.tile(time_points, X_raw.shape[0]).reshape(-1, 1)

    # 命名变量
    C_pol = X_repeat[:, 0:1]
    C_eth = X_repeat[:, 1:2]
    C_pg = X_repeat[:, 2:3]
    t = T

    # 基础特征
    feature_matrix = np.hstack([
        C_pol, C_eth, C_pg, t,
        np.sqrt(t), np.log1p(t - 1), 1 / (t + 1e-6), np.exp(-t), 1 - np.exp(-t), t**2,
        C_pg * C_eth, C_pg * C_pol, C_eth * C_pol, C_pg / (C_pol + 1e-6), C_pg / (C_eth + 1e-6), C_eth / (C_pol + 1e-6), C_pg**2, C_eth**2, C_pol**2,
        C_pg * np.log1p(t - 1), C_eth * np.log1p(t - 1), C_pol * np.log1p(t - 1), C_eth * t, C_pg * t, C_pol * t
    ])

    feature_matrix_updated = feature_matrix.copy()  # 用于存储更新后的特征矩阵

    # Loop for symbolic regression runs
    print("\n=== Starting Symbolic Regression Loop ===")
    print(f"Total runs: {iter}")

    for run in range(iter):
        print(f"\n=== Run {run + 1} ===")
        # --- Step Loop-1: Feature expansion and selection ---
        X_selected, feature_matrix, selected_features, selected_indices, feature_names = feature_bank(df_X, y_all, time_points,
                                                                                                   run + 1, feature_names_update, feature_matrix_updated, n_febank)
        print("\nSymbolic Regression input variables: ")
        for i, feat in enumerate(selected_features):
            print(f"x{i}: {feat}")
        
        # Save corresponding selected variable index mapping for run
        mapping_csv = f"Symbolic Regression/srloop/data/variable_index_mapping_run-{run + 1}.csv"
        pd.DataFrame({
            "Variable": selected_features,
            "Index": [i for i in range(len(selected_features))]
        }).to_csv(mapping_csv, index=False)

        # --- Step Loop-2: Fit symbolic regression model ---
        model = PySRRegressor(
            model_selection="best",
            niterations=1000,
            parsimony=0.005,  # 控制模型复杂度
            adaptive_parsimony_scaling=500,  # 自适应简约性缩放
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "exp", "square", "inv"],
            loss="loss(x, y) = (x - y)^2 + 10 * max(0, -x)",  # 加入了惩罚项
            maxsize=30,
            verbosity=1,
        )
        model.fit(X_selected, y_all)

        # --- Step Loop-3: Model evaluation ---
        y_pred = model.predict(X_selected)
        print("R2 score:", r2_score(y_all, y_pred))

        # --- Step Loop-4: Output equations ---
        print("Best symbolic model:")
        print(model)
        fame_csv = f'Symbolic Regression/srloop/data/hall_of_fame_run-{run + 1}.csv'
        hall_of_fame = model.equations_
        hall_of_fame.to_csv(fame_csv, index=False)
        print(f"Hall of fame include: {hall_of_fame}")

        # --- Step Loop-5: Plot performance ---
        plot_performance(y_all, y_pred, run + 1)

        # --- Step Loop-5.1 Physics-informed constraints filtering expression ---
        print("Restoring the pareto front expression ... ")
        output_csv = f'Symbolic Regression/srloop/data/hall_of_fame_run-{run + 1}_restored.csv'
        fame_df_restored = restore_equations(fame_csv, mapping_csv, output_csv)

        print("\nChecking physics-informed constraints...")
        physics_constraints_output_csv = f"Symbolic Regression/srloop/data/physics_filtered_valid_models_run-{run + 1}.csv"

        df_filtered_physics = filter_and_save_physics_valid_models(
            input_csv=output_csv,
            output_csv=physics_constraints_output_csv,
            expression_column='restored_equation',
            time_points=time_points,
            param_ranges=param_ranges
        )
        print(df_filtered_physics[['restored_equation', 'physics_score']])

        # Read the

        # --- Step Loop-6: Extract substructures & build new feature bank ---
        top_structures = statistics_for_structure_frequency(df_filtered_physics) 
        print("\nExtracting substructures from equations...")
        new_feature_bank, n_febank = build_new_feature(top_structures, run + 1, n_febank)
        feature_names_update = feature_names  # 更新特征名称
        feature_matrix_updated = feature_matrix

        # --- Step Loop-8: Justify the feature bank ---
        print("\nJustifying the new feature bank...")

        # Check if new features were added
        if n_febank is False:
            print(f"No new features found. Stopping loop at run {run + 1}.")
            break
        else:
            print(f"--- Continuing loop ---")
            ir += 1

    print("\n=== All runs completed ===")

    # --- Step Loop-10: Finalize ---
    print("Finalizing symbolic regression process...")
    print(f"All runs completed. Check the saved models and feature banks. Total runs: {ir}")
