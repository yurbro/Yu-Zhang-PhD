#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   sr_FeEng.py
# Time    :   2025/08/06 11:10:09
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

__all__ = [
    'feature_selection',
    'load_ivpt_data',
    'build_feature_matrix',
    'feature_names_default'
]

# === Feature name definitions (corresponding to feature construction order) ===
feature_names_default = [
    "C_pol", "C_eth", "C_pg", "t",
    "sqrt(t)", "log(t)", "1/t", "exp(-t)", "1-exp(-t)",   # time-related features
    "C_pg * C_eth", "C_pg * C_pol", "C_eth * C_pol", "C_pg / C_pol", "C_pg / C_eth", "C_eth / C_pol", "C_pg^2", "C_eth^2", "C_pol^2",     # interaction and polynomial features
    "C_pg * log(t)", "C_pg * log(t)", "C_eth * log(t)", "C_pg * log(t)", "C_eth * t", "C_pg * t"                # time * excipient interaction features
]

# === Load formulation and permeation data ===
def load_ivpt_data(file_path, x_sheet='Formulas', y_sheet='C'):
    """
    Load IVPT Excel data. Returns:
    df_X: formulation variables (without RunOrder)
    df_Y: permeation data matrix (time points)
    """
    df_X = pd.read_excel(file_path, sheet_name=x_sheet)
    df_Y = pd.read_excel(file_path, sheet_name=y_sheet)
    df_X = df_X.iloc[:, 1:]  # drop RunOrder
    df_Y = df_Y.iloc[:, 1:]
    return df_X, df_Y

# === Feature construction ===
def build_feature_matrix(df_X, time_points, feature_names, feature_matrix_updated, run, n_febank):
    """
    Construct extended feature matrix based on physical principles.
    If new_feature_bank is provided (DataFrame with columns ['Original Structure', 'Feature Expression']),
    add new features not already in feature_names_default.
    Returns:
        - feature_matrix: full input matrix
        - X_repeat: repeated formulation matrix
        - T: time matrix
        - feature_names: updated feature name list
    """
    # 原始变量
    X_raw = df_X[['Poloxamer 407', 'Ethanol', 'Propylene glycol']].values
    X_repeat = np.repeat(X_raw, len(time_points), axis=0)
    T = np.tile(time_points, X_raw.shape[0]).reshape(-1, 1)

    # 命名变量
    C_pol = X_repeat[:, 0:1]
    C_eth = X_repeat[:, 1:2]
    C_pg = X_repeat[:, 2:3]
    t = T

    # # 基础特征
    # feature_matrix = np.hstack([
    #     C_pol, C_eth, C_pg, t,
    #     np.sqrt(t), np.log1p(t - 1), 1 / (t + 1e-6), np.exp(-t), 1 - np.exp(-t),
    #     C_pg * C_eth, C_pg * C_pol, C_eth * C_pol, C_pg / (C_pol + 1e-6), C_pg / (C_eth + 1e-6), C_eth / (C_pol + 1e-6), C_pg**2, C_eth**2, C_pol**2,
    #     C_pg * np.log1p(t - 1), C_eth * np.log1p(t - 1), C_pol * np.log1p(t - 1), C_eth * t, C_pg * t, C_pol * t
    # ])

    # 兼容 feature_names 传入的自定义特征名
    if feature_names is None:
        feature_names = feature_names_default.copy()

    # 如果有新特征库，则添加新特征, 如果没有新特征库则跳过
    if n_febank:
        try:
            new_feature_bank = pd.read_csv(f"Symbolic Regression/srloop/data/new_feature_bank_run-{run-1}.csv")
            for idx, row in new_feature_bank.iterrows():
                expr = row['Feature Expression']
                name = row['Original Structure']
                local_dict = {
                    'C_pol': C_pol,
                    'C_eth': C_eth,
                    'C_pg': C_pg,
                    't': t,
                    'np': np,
                    'log': np.log,
                    'log1p': np.log1p,
                    'sqrt': np.sqrt,
                    'exp': np.exp,
                    'inv': lambda x: 1/(x+1e-6)
                }
                try:
                    new_feat = eval(expr, {"__builtins__": {}}, local_dict)
                    if new_feat.ndim == 1:
                        new_feat = new_feat.reshape(-1, 1)
                    feature_matrix_updated = np.hstack([feature_matrix_updated, new_feat])
                    feature_names.append(str(expr))
                except Exception as e:
                    print(f"[WARN] Failed to evaluate new feature '{name}': {expr} ({e})")
            print(f"[INFO] All new features from new_feature_bank_run-{run-1}.csv have been added.")
        except Exception as e:
            print(f"[WARN] Could not read new_feature_bank_run-{run-1}.csv: {e}")

    # 保证特征名和特征数一致
    assert feature_matrix_updated.shape[1] == len(feature_names), f"feature_matrix.shape[1]={feature_matrix_updated.shape[1]}, len(feature_names)={len(feature_names)}. Feature names and matrix columns must match!"
    
    # save corresponding variable index mapping for run
    pd.DataFrame({
        "Variable": feature_names,
        "Index": [f"x{i}" for i in range(len(feature_names))]
    }).to_csv(f"Symbolic Regression/srloop/data/raw_feature_index_mapping_run-{run}.csv", index=False)

    return feature_matrix_updated, X_repeat, T, feature_names

# === Feature selection using MI and Lasso ===
def feature_selection(feature_matrix, y, run, feature_names, plot=False):
    
    if feature_names is None:
        feature_names = feature_names_default.copy()

    # Mutual Information
    mi = mutual_info_regression(feature_matrix, y)
    mi_series = pd.Series(mi, index=feature_names)  # 使用默认特征名
    if plot:
        mi_series.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5), title="Mutual Information per Feature")
        plt.ylabel("MI Score")
        plt.tight_layout()
        plt.savefig(fr"Symbolic Regression\srloop\visualisation\feature_importance_mi_run-{run}.png")
        plt.close()
        # plt.show()

    # LassoCV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    model = LassoCV(cv=5).fit(X_scaled, y)
    importance = np.abs(model.coef_)
    lasso_series = pd.Series(importance, index=feature_names)
    if plot:
        lasso_series.sort_values(ascending=False).plot(kind='bar', title="Feature Importance from LassoCV")
        plt.xlabel("Features", rotation=45, ha='right')
        plt.ylabel("Importance", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(fr"Symbolic Regression\srloop\visualisation\feature_importance_lasso_run-{run}.png")
        plt.close()
        # plt.show()

    # === 分层保留策略 ===
    core_features = ["C_pol", "C_eth", "C_pg", "t"]  # 原始物理变量
    mi_selected = mi_series[mi_series > 0.01].index.tolist()
    lasso_selected = lasso_series[lasso_series > 0.01].index.tolist()

    # 合并特征（去重）
    selected_features = list(set(core_features + mi_selected + lasso_selected))

    print("\n[INFO] Final selected features for SR modeling:")
    for feat in selected_features:
        print(f" - {feat}")

    return mi_series, lasso_series, selected_features
