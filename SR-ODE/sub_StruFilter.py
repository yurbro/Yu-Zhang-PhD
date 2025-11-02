#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   sub_StruFilter.py
# Time    :   2025/08/06 16:25:45
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Desc    :   Substructure extraction and feature bank for symbolic regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

def extract_substructures(equation):
    """
    粗略地提取括号内的结构表达式作为子结构，
    也提取如 sqrt(x8), inv(x13 + 5.1) 等模式。
    """
    substructures = []

    # 提取括号结构
    parens = re.findall(r"\(([^()]+)\)", equation)
    substructures.extend(parens)

    # 提取函数结构，如 sqrt(...) / inv(...) / log(...) 等
    funcs = re.findall(r"(sqrt|inv|log|exp)\(([^()]+)\)", equation)
    substructures.extend([f"{f}({s})" for f, s in funcs])

    return substructures

# TODO: 补全子结构获取的功能实现，包括获取Pareto front得到的表达式以及对应的复杂度和损失

def statistics_for_structure_frequency(df):
    """
    统计每个子结构在所有表达式中的出现频率
    """
    structure_counter = Counter()
    for eq in df["equation"]:
        subs = extract_substructures(eq)
        structure_counter.update(subs)

    # 输出频率大于1的子结构
    top_structures = [(s, freq) for s, freq in structure_counter.items() if freq > 1]
    top_structures = sorted(top_structures, key=lambda x: x[1], reverse=True)
    print("Top substructures by frequency:")
    for s, freq in top_structures:
        print(f"{s:40} : {freq}")

    return top_structures

def build_new_feature(top_structures, run, n_febank):
    """
    根据结构频率构建新的特征
    """
    # read the variable index to name mapping
    var_map_df = pd.read_csv(f"Symbolic Regression/srloop/data/variable_index_mapping_run-{run}.csv")
    var_index_to_name = dict(zip(var_map_df["Index"], var_map_df["Variable"]))

    new_feature_bank = {}
    for s, _ in top_structures:
        # 替换变量索引为实际变量名
        feature_expr = replace_var_indices(s, var_index_to_name)
        new_feature_bank[s] = feature_expr  # s为原结构，feature_expr为变量名表达式

    # 打印新特征表达式
    for orig, expr in new_feature_bank.items():
        print(f"{orig:40} -> {expr}")

    # 如果n_febank为True，则表示有新特征库
    print("New feature bank created with", len(new_feature_bank), "features.")

    # 判断新特征是否与已有特征重复
    # 读取已有特征
    existing_features = pd.read_csv(f"Symbolic Regression/srloop/data/raw_feature_index_mapping_run-{run}.csv")
    # 比较新特征与已有特征
    for orig, expr in new_feature_bank.items():
        if expr in existing_features['Variable'].values:
            print(f"[WARN] Feature '{expr}' already exists in existing features, skipping.")
            continue
        else:
            print(f"[INFO] New feature '{expr}' is unique, adding to feature bank.")

    # 仅保存新特征库
    new_feature_bank = {k: v for k, v in new_feature_bank.items() if v not in existing_features['Variable'].values}
    has_new_features = bool(new_feature_bank)
    if has_new_features:
        n_febank = True
        new_feature_df = pd.DataFrame(list(new_feature_bank.items()), columns=["Original Structure", "Feature Expression"])
        new_feature_df.to_csv(f"Symbolic Regression/srloop/data/new_feature_bank_run-{run}.csv", index=False)
        print(f"New feature bank saved to 'Symbolic Regression/srloop/data/new_feature_bank_run-{run}.csv'.")
    else:
        n_febank = False
        print("No new unique features to add to the feature bank.")

    return new_feature_bank, n_febank

def replace_var_indices(expr, var_map):
    # 替换表达式中的 xN 为实际变量名
    def repl(m):
        idx = m.group(1)
        return var_map.get(int(idx), f"x{idx}")
    return re.sub(r"x(\d+)", repl, expr)