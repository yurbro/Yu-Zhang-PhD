#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   str_feExtra.py
# Time    :   2025/08/08 17:45:28
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Dec: This script implements feature extraction from symbolic expressions.

import pandas as pd
import re

# === Step 1: 读取CSV文件 ===
df = pd.read_csv("Symbolic Regression\srloop\data\hall_of_fame_run-5_restored.csv")

# === Step 2: 定义核心配方变量 ===
core_vars = ["C_pg", "C_eth", "C_pol", "t"]

# === Step 3: 定义表达式结构特征提取函数 ===
def extract_expression_features(expr):
    op_counts = {
        "+": expr.count("+"),
        "-": expr.count("-"),
        "*": expr.count("*"),
        "/": expr.count("/"),
        "exp": expr.count("exp"),
        "sqrt": expr.count("sqrt"),
        "log": expr.count("log"),
        "inv": expr.count("inv"),
        "pow": expr.count("pow"),
    }
    max_depth = max([len(re.findall(r"\(", expr[:i])) for i in range(len(expr))]) if "(" in expr else 1
    var_inclusion = {f"has_{v}": int(v in expr) for v in core_vars}
    return {
        "depth": max_depth,
        "num_ops": sum(op_counts.values()),
        **op_counts,
        **var_inclusion
    }

# === Step 4: 应用表达式特征提取函数（使用‘restored_equation’列）===
feature_df = df["restored_equation"].apply(extract_expression_features)
feature_df = pd.DataFrame(feature_df.tolist())

# === Step 5: 合并特征结果与原始数据 ===
combined_df = pd.concat([df, feature_df], axis=1)

# === Step 6: 增加 coverage 标志 ===
combined_df["coverage"] = combined_df[[f"has_{v}" for v in core_vars]].sum(axis=1) == len(core_vars)

# === Step 7: 保存结构化结果 ===
combined_df.to_csv("Symbolic Regression/srloop/self_supervised_scoring/data/structured_expression_features.csv", index=False)
