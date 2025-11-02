#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   mulCriDecMak.py
# Time    :   2025/08/08 12:30:40
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Description: This script implements a multi-criteria decision-making process.

import pandas as pd
import numpy as np
from tqdm import tqdm
from physics_score import physics_score_from_expr

# === CONFIG: 指标权重（可修改）===
ALPHA = {
    'loss': 0.15,
    'complexity': 0.15,
    'coverage': 0.25,
    'physics': 0.10,
    'stability': 0.15,
    'generalization': 0.10,
    'interpretability': 0.10
}

# === STEP 1: 加载数据 ===
num = 8
file_path = f"Symbolic Regression/srloop/data/hall_of_fame_run-{num}_restored.csv"
df = pd.read_csv(file_path)

# ✅ 修改为你的表达式列名，默认使用 'restored_equation'
expression_column = 'restored_equation'  # 如无此列，可改为 'equation' 或 'sympy_format'

# === STEP 2: 归一化指标 ===
df['loss_norm'] = (df['loss'] - df['loss'].min()) / (df['loss'].max() - df['loss'].min())
df['complexity_norm'] = (df['complexity'] - df['complexity'].min()) / (df['complexity'].max() - df['complexity'].min())

# === STEP 3: 计算 coverage（包含配方变量的程度）===
def compute_coverage(expr, keywords=['C_pol', 'C_eth', 'C_pg', 't']):
    return sum([kw in str(expr) for kw in keywords]) / len(keywords)

df['coverage_score'] = df[expression_column].apply(compute_coverage)

# === STEP 4: 物理约束得分（可扩展）===
# TODO: 实现如单调性、边界条件等具体物理检查
# df['physics_score'] = 0.5  # 目前为 placeholder，可统一赋值或通过函数打分
def compute_physics_scores_for_df(df, expr_col='restored_equation'):
    scores = []
    details = []
    for expr in tqdm(df[expr_col].astype(str).values):
        score, det = physics_score_from_expr(expr)
        scores.append(score)
        details.append(det)
    df['physics_score'] = scores
    df['physics_detail'] = details  # optional, for debugging
    return df

df = compute_physics_scores_for_df(df, expr_col=expression_column)

# === STEP 5: 稳定性指标 ===
# 假设已有稳定性分数列 'score'（结构出现频率）TODO: 确定一下这个‘score’具体指的是什么
df['stability_score'] = df['score'] if 'score' in df.columns else 0.5

# === STEP 6: 泛化性与可解释性占位符（可补充）===
df['generalization_score'] = 0.5  # placeholder
df['interpretability_score'] = 0.5  # placeholder

# === STEP 7: 计算最终多指标总得分 ===
df['total_score'] = (
    ALPHA['loss'] * (1 - df['loss_norm']) +
    ALPHA['complexity'] * (1 - df['complexity_norm']) +
    ALPHA['coverage'] * df['coverage_score'] +
    ALPHA['physics'] * df['physics_score'] +
    ALPHA['stability'] * df['stability_score'] +
    ALPHA['generalization'] * df['generalization_score'] +
    ALPHA['interpretability'] * df['interpretability_score']
)

# === STEP 8: 排序并查看前几个结果 ===
df_sorted = df.sort_values(by='total_score', ascending=False).reset_index(drop=True)

# 显示所有模型
print("\n📌 All expressions after multi-criteria selection:\n")
print(df_sorted[[expression_column, 'total_score', 'loss', 'complexity', 'coverage_score']])

# 显示coverage_score为1的模型
print("\n📌 All expressions with coverage_score = 1:\n")
print(df_sorted[df_sorted['coverage_score'] == 1][[expression_column, 'total_score', 'loss', 'complexity']])

# === 可选：保存排序后的结果 ===
df_sorted.to_csv("Symbolic Regression/srloop/data/sorted_model_selection.csv", index=False)

# 保存coverage_score为1的模型（不保留sympy_format, lambda_format, equation项）
cols_to_exclude = ['sympy_format', 'lambda_format', 'equation']
cols_to_save = [col for col in df_sorted.columns if col not in cols_to_exclude]
df_sorted[df_sorted['coverage_score'] == 1][cols_to_save].to_csv(
    "Symbolic Regression/srloop/data/sorted_model_selection_coverage_1.csv", index=False
)

print("\n✅ Multi-criteria decision-making completed and results saved.")

# plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.barh(df_sorted[expression_column], df_sorted['total_score'], color='skyblue')
plt.xlabel('Total Score')
plt.title('Multi-Criteria Decision-Making Results')
plt.grid(axis='x')
plt.show()