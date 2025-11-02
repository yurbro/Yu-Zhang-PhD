#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   restore_expr.py
# Time    :   2025/08/07 19:51:44
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

# Description: Restore symbolic expressions from the results

import pandas as pd
import re
import sympy as sp
from typing import Optional

def restore_expr(expr: str, mapping: dict) -> str:
    """
    将表达式中的x0,x1等变量替换为原始表达式
    """
    def repl(m):
        return f'({mapping[m.group(0)]})'
    return re.sub(r'x\d+', repl, expr)

def restore_equations(
    fame_csv_path: str,
    mapping_csv_path: str,
    output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    还原符号回归表达式中的变量为原始表达式。
    fame_csv_path: hall_of_fame_run-5.csv 路径
    mapping_csv_path: variable_index_mapping_run-5.csv 路径
    output_csv_path: 可选，保存还原后结果的路径
    返回: 包含还原表达式的新DataFrame
    """
    mapping_df = pd.read_csv(mapping_csv_path)
    index_to_var = {f'x{row.Index}': row.Variable for _, row in mapping_df.iterrows()}
    fame_df = pd.read_csv(fame_csv_path)
    fame_df['restored_equation'] = fame_df['equation'].apply(lambda x: restore_expr(x, index_to_var))
    if output_csv_path:
        fame_df.to_csv(output_csv_path, index=False)
    return fame_df


def check_features_in_expressions(df: pd.DataFrame, features=None, print_summary=True):
    """
    检查每个表达式是否包含指定原始特征，并可打印结果。
    df: 包含'restored_equation'列的DataFrame
    features: 要检查的特征列表，默认为["C_pol", "C_eth", "C_pg", "t"]
    print_summary: 是否打印包含情况和统计
    返回：每行包含特征情况的列表，以及全部包含的表达式DataFrame
    """
    if features is None:
        features = ["C_pol", "C_eth", "C_pg", "t"]
    contains_list = []
    if print_summary:
        print("\nthe features contained in each expression:")
    for idx, row in df.iterrows():
        expr = row['restored_equation']
        contains = {feat: (feat in expr) for feat in features}
        contains_list.append(contains)
        if print_summary:
            print(f"Expression {idx+1}: {contains}")
    all_features = df[df['restored_equation'].apply(lambda x: all(feat in x for feat in features))]
    if print_summary:
        print(f"\n包含所有特征的表达式数量: {all_features.shape[0]}")
    return contains_list, all_features

def filter_positive_derivative(df, expr_col='restored_equation', var='t', print_summary=True, save_path=None):
    """
    对DataFrame中表达式对指定变量求导, 筛选导数大于0的表达式。
    df: DataFrame, 包含表达式列
    expr_col: 表达式所在列名
    var: 对哪个变量求导
    print_summary: 是否打印筛选结果
    save_path: 可选, 保存筛选结果的csv路径
    返回: 筛选后的DataFrame
    """
    t = sp.symbols(var)
    filtered_rows = []
    for idx, row in df.iterrows():
        expr_str = row[expr_col]
        try:
            expr = sp.sympify(expr_str)
            deriv = sp.diff(expr, t)
            # 判断dQ/dt是否恒大于0（这里只做符号正负判断，复杂表达式可用lambdify数值采样）
            # 这里用subs采样法，采样t=1,2,5,10,20
            test_vals = [1, 2, 5, 10, 20]
            always_positive = all(deriv.subs(t, v).evalf() > 0 for v in test_vals)
        except Exception as e:
            always_positive = False
        if always_positive:
            filtered_rows.append({
                'expression_num': idx+1,
                'complexity': row.get('complexity', None),
                'loss': row.get('loss', None),
                'restored_equation': expr_str
            })
    result_df = pd.DataFrame(filtered_rows)
    if print_summary:
        print("\n满足 dQ/dt > 0 的表达式：")
        for _, r in result_df.iterrows():
            print(f"Expression {r['expression_num']}: complexity={r['complexity']}, loss={r['loss']}, expr={r['restored_equation']}")
    if save_path:
        result_df.to_csv(save_path, index=False)
        if print_summary:
            print(f"已保存筛选结果到: {save_path}")
    return result_df

# 示例用法（可删除或注释）
if __name__ == "__main__":
    
    num = 6
    fame_csv = f'Symbolic Regression/srloop/data/hall_of_fame_run-{num}.csv'
    mapping_csv = f'Symbolic Regression/srloop/data/variable_index_mapping_run-{num}.csv'
    output_csv = f'Symbolic Regression/srloop/data/hall_of_fame_run-{num}_restored.csv'
    df = restore_equations(fame_csv, mapping_csv, output_csv)
    print(f'已生成还原表达式文件：{output_csv}')

    # 检查特征包含情况
    contains_list, all_features = check_features_in_expressions(df)
    print(f"包含所有特征的表达式是：")
    for idx, row in all_features.iterrows():
        print(f"Expression {idx+1}: {row['restored_equation']}")

    # 对这些表达式求导并筛选dQ/dt>0
    save_filtered = f'Symbolic Regression/srloop/data/hall_of_fame_run-{num}_positive_deriv.csv'
    filter_positive_derivative(all_features, expr_col='restored_equation', var='t', print_summary=True, save_path=save_filtered)
    