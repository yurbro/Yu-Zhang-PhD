#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   subStruFilterCom.py
# Time    :   2025/08/12 16:32:11
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import re
import pandas as pd
import sympy as sp
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

def calc_complexity(expr_str, mode="count_ops"):  #TODO: 应该选node_count
    """
    计算表达式复杂度
    mode:
        'count_ops'  - 按运算符数量计算（推荐，和 PySR 一致）TODO: 可能需要调整. PySR其实是按节点数计算的
        'node_count' - 按表达式节点总数计算
    """
    try:
        expr = sp.sympify(expr_str)
    except Exception:
        return 999  # 无法解析直接判为高复杂度

    if mode == "count_ops":
        return sp.count_ops(expr, visual=False)
    elif mode == "node_count":
        return sum(1 for _ in sp.preorder_traversal(expr))
    else:
        raise ValueError("mode must be 'count_ops' or 'node_count'")

def statistics_for_structure_frequency(df, complexity_mode="count_ops"):
    """统计每个子结构在所有表达式中的出现频率"""
    structure_counter = Counter()
    for eq in df["equation"]:
        subs = extract_substructures(eq)
        structure_counter.update(subs)

    top_structures = [(s, freq) for s, freq in structure_counter.items() if freq > 1]
    top_structures = sorted(top_structures, key=lambda x: x[1], reverse=True)

    print("Top substructures by frequency:")
    for s, freq in top_structures:
        comp = calc_complexity(s, mode=complexity_mode)
        print(f"{s:40} | freq={freq:3d} | complexity={comp}")

    return top_structures

def build_new_feature(top_structures, run, n_febank, max_complexity=6, complexity_mode="count_ops"):  # TODO: 按照目前的结果，这里应该选的是8
    """
    根据结构频率构建新的特征，增加复杂度(limitation=6, 控制子结构不会无限复杂化)筛选
    """
    # 读取变量索引映射
    var_map_df = pd.read_csv(f"Symbolic Regression/srloop/data/variable_index_mapping_run-{run}.csv")
    var_index_to_name = dict(zip(var_map_df["Index"], var_map_df["Variable"]))

    new_feature_bank = {}
    for s, _ in top_structures:
        feature_expr = replace_var_indices(s, var_index_to_name)

        # 复杂度筛选
        comp = calc_complexity(feature_expr, mode=complexity_mode)
        if comp > max_complexity:
            print(f"[SKIP:Complexity>{max_complexity}] {feature_expr} (complexity={comp})")
            continue
        print(f"[ADD] {feature_expr} (complexity={comp})")

        new_feature_bank[s] = feature_expr

    print(f"✅ Passed filtering: {len(new_feature_bank)} features.")

    # 检查唯一性
    existing_features = pd.read_csv(f"Symbolic Regression/srloop/data/raw_feature_index_mapping_run-{run}.csv")
    unique_new_feature_bank = {
        k: v for k, v in new_feature_bank.items()
        if v not in existing_features['Variable'].values
    }

    # 计算unique_new_feature_bank的复杂度
    unique_complexities = {k: calc_complexity(v, mode=complexity_mode) for k, v in unique_new_feature_bank.items()}
    max_unique_comp = max(unique_complexities.values()) if unique_complexities else 0

    if unique_new_feature_bank:
        n_febank = True
        new_feature_df = pd.DataFrame(list(unique_new_feature_bank.items()),
                                      columns=["Original Structure", "Feature Expression"])
        new_feature_df.to_csv(f"Symbolic Regression/srloop/data/new_feature_bank_run-{run}.csv", index=False)
        print(f"💾 Saved {len(unique_new_feature_bank)} features to new_feature_bank_run-{run}.csv")
    else:
        n_febank = False
        print("⚠ No new unique features to add.")
    

    return unique_new_feature_bank, n_febank, max_unique_comp

def replace_var_indices(expr, var_map):
    """替换表达式中的 xN 为实际变量名"""
    def repl(m):
        idx = m.group(1)
        return var_map.get(int(idx), f"x{idx}")
    return re.sub(r"x(\d+)", repl, expr)

if __name__ == "__main__":

    # Test the functions here
    test_equation = "inv(C_pg - 10.475055) * C_pg / C_eth * C_pg / C_eth * 1.3445362"
    print("Extracted substructures:", extract_substructures(test_equation))
    print("Complexity of test equation:", calc_complexity(test_equation))

    # Create a dummy DataFrame for testing
    df_test = pd.DataFrame({
        "equation": [
            "sqrt(x1 + x2) * inv(x3)",
            "log(x1) + exp(x2)",
            "x1 * x2 + x3"
        ]
    })
    print("Statistics for structure frequency:")
    statistics_for_structure_frequency(df_test)
