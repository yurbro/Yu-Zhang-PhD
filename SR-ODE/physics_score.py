#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   physics_score.py
# Time    :   2025/08/08 20:35:31
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk

import pandas as pd
import numpy as np
import sympy as sp

def physics_score_from_expr(expr_str,
                            expr_var_names=('C_pol','C_eth','C_pg','t'),
                            param_ranges=None,
                            n_param_samples=6,
                            t_min=0.0, t_max=28.0, n_t=100,
                            tol_Q0=1e-4,
                            tol_dQdt=1e-6,
                            allow_negative_fraction=0.01,
                            verbose=False):
    """
    对单个表达式字符串计算 physics_score（0..1）。
    逻辑：
      - 在若干代表性配方组合上（角点+随机），计算 Q(t) 在 t 网格上的数值；
      - 检查 Q(0) 是否接近 0；
      - 检查 dQ/dt 是否基本 >= 0（允许小量负值 due to numeric noise）；
      - 检查 Q(t) 是否有大量负值或 NaN/Inf；
      - 将这些结果汇总成 0..1 的 score。

    参数:
      expr_str: str, 表达式（例如："(t - sqrt(t)) * ( ... )"）
      expr_var_names: tuple/list, 变量顺序，必须包含 't'
      param_ranges: dict, {'C_pol': (low,high), ...} 若为 None 使用默认建议范围
      n_param_samples: int, 每种配方抽样数量（包含角点）
      t_min, t_max, n_t: 时间网格参数
      tol_Q0: 允许 Q(0) 与 0 的绝对误差（可按具体量级调节）
      tol_dQdt: 允许 dQ/dt 的小负误差（数值噪声）
      allow_negative_fraction: 允许 Q(t)<0 的最大比例（越小越严格）
      verbose: 是否打印调试信息
    返回:
      score: float in [0,1]
      detail: dict 包含各子分数与错误信息（便于 debug）
    """

    # --- 默认参数范围（请根据实际实验调整） ---
    if param_ranges is None:
        param_ranges = {
            'C_pol': (20.0, 30.0),   # % w/w typical range — 请视实际数据调整
            'C_eth': (10.0, 20.0),
            'C_pg' : (10.0, 20.0)
        }

    detail = {'ok': True, 'msg': '', 'Q0_violations': [], 'dQdt_violations': [], 
              'neg_frac': [], 'nan_flag': False}

    # --- 解析表达式并构造数值函数 ---
    try:
        # 确保 sympy 能识别 sqrt, exp, inv 等
        sympy_expr = sp.sympify(expr_str)
    except Exception as e:
        detail['ok'] = False
        detail['msg'] = f"sympify error: {e}"
        return 0.0, detail

    # declare symbols in the same order
    syms = [sp.symbols(v) for v in expr_var_names]
    try:
        f_numeric = sp.lambdify(tuple(syms), sympy_expr, modules=['numpy'])
    except Exception as e:
        detail['ok'] = False
        detail['msg'] = f"lambdify error: {e}"
        return 0.0, detail

    # --- 构造时间网格 ---
    t_grid = np.linspace(t_min, t_max, n_t)

    # --- 构造代表性参数组合：包含角点 + 随机采样 ---
    param_names = [v for v in expr_var_names if v != 't']
    # build corners
    corners = []
    lows = [param_ranges[p][0] for p in param_names]
    highs = [param_ranges[p][1] for p in param_names]
    # generate all corners if not too many, else sample subset
    if len(param_names) <= 3:
        # all 2^k corners
        for bits in range(2 ** len(param_names)):
            vals = []
            for i in range(len(param_names)):
                if (bits >> i) & 1:
                    vals.append(highs[i])
                else:
                    vals.append(lows[i])
            corners.append(vals)
    # random samples to fill up to n_param_samples
    rng = np.random.default_rng(12345)
    while len(corners) < n_param_samples:
        sample = [float(rng.uniform(low, high)) for (low, high) in zip(lows, highs)]
        corners.append(sample)
    param_combos = corners[:n_param_samples]

    # --- 逐个参数组合评估 Q(t) 并检测物理约束 ---
    q0_viol_count = 0
    dQdt_violation_fractions = []
    neg_frac_list = []
    nan_flag = False

    for combo in param_combos:
        # build inputs in same order as expr_var_names
        # each input must be either scalar or array of t_grid
        input_map = {}
        for v in expr_var_names:
            if v == 't':
                input_map['t'] = t_grid
            else:
                # find index in param_names
                idx = param_names.index(v)
                input_map[v] = combo[idx]

        # prepare args in order
        args = tuple(input_map[v] for v in expr_var_names)
        try:
            Q = f_numeric(*args)
            # ensure numpy array
            Q = np.array(Q, dtype=float)
        except Exception as e:
            # evaluation failed
            detail['ok'] = False
            detail['msg'] = f"numeric eval error for combo {combo}: {e}"
            nan_flag = True
            break

        # check nan/inf
        if not np.isfinite(Q).all():
            nan_flag = True
            detail['nan_info'] = f"non-finite values for combo {combo}"
            # treat as strong violation, but continue to measure
            Q = np.nan_to_num(Q, nan=1e12, posinf=1e12, neginf=-1e12)

        # Q(0) check: absolute or relative tolerance
        Q0 = float(np.nan_to_num(Q[0]))
        max_absQ = max(1.0, np.nanmax(np.abs(Q)))
        if abs(Q0) > max(tol_Q0, tol_Q0 * max_absQ):
            q0_viol_count += 1

        # dQ/dt numeric derivative
        dQdt = np.gradient(Q, t_grid)
        # allow small negative due to noise; count fraction of negative entries below -tol_dQdt
        neg_frac = np.mean(dQdt < -tol_dQdt)
        dQdt_violation_fractions.append(neg_frac)

        # fraction of Q negative
        negQ_frac = np.mean(Q < -1e-12)
        neg_frac_list.append(negQ_frac)

    # assemble detail
    detail['Q0_violations'] = q0_viol_count
    detail['dQdt_violation_fractions'] = dQdt_violation_fractions
    detail['neg_frac'] = neg_frac_list
    detail['nan_flag'] = nan_flag

    # --- scoring ---
    # Q0 score: proportion of combos that satisfy Q0 approx 0
    q0_pass_frac = 1.0 - (q0_viol_count / max(1, len(param_combos)))
    # monotonicity score: 1 - mean(violation fraction), clipped
    mono_score = 1.0 - float(np.mean(dQdt_violation_fractions))
    mono_score = float(np.clip(mono_score, 0.0, 1.0))
    # non-negativity score
    nonneg_score = 1.0 - float(np.clip(np.mean(neg_frac_list) / max(allow_negative_fraction, 1e-12), 0.0, 1.0))
    # nan penalty
    if nan_flag:
        nan_penalty = 0.0
    else:
        nan_penalty = 1.0

    # combine with weights (you can tune these)
    w_q0 = 0.25
    w_mono = 0.55
    w_nonneg = 0.15
    w_nan = 0.05

    physics_score = (w_q0 * q0_pass_frac +
                     w_mono * mono_score +
                     w_nonneg * nonneg_score) * nan_penalty

    # clip and return
    physics_score = float(np.clip(physics_score, 0.0, 1.0))
    detail['subscores'] = {'q0_pass_frac': q0_pass_frac,
                           'mono_score': mono_score,
                           'nonneg_score': nonneg_score,
                           'nan_penalty': nan_penalty,
                           'combined': physics_score}
    if verbose:
        print("physics detail:", detail)

    return physics_score, detail

def physics_check_expr(expr_str, 
                       time_points, 
                       param_ranges,
                       tol_Q0=1e-6,
                       tol_pos=1e-6,
                       tol_dQdt=1e-6,
                       allow_neg_frac=0.05,
                       n_param_samples=5):
    """
    对符号回归表达式进行物理约束检查。
    
    参数:
    --------
    expr_str : str
        表达式字符串，例如 "((t - sqrt(t)) * ((14.89 - (-0.33 / (C_pol + -25.26))) - exp(C_pg * C_eth / 238.52)))"
    time_points : list or array
        用于检查的时间点（建议用实验采样时间）
    param_ranges : dict
        参数取值范围，例如：
        {'C_pol': (0, 10), 'C_eth': (0, 50), 'C_pg': (0, 20)}
    tol_Q0 : float
        允许 Q(0) 偏离 0 的容差
    tol_pos : float
        Q(t) 正值的最小阈值
    tol_dQdt : float
        dQ/dt 允许的小负值容差
    allow_neg_frac : float
        允许 Q(t) 或 dQ/dt 违反约束的比例
    n_param_samples : int
        每个变量随机采样的数量（配方组合）
    
    返回:
    --------
    physics_pass : bool
        是否通过物理约束（硬筛）
    physics_score : float
        物理合理性得分（0~1）
    detail : dict
        每个检查项的结果详情
    """
    # --- 默认参数范围（请根据实际实验调整） ---
    if param_ranges is None:
        param_ranges = {
            'C_pol': (20.0, 30.0),   # % w/w typical range — 请视实际数据调整
            'C_eth': (10.0, 20.0),
            'C_pg' : (10.0, 20.0)
        }

    # 符号化变量
    t = sp.Symbol('t')
    C_pol = sp.Symbol('C_pol')
    C_eth = sp.Symbol('C_eth')
    C_pg = sp.Symbol('C_pg')

    try:
        expr_sympy = sp.sympify(expr_str)
    except Exception as e:
        return False, 0.0, {'error': f'无法解析表达式: {e}'}

    # lambdify
    func = sp.lambdify((t, C_pol, C_eth, C_pg), expr_sympy, modules=["numpy"])

    # 生成配方组合（角点 + 随机）
    param_samples = []
    bounds = list(param_ranges.items())
    # 角点
    for vals in [(lo, hi) for lo, hi in [b[1] for b in bounds]]:
        pass  # 可以自己加角点生成逻辑，这里主要用随机采样 TODO: 这里用MC采样？或者其他什么方式进行采样？
    # 随机
    rng = np.random.default_rng(42)
    for _ in range(n_param_samples):
        sample = [rng.uniform(lo, hi) for _, (lo, hi) in bounds]
        param_samples.append(sample)
    param_samples = np.array(param_samples)

    total_checks = 0
    fail_checks = 0

    # 详情记录
    detail = {
        'Q0_fail_count': 0,
        'Qpos_fail_count': 0,
        'dQdt_fail_count': 0,
        'nan_inf_count': 0
    }

    for Cpol_val, Ceth_val, Cpg_val in param_samples:
        try:
            Q_vals = np.array([func(ti, Cpol_val, Ceth_val, Cpg_val) for ti in time_points], dtype=float)
        except Exception:
            detail['nan_inf_count'] += 1
            fail_checks += len(time_points)
            continue

        if np.any(~np.isfinite(Q_vals)):
            detail['nan_inf_count'] += np.sum(~np.isfinite(Q_vals))
            fail_checks += np.sum(~np.isfinite(Q_vals))
            continue

        # Q(0) 检查
        if abs(Q_vals[0]) > tol_Q0:
            detail['Q0_fail_count'] += 1
            fail_checks += 1
        total_checks += 1

        # Q(t) > tol_pos 检查
        neg_mask = Q_vals[1:] < tol_pos
        detail['Qpos_fail_count'] += np.sum(neg_mask)
        fail_checks += np.sum(neg_mask)
        total_checks += len(Q_vals) - 1

        # dQ/dt 检查
        dQdt = np.gradient(Q_vals, time_points)
        dQdt_neg_mask = dQdt < -tol_dQdt
        detail['dQdt_fail_count'] += np.sum(dQdt_neg_mask)
        fail_checks += np.sum(dQdt_neg_mask)
        total_checks += len(dQdt)


    # 计算得分
    physics_score = 1 - (fail_checks / max(total_checks, 1))

    # 硬筛通过条件
    physics_pass = (
        physics_score >= (1 - allow_neg_frac)
        and detail['nan_inf_count'] == 0
    )

    return physics_pass, physics_score, detail

def filter_and_save_physics_valid_models(input_csv, output_csv, expression_column, time_points, param_ranges):
    """
    从输入CSV筛选通过物理约束的表达式并保存
    """
    df = pd.read_csv(input_csv)
    pass_flags = []
    scores = []
    details = []

    for expr in df[expression_column]:
        p_flag, p_score, p_detail = physics_check_expr(
            expr_str=expr,
            time_points=time_points,
            param_ranges=param_ranges
        )
        pass_flags.append(p_flag)
        scores.append(p_score)
        details.append(p_detail)

    df['physics_pass'] = pass_flags
    df['physics_score'] = scores
    df['physics_detail'] = details

    df_pass = df[df['physics_pass'] == True].reset_index(drop=True)
    df_pass.to_csv(output_csv, index=False)

    print(f"✅ 已保存 {len(df_pass)} 个通过物理约束的表达式到 {output_csv}")
    return df_pass

def physics_check_expr_v2(
    expr_str, 
    time_points, 
    param_ranges=None,
    tol_Q0=1e-6,
    tol_pos=1e-6,
    tol_dQdt=1e-6,
    allow_neg_frac=0.05,
    n_param_samples=5,
    key_t_points=[1.0]  # 关键时间点
):
    """
    改进版物理约束检查：
    - 加入关键点检查(Q(t_key) > tol_pos)
    - 参数采样包含随机+角点
    - 保持比例容忍机制，但关键点严格限制
    """
    if param_ranges is None:
        param_ranges = {
            'C_pol': (20.0, 30.0),
            'C_eth': (10.0, 20.0),
            'C_pg': (10.0, 20.0)
        }

    # 符号变量
    t, C_pol, C_eth, C_pg = sp.symbols('t C_pol C_eth C_pg')

    try:
        expr_sympy = sp.sympify(expr_str)
    except Exception as e:
        return False, 0.0, {'error': f'无法解析表达式: {e}'}

    func = sp.lambdify((t, C_pol, C_eth, C_pg), expr_sympy, modules=["numpy"])

    bounds = list(param_ranges.items())
    param_samples = []

    # 角点采样
    from itertools import product
    bound_values = [b for _, b in bounds]  # 取出区间
    for combo in product(*bound_values):
        param_samples.append(combo)

    # 随机采样
    rng = np.random.default_rng(42)
    for _ in range(n_param_samples):
        sample = [rng.uniform(lo, hi) for _, (lo, hi) in bounds]
        param_samples.append(sample)

    param_samples = np.array(param_samples)

    total_checks = 0
    fail_checks = 0
    detail = {'Q0_fail_count': 0, 'Qpos_fail_count': 0, 'dQdt_fail_count': 0, 'nan_inf_count': 0, 'key_t_fail_count': 0}

    for Cpol_val, Ceth_val, Cpg_val in param_samples:
        try:
            Q_vals = np.array([func(ti, Cpol_val, Ceth_val, Cpg_val) for ti in time_points], dtype=float)
        except Exception:
            detail['nan_inf_count'] += 1
            fail_checks += len(time_points)
            continue

        if np.any(~np.isfinite(Q_vals)):
            detail['nan_inf_count'] += np.sum(~np.isfinite(Q_vals))
            fail_checks += np.sum(~np.isfinite(Q_vals))
            continue

        # Q(0) 检查
        if abs(Q_vals[0]) > tol_Q0:
            detail['Q0_fail_count'] += 1
            fail_checks += 1
        total_checks += 1

        # 正值检查
        neg_mask = Q_vals[1:] < tol_pos
        detail['Qpos_fail_count'] += np.sum(neg_mask)
        fail_checks += np.sum(neg_mask)
        total_checks += len(Q_vals) - 1

        # 单调性检查
        dQdt = np.gradient(Q_vals, time_points)
        dQdt_neg_mask = dQdt < -tol_dQdt
        detail['dQdt_fail_count'] += np.sum(dQdt_neg_mask)
        fail_checks += np.sum(dQdt_neg_mask)
        total_checks += len(dQdt)

        # 关键点检查（强约束）
        for t_key in key_t_points:
            if t_key in time_points:
                idx = np.where(time_points == t_key)[0][0]
            else:
                idx = np.argmin(np.abs(time_points - t_key))
            if Q_vals[idx] <= tol_pos:
                detail['key_t_fail_count'] += 1
                return False, 0.0, detail  # 直接 fail

    physics_score = 1 - (fail_checks / max(total_checks, 1))
    physics_pass = (
        physics_score >= (1 - allow_neg_frac)
        and detail['nan_inf_count'] == 0
    )

    return physics_pass, physics_score, detail

# ===== 示例用法 =====
if __name__ == "__main__":
    time_points = [1, 2, 3, 4, 6, 8, 22, 24, 26, 28]
    param_ranges = {
        'C_pol': (20, 30),
        'C_eth': (10, 20),
        'C_pg': (10, 20)
    }

    input_csv = "Symbolic Regression/srloop/data/hall_of_fame_run-5_restored_testcode.csv"
    output_csv = "Symbolic Regression/srloop/data/physics_valid_models_testcode.csv"

    df_valid = filter_and_save_physics_valid_models(
        input_csv=input_csv,
        output_csv=output_csv,
        expression_column='restored_equation',  # 你的表达式列
        time_points=time_points,
        param_ranges=param_ranges
    )

    print(df_valid[[ 'restored_equation', 'physics_score']])