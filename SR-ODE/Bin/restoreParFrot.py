
import pandas as pd
import sympy as sp

# --- 自定义函数（保持与之前一致） ---
class Inv(sp.Function):
    nargs = 1
class Square(sp.Function):
    nargs = 1
class Sqrt(sp.Function):
    nargs = 1

def sympify_with_locals(expr_str: str) -> sp.Expr:
    local_dict = {
        "inv": Inv,
        "square": Square,
        "sqrt": Sqrt,
        "log": sp.log,
        "exp": sp.exp,
    }
    return sp.sympify(expr_str, locals=local_dict)

def _is_unary_minus(term: sp.Expr):
    if term.is_Mul:
        nums = [a for a in term.args if (a.is_Number and a == -1)]
        others = [a for a in term.args if not (a.is_Number and a == -1)]
        if len(nums) == 1 and len(others) == 1:
            return True, others[0]
    return False, term

def pysr_complexity_from_expr(expr: sp.Expr, in_mul: bool = False) -> int:
    if expr.is_Symbol:
        return 1
    if expr.is_Number:
        return 1
    if expr.is_Add:
        terms = list(expr.args)
        op_count = max(len(terms) - 1, 0)
        total = op_count
        for t in terms:
            is_neg, core = _is_unary_minus(t)
            if is_neg:
                total += pysr_complexity_from_expr(core, in_mul=False)
            else:
                total += pysr_complexity_from_expr(t, in_mul=False)
        return total
    if expr.is_Mul:
        args = list(expr.args)
        op_count = max(len(args) - 1, 0)
        total = op_count
        for a in args:
            if a.is_Pow and a.exp == -sp.Integer(1):
                total += pysr_complexity_from_expr(a.base, in_mul=True)
            else:
                total += pysr_complexity_from_expr(a, in_mul=True)
        return total
    if expr.is_Pow:
        base, exp = expr.base, expr.exp
        if exp == sp.Rational(1, 2):
            return 1 + pysr_complexity_from_expr(base, in_mul=False)
        if exp == -sp.Integer(1):
            if in_mul:
                return pysr_complexity_from_expr(base, in_mul=False)
            else:
                return 1 + pysr_complexity_from_expr(base, in_mul=False)
        return 1 + pysr_complexity_from_expr(base, in_mul=False) + pysr_complexity_from_expr(exp, in_mul=False)
    if isinstance(expr, sp.Function):
        return 1 + sum(pysr_complexity_from_expr(a, in_mul=False) for a in expr.args)
    if hasattr(expr, "args"):
        return sum(pysr_complexity_from_expr(a, in_mul=False) for a in expr.args)
    return 0

def recompute_complexity(expr_str: str):
    try:
        expr = sympify_with_locals(expr_str)
    except Exception as e:
        print(f"解析失败: {expr_str}, 错误: {e}")
        return None
    return pysr_complexity_from_expr(expr)

def pareto_front(df, loss_col="loss", comp_col="restored_complexity"):
    points = df[[loss_col, comp_col]].values
    is_efficient = [True] * len(points)
    for i, (loss_i, comp_i) in enumerate(points):
        if not is_efficient[i]:
            continue
        for j, (loss_j, comp_j) in enumerate(points):
            if (loss_j <= loss_i and comp_j <= comp_i) and (loss_j < loss_i or comp_j < comp_i):
                is_efficient[i] = False
                break
    return df[is_efficient]

# === 主流程 ===
for i in range(1, 9):  # 处理run-1,- , run-8
    file_path = fr"Symbolic Regression/srloop/data/hall_of_fame_run-{i}_restored.csv"
    df = pd.read_csv(file_path)

    # 重新计算 restored_equation 的复杂度（PySR 视角）
    df["restored_complexity"] = df["restored_equation"].apply(recompute_complexity)

    # （可选）只算运算符数量，便于排查
    def ops_only(expr_str):
        expr = sympify_with_locals(expr_str)
        def _ops(e):
            if e.is_Add:
                return max(len(e.args) - 1, 0) + sum(_ops(a) for a in e.args)
            if e.is_Mul:
                return max(len(e.args) - 1, 0) + sum(_ops(a.base) if (a.is_Pow and a.exp == -1) else _ops(a) for a in e.args)
            if e.is_Pow:
                if e.exp in (sp.Rational(1, 2), -sp.Integer(1)):
                    return 1 + _ops(e.base)
                return 1 + _ops(e.base) + _ops(e.exp)
            if isinstance(e, sp.Function):
                return 1 + sum(_ops(a) for a in e.args)
            return 0
        return _ops(expr)

    df["ops_only_complexity"] = df["restored_equation"].apply(ops_only)

    # Pareto front
    pareto_df = pareto_front(df, loss_col="loss", comp_col="restored_complexity")
    output_path = fr"Symbolic Regression/srloop/data/hall_of_fame_run-{i}_restored_paretofront.csv"
    pareto_df.to_csv(output_path, index=False)

    print(f"完成！Pareto front 结果已保存到 {output_path}")
