
import numpy as np

# 关键变量
required_vars = ["C_pol", "C_pg", "C_eth", "t"]

# ==== 2. coverage penalty 计算 ====
def coverage_penalty(expr, alpha=10.0):
    """
    检查表达式是否覆盖所有 required_vars。
    如果缺少某个变量 -> 每个缺失变量加 alpha 的惩罚。
    """
    try:
        present_vars = {str(s) for s in expr.free_symbols}
    except Exception:
        return alpha * len(required_vars)
    missing = [v for v in required_vars if v not in present_vars]
    return alpha * len(missing)

# ==== 3. 自定义 loss ====
def custom_loss(y_true, y_pred, expr, complexity, mu=0.001, lam=1.0):
    """
    y_true: numpy 数组，真实值
    y_pred: numpy 数组，预测值
    expr: Sympy 表达式 (当前候选模型)
    complexity: int，表达式复杂度
    mu: float，复杂度惩罚系数
    lam: float，coverage penalty 权重
    """
    mse = np.mean((y_true - y_pred) ** 2)
    penalty = coverage_penalty(expr, alpha=10.0)
    return mse + mu * complexity + lam * penalty