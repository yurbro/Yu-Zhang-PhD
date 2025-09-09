
import numpy as np

# 关键变量
required_vars = ["C_pol", "C_pg", "C_eth", "t"]

def coverage_penalty(expr, alpha=10.0):
    """检查表达式是否包含所有 required_vars"""
    try:
        present_vars = {str(s) for s in expr.free_symbols}
    except Exception:
        return alpha * len(required_vars)
    missing = [v for v in required_vars if v not in present_vars]
    return alpha * len(missing)

def custom_loss(y_true, y_pred, expr, complexity, mu=0.001, lam=1.0):
    mse = np.mean((y_true - y_pred) ** 2)
    penalty = coverage_penalty(expr, alpha=10.0)
    return mse + mu * complexity + lam * penalty