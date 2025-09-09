
"""
This script stored the selected expressions.
"""

def Q_pred_pysr_38(C_pol, C_eth, C_pg, t):
    """
    Q(t,C) = ((x3 + inv(x3)) - 1.7105602) * (((20.446016 - (((x0 * (x1 + (((x0 * -0.33102357) + x2) / (x1 + -15.172693)))) * 0.025330327) + x2)) / ((x2 / (x1 + -16.764357)) + (x2 * -0.0822702))) + 11.088998)
    x0: C_pol
    x1: C_eth
    x2: C_pg
    x3: t
    """
    # 计算表达式
    x0 = C_pol
    x1 = C_eth
    x2 = C_pg
    x3 = t
    # inv(x3) = 1/x3
    inv_x3 = 1.0 / x3
    # inner1 = x0 * (x1 + (((x0 * -0.33102357) + x2) / (x1 + -15.172693)))
    inner1 = x0 * (x1 + (((x0 * -0.33102357) + x2) / (x1 + -15.172693)))
    # term_a = (inner1 * 0.025330327) + x2
    term_a = (inner1 * 0.025330327) + x2
    # numerator = 20.446016 - term_a
    numerator = 20.446016 - term_a
    # denominator = (x2 / (x1 + -16.764357)) + (x2 * -0.0822702)
    denominator = (x2 / (x1 + -16.764357)) + (x2 * -0.0822702)
    # main_term = (numerator / denominator) + 11.088998
    main_term = (numerator / denominator) + 11.088998
    # result = ((x3 + inv_x3) - 1.7105602) * main_term
    result = ((x3 + inv_x3) - 1.7105602) * main_term

    return result

def Q_pred_pysr_31(C_pol, C_eth, C_pg, t):

    """
    Q(t,C)=(x3 * (((19.271553 - ((((x1 + (((x0 * -0.3272755) + x2) / (x1 + -15.183248))) * x0) * 0.023550149) + x2)) / (x2 / (x1 + -16.631844))) + 10.830311)) - x2
    """
    # 计算表达式
    x0 = C_pol
    x1 = C_eth
    x2 = C_pg
    x3 = t
    # 计算内部表达式
    inner = (19.271553 - ((((x1 + (((x0 * -0.3272755) + x2) / (x1 + -15.183248))) * x0) * 0.023550149) + x2)) / (x2 / (x1 + -16.631844)) + 10.830311
    result = x3 * inner - x2

    return result

def Q_pred_pysr_29(C_pol, C_eth, C_pg, t):

    """
    Q(t,C)=(x3 * (((20.234722 - ((((x2 + (x0 * -0.18404673)) / (x1 + -15.171272)) + (x1 * 0.60644144)) + x2)) / (x2 / (x1 + -15.681594))) + 11.665971)) - x2
    """
    # 计算表达式
    x0 = C_pol
    x1 = C_eth
    x2 = C_pg
    x3 = t
    # 计算内部表达式
    inner = (20.234722 - ((((x2 + (x0 * -0.18404673)) / (x1 + -15.171272)) + (x1 * 0.60644144)) + x2)) / (x2 / (x1 + -15.681594)) + 11.665971
    result = x3 * inner - x2

    return result