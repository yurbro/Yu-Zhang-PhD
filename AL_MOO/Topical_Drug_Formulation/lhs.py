# requirements: pip install pyDOE2 numpy

import numpy as np
from pyDOE2 import lhs

def generate_lhs_samples(bounds, n_samples, criterion='maximin'):
    """
    使用拉丁超立方体采样生成样本，并映射到指定范围。
    
    Parameters
    ----------
    bounds : ndarray of shape (d, 2)
        每个变量的上下界 [[min1, max1], [min2, max2], ..., [mind, maxd]]。
    n_samples : int
        需要生成的样本点数量。
    criterion : str
        LHS 的优化标准，可选 'center', 'maximin', 'centermaximin', 'correlation' 等。

    Returns
    -------
    samples : ndarray of shape (n_samples, d)
        采样结果，每行是一个样本，各列对应不同变量。
    """
    bounds = np.asarray(bounds)
    d = bounds.shape[0]
    # 1. 在 [0,1]^d 上生成 LHS 点
    unit_lhs = lhs(d, samples=n_samples, criterion=criterion)
    # 2. 映射到实际范围
    samples = np.zeros_like(unit_lhs)
    for i in range(d):
        lo, hi = bounds[i]
        samples[:, i] = lo + unit_lhs[:, i] * (hi - lo)
    return samples

if __name__ == "__main__":
    # 示例：3 个变量，范围分别为
    # Poloxamer407: [20, 30], Ethanol: [10, 20], PG: [10, 20]
    bounds = np.array([
        [20.0, 30.0],
        [10.0, 20.0],
        [10.0, 20.0],
    ])
    n_samples = 500  # 生成 500 个样本
    lhs_samples = generate_lhs_samples(bounds, n_samples)

    # 输出前 5 个样本查看
    print(lhs_samples[:20])
