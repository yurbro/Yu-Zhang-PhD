from skopt.space import Real

def create_skopt_bounds(X_train, pd_upper=1e3, pd_lower=1e-3):
    feature_count = X_train.shape[1]
    space = [
        Real(pd_lower, pd_upper, prior='log-uniform', name='v0'),
        Real(pd_lower, pd_upper, prior='log-uniform', name='a0'),
        Real(pd_lower, pd_upper, prior='log-uniform', name='a1'),
        Real(pd_lower, pd_upper, prior='log-uniform', name='v1')
    ]
    # 添加每个特征的权重参数
    space += [Real(pd_lower, pd_upper, prior='log-uniform', name=f'wl{i+1}') for i in range(feature_count)]
    return space
