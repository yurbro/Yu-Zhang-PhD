import numpy as np
import pandas as pd
from datetime import datetime  # 新增
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from lhs_sample import lhs_samples
# from ackley_func import ackley_max
# from zakharov_func import ackley_max, run_ackley        # TODO: this is zakharove function actually
# from rastrigin_func import ackley_max, run_ackley  # This is actually the Rastrigin function, but we use it as a placeholder for the benchmark
# from rosenbrock_func import ackley_max, run_ackley  # This is actually the Rosenbrock function, but we use it as a placeholder for the benchmark
from griewank_func import ackley_max, run_ackley  # This is actually the Griewank function, but we use it as a placeholder for the benchmark

# np.random.seed(42) 

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    计算期望改进（Expected Improvement）采集函数
    X: 待评价点 (n_points, dim)
    X_sample: 已采样点 (n_sample, dim)
    Y_sample: 已采样观测值 (n_sample,)
    gpr: 已训练的GP模型
    xi: 探索参数
    返回: EI 值 (n_points,)
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def upper_confidence_bound(X, X_sample, Y_sample, gpr, kappa=2.576):
    """
    计算上置信界（Upper Confidence Bound）采集函数
    X: 待评价点 (n_points, dim)
    X_sample: 已采样点 (n_sample, dim)
    Y_sample: 已采样观测值 (n_sample,)
    gpr: 已训练的GP模型
    kappa: 探索参数
    返回: UCB 值 (n_points,)
    """
    mu, sigma = gpr.predict(X, return_std=True)
    return mu + kappa * sigma

def probability_of_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    计算改进概率（Probability of Improvement）采集函数
    X: 待评价点 (n_points, dim)
    X_sample: 已采样点 (n_sample, dim)
    Y_sample: 已采样观测值 (n_sample,)
    gpr: 已训练的GP模型
    xi: 探索参数
    返回: PI 值 (n_points,)
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0
    return pi

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    """
    在给定的边界内，通过多次随机重启优化采集函数，寻找最优下一个采样点
    bounds: 列表 [(min, max), ...]
    返回: 最优点 x (1, dim)
    """
    dim = X_sample.shape[1]
    min_val = 1e20
    best_x = None

    for i in range(n_restarts):
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(dim,))
        # 简单局部搜索：随机微调
        for _ in range(100):
            xi = x0 + 0.01 * np.random.randn(dim)
            xi = np.clip(xi, [b[0] for b in bounds], [b[1] for b in bounds])
            acq_val = acquisition(xi.reshape(1, -1), X_sample, Y_sample, gpr)
            # print(f"acquisition({acquisition.__name__}) at {xi.round(4)} = {acq_val}")  # 打印采集函数值
            val = -acq_val
            if val < min_val:
                min_val = val
                best_x = xi

    return np.clip(best_x, [b[0] for b in bounds], [b[1] for b in bounds]).reshape(1, -1)


def bayesian_optimization(objective_func, X_init, Y_init, bounds, n_iter, af_func, benchmark, acquisition_func):
    """
    单目标贝叶斯优化主函数
    objective_func: 目标函数, 输入 (dim,) 返回标量
    X_init: 初始采样点, numpy array, (n_init, dim)
    Y_init: 初始观测值, numpy array, (n_init,)
    bounds: 变量边界列表 [(min, max), ...]
    n_iter: 迭代次数
    返回: 样本点和对应观测值
    """
    # 复制初始数据
    X_sample = X_init.copy()
    Y_sample = Y_init.copy()

    # 记录每6次的结果
    record = []

    # 定义 GP 模型
    # kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
    # kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(alpha=1e-10, normalize_y=True)

    for i in range(n_iter):
        # 训练 GP
        gpr.fit(X_sample, Y_sample)

        # 提议下一个点
        # acquisition_func = probability_of_improvement # 可以替换为 upper_confidence_bound 或 probability_of_improvement
        x_next = propose_location(acquisition_func, X_sample, Y_sample, gpr, bounds)

        # 计算目标函数真实值
        y_next = objective_func(x_next.ravel())

        # 增加新的样本
        X_sample = np.vstack((X_sample, x_next))
        Y_sample = np.append(Y_sample, y_next)

        # 判断是否找到更大的值
        if y_next > np.max(Y_sample[:-1]):
            print(f"\033[95mIteration {i+1}: x_next={x_next.ravel()}, y_next={y_next}, New best {benchmark}={y_next}\033[0m")
        else:
            print(f"Iteration {i+1}: x_next={x_next.ravel()}, y_next={y_next}, Current best {benchmark}={np.max(Y_sample)}")

        # 每n次记录一次
        if (i + 1) % 6 == 0:
            record.append({
            'iteration': i + 1,
            **{f'x{j+1}': x_next.ravel()[j] for j in range(x_next.shape[1])},
            'y_next': y_next,
            f'Current best {benchmark}': np.max(Y_sample),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 新增
            })

    # 保存为Excel（追加模式，不覆盖原有结果）
    if record:
        df = pd.DataFrame(record)
        save_path = fr"Multi-Objective Optimisation\Benchmark\Package Module-III-griewank\BO-RE\single_bo_{af_func}_{benchmark}_{dim}D.xlsx"
        try:
            # 如果文件已存在，读取旧数据并追加
            old_df = pd.read_excel(save_path)
            df = pd.concat([old_df, df], ignore_index=True)
        except FileNotFoundError:
            pass  # 文件不存在则直接保存新数据
        df.to_excel(save_path, index=False)
        print(f"Saved record to {save_path}")

    return X_sample, Y_sample

def ackley_function(**kwargs) -> float:
    # 将关键字参数转换为有序数组
    x = np.array([kwargs[f'x{i+1}'] for i in range(len(kwargs))], dtype=float)
    """
    计算标准 Griewank 函数值（最小化目标）。
    输入：
        x: np.ndarray 或 list, 形状可以是 (d,) 或 (n,d)
           每行表示一个 d 维向量 [x1, x2, ..., xd]。
    返回：
        np.ndarray, shape (n,)
    公式：
        f(x) = sum(x_i^2) / 4000 - prod(cos(x_i / sqrt(i))) + 1
        其中 i=1,...,d
    """
    
    x_arr = np.atleast_2d(x).astype(float)
    n, d = x_arr.shape
    return -(np.sum(x_arr**2, axis=1) / 4000 - np.prod(np.cos(x_arr / np.sqrt(np.arange(1, d + 1))), axis=1) + 1)

def array_to_ackley(x):
    # x: 1D array
    return ackley_function(**{f'x{i+1}': x[i] for i in range(len(x))})

def run_moo_initial_experiment(n_init, lb, ub, benchmark):
    """
    Run the initial experiment for multi-objective optimisation.
    """
    # 1. Initialise the experiment data by using the lhs_sampling
    X_init = lhs_samples(n_init, lb, ub)  # Shape (n_init, dim)
    # 2.1 Evaluate the initial samples using the griewank function
    Y_init = ackley_max(X_init)
    # Y_init = np.array([griewank_function(**{f'x{i+1}': x[i] for i in range(len(x))}) for x in X_init])
    for xi, yi in zip(X_init, Y_init):
        print(f"x={xi.round(3)} → {benchmark}={yi:.4f}")

    return X_init, Y_init

# 示例用法
if __name__ == "__main__":

    # 设置初始参数
    start_time = datetime.now()
    print(f"Starting Bayesian Optimization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    n_init = 10  # 初始采样点数量
    n_iter = 180  # 优化迭代次数
    dim = 5  # 维度
    af_func = 'poi'  # 采集函数类型
    benchmark = 'Griewank'  # 基准函数 Griewank
    acquisition_func = {
        'ei': expected_improvement,
        'ucb': upper_confidence_bound,
        'poi': probability_of_improvement
    }
    lb, ub = np.array([-600] * dim), np.array([600] * dim)  # Lower and upper bounds
    X_init, Y_init = run_moo_initial_experiment(n_init, lb, ub, benchmark= benchmark)
    # 打印初始采样点和对应的目标函数值
    print(f"Initial samples (shape {X_init.shape}):")
    print(X_init.round(3))
    print(f"Initial objective values (shape {Y_init.shape}):")
    print(Y_init.round(3))

    # 根据X的维度设置边界
    dim = X_init.shape[1]
    bounds = {f'x{i+1}': (-600, 600) for i in range(dim)}
    bounds = [(b[0], b[1]) for b in bounds.values()]
    # np.random.seed(42) 
    X_opt, Y_opt = bayesian_optimization(array_to_ackley, 
                                         X_init, Y_init, 
                                         bounds, n_iter=n_iter, 
                                         af_func=af_func,
                                         benchmark=benchmark, 
                                         acquisition_func=acquisition_func[af_func])
    print("Optimization completed.")

    # 输出最佳优化值及对应输入参数
    best_idx = np.argmax(Y_opt)
    best_x = X_opt[best_idx]
    best_y = Y_opt[best_idx]
    print(f"Best value: {best_y}")
    print(f"Best input: {best_x}")
    print("---------Successfully completed the Bayesian Optimization process---------")
    end_time = datetime.now()
    print(f"Ending time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {end_time - start_time}")