# -*- coding: utf-8 -*-
"""
main.py
-------
你的主程序：读数据 -> 构造 Dataset -> 设定 Config -> 调用训练 -> 使用最优模型
把真实 IVPT 数据替换到 make_dataset() 即可。
"""

import numpy as np
from sr_ode import Dataset, Record, Config, train_symbolic_ode, compile_individual, simulate_with_model, list_available_primitives, register_primitives
import matplotlib.pyplot as plt

def make_dataset() -> Dataset:
    rng = np.random.default_rng(0)
    t = np.array([0, 0.5, 1, 2, 3, 4, 5, 6], dtype=float)

    designs = [
        dict(C1=0.25, C2=0.12, C3=0.15),
        dict(C1=0.28, C2=0.18, C3=0.12),
        dict(C1=0.22, C2=0.15, C3=0.20),
        dict(C1=0.30, C2=0.10, C3=0.10),
        dict(C1=0.27, C2=0.14, C3=0.18),
        dict(C1=0.24, C2=0.20, C3=0.14),
        dict(C1=0.26, C2=0.16, C3=0.16),
        dict(C1=0.29, C2=0.11, C3=0.13),
        dict(C1=0.23, C2=0.19, C3=0.19),
        dict(C1=0.21, C2=0.13, C3=0.17),
        dict(C1=0.31, C2=0.17, C3=0.11),
        dict(C1=0.20, C2=0.15, C3=0.15),
        dict(C1=0.32, C2=0.14, C3=0.12),
        dict(C1=0.22, C2=0.18, C3=0.16),
        dict(C1=0.28, C2=0.12, C3=0.14),
        dict(C1=0.25, C2=0.16, C3=0.13),
    ]

    recs = []
    for d in designs:
        k = 0.6 + 0.1 * d["C2"]
        Qmax = 500.0 * (1.0 + 0.2 * d["C3"])
        Q = Qmax * (1.0 - np.exp(-k * t))
        Q += rng.normal(scale=3.0, size=Q.shape)

        # 现在把配方变量放进 vars 字典；只需要 C2 和 C3
        recs.append(Record(
            t=t,
            Q=Q,
            Q0=0.0,
            vars={"C2": d["C2"], "C3": d["C3"]}
        ))

    return Dataset(recs, normalize=True)

if __name__ == "__main__":
    ds = make_dataset()

    # —— 配置 ——（必要时微调）
    # 1) 查看内置原子
    print("Available primitives:", list_available_primitives())

    # 2) （可选）注册你自己的原子
    def my_clip01(a):
        return 0.0 if a < 0 else (1.0 if a > 1.0 else a)
    register_primitives({"clip01": (my_clip01, 1)})

    # 3) 配置里点名要用哪些
    cfg = Config(
        var_names=("Q","C2","C3"),
        must_have=("Q","C2","C3"),

        # 选择原子
        primitive_names=("add","sub","mul","div",
                        #  "log1p","exp","sqrt"
                         ),

        # —— GA 参数 ——
        pop_size=100, ngen=200, cxpb=0.6, mutpb=0.3,
        tournsize=5, tree_len_max=25,
        init_depth_min=1, init_depth_max=3,

        # —— 积分器参数（注意 substeps 只出现一次）——
        integrator="rk4",      # "euler" / "rk2" / "rk4" / "dopri5"
        substeps=8,
        adapt_refine_max=8,
        dt_floor=1e-6,
        qcap_factor=1.5,
        clamp_nonneg=True,

        # —— 惩罚项/约束 ——
        alpha_complexity=1e-2,
        enable_nest_penalty=True, nest_op="exp, log1p", nest_weight=1e-3,
        lambda_phys=0.0,
        lambda_cov=0.0,

        n_jobs=None,
        seed=13,
    )

    # —— 训练 ——（Windows 下请确保在 __main__ 保护内）
    hof, log, best, pset = train_symbolic_ode(ds, cfg)
    print("\nBest individual:", best)

    # —— 使用最优模型：编译并在一条记录上做对比 —— 
    f = compile_individual(best, pset)
    rec0 = ds.records[0]
    Q_pred = simulate_with_model(f, rec0, cfg)
    mse0 = float(np.mean((ds.scaled(Q_pred) - ds.scaled(rec0.Q))**2))
    print(f"Design#0 MSE (normalized): {mse0:.6f}")

    # —— 画图对比 —— 
    plt.figure()
    plt.plot(rec0.t, rec0.Q, 'o-', label='Q')
    plt.plot(rec0.t, Q_pred, 's--', label='Q_pred')
    plt.xlabel('t')
    plt.ylabel('Q')
    plt.title('Best Model Prediction on Design')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # scatter plot of true vs predicted for all records
    plt.figure()
    Q_all = np.hstack([rec.Q for rec in ds.records])
    Q_pred_all = np.hstack([simulate_with_model(f, rec, cfg) for rec in ds.records])
    plt.scatter(Q_all, Q_pred_all, alpha=0.5)
    plt.plot([Q_all.min(), Q_all.max()], [Q_all.min(), Q_all.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.tight_layout()
    plt.show()