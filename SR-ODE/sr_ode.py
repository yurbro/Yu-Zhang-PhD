# -*- coding: utf-8 -*-
"""
sr_ode.py
---------
通用的符号回归 ODE 框架（DEAP-GP）：
  学习 dQ/dt = f(Q, C1, C2, C3) 的解析式 f
  适应度 = 积分残差 (对齐所有采样点) + 可选惩罚（复杂度/物理/覆盖/嵌套）
  数值稳健：安全原子、状态裁剪、自适应子步、NaN早退

Windows 友好：所有用于并行的函数均为“顶层可pickle”，不使用 lambda。
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
from deap import base, creator, tools, gp

# ======== 数值安全 / 公共工具（保持原样） ========
SAT = 1e6
EPS = 1e-9
LOG_CAP = 50.0
DIV_MIN = 1e-6

def _sat(x: float, lo: float = -SAT, hi: float = SAT) -> float:
    if not math.isfinite(x): return 0.0
    if x > hi: return hi
    if x < lo: return lo
    return x

# ======== 安全原子算子（顶层、可 pickle） ========
def p_add(a,b): return _sat(a+b)
def p_sub(a,b): return _sat(a-b)
def p_mul(a,b): a=_sat(a); b=_sat(b); return _sat(a*b)
def p_div(a,b):
    a=_sat(a); b=_sat(b)
    if not math.isfinite(b) or abs(b) < DIV_MIN: return _sat(a)
    return _sat(a/b)

def p_log1p(a): a=_sat(a); return _sat(math.log1p(abs(a)))
def p_exp(a):   a=_sat(a, -LOG_CAP, LOG_CAP); return _sat(math.exp(a))
def p_sqrt(a):  a=_sat(a); return _sat(math.sqrt(abs(a)))

def p_abs(a):   return _sat(abs(_sat(a)))
def p_relu(a):  a=_sat(a); return _sat(a if a>0 else 0.0)
def p_tanh(a):  a=_sat(a, -LOG_CAP, LOG_CAP); return _sat(math.tanh(a))
def p_softplus(a):  # log(1+exp(x)) 的稳定实现
    a=_sat(a, -LOG_CAP, LOG_CAP)
    return _sat(math.log1p(math.exp(a)))

def p_min(a,b): return _sat(min(_sat(a), _sat(b)))
def p_max(a,b): return _sat(max(_sat(a), _sat(b)))
def p_pow2(a):  a=_sat(a); return _sat(a*a)
def p_pow3(a):  a=_sat(a); return _sat(a*a*a)

def p_neg(x): return -x


# ======== 原子注册表（名字 -> (函数, 元数)） ========
PRIMITIVE_REGISTRY: Dict[str, Tuple[callable,int]] = {
    # 基本算术
    "add": (p_add, 2), "sub": (p_sub, 2), "mul": (p_mul, 2), "div": (p_div, 2),
    # 单调/稳定的常用函数
    "log1p": (p_log1p, 1), "exp": (p_exp, 1), "sqrt": (p_sqrt, 1),
    "abs": (p_abs, 1), "relu": (p_relu, 1), "tanh": (p_tanh, 1),
    "softplus": (p_softplus, 1),
    # 其他可选
    "min": (p_min, 2), "max": (p_max, 2), "pow2": (p_pow2, 1), "pow3": (p_pow3, 1),
    # 如确需再加 sin/cos，但默认不推荐：振荡易过拟合
    # "sin": (math.sin, 1), "cos": (math.cos, 1),
    "neg": (p_neg, 1),
}

# 原子集合（默认不含 sin/cos；如需打开，请在主程序里传入）
DEFAULT_PRIMITIVES: List[Tuple[str, callable, int]] = [
    ("add", p_add, 2),
    ("sub", p_sub, 2),
    ("mul", p_mul, 2),
    ("div", p_div, 2),
    ("log1p", p_log1p, 1),
    ("exp", p_exp, 1),
    ("sqrt", p_sqrt, 1),
]

# =========================
# 数据结构
# =========================
@dataclass
class Record:
    """
    一条配方/实验的时间序列。
    - Q(t): 观测的累计渗透量
    - vars: 任意自变量（配方/环境等），键名须与 Config.var_names 中除 'Q' 以外的一致。
      例如：{"P407": 0.27, "EtOH": 0.15, "PG": 0.18}
    """
    t: np.ndarray                  # shape (K,), 递增
    Q: np.ndarray                  # shape (K,)
    Q0: float = 0.0
    vars: Dict[str, float] = field(default_factory=dict)

@dataclass
class Dataset:
    records: List[Record]
    normalize: bool = True
    q_scale: float = field(init=False, default=1.0)
    def __post_init__(self):
        if self.normalize and self.records:
            allQ = np.concatenate([r.Q for r in self.records])
            vmax = float(np.max(allQ)) if allQ.size else 1.0
            self.q_scale = vmax if vmax > 0 else 1.0
        else:
            self.q_scale = 1.0
    def scaled(self, Q: np.ndarray) -> np.ndarray:
        return Q / self.q_scale
    def descale(self, Qn: np.ndarray) -> np.ndarray:
        return Qn * self.q_scale

# =========================
# 模块配置
# =========================
@dataclass
class Config:
    var_names: Tuple[str, ...] = ("Q","C2","C3")
    must_have: Tuple[str, ...] = ("Q",)

    # —— 原子定义 —— 
    # 方案A：直接给名字列表（推荐，像 PySR）
    primitive_names: Tuple[str, ...] = ("add","sub","mul","div","log1p","exp","sqrt")
    # 方案B：额外自定义原子（名字->(函数,元数)），与注册表合并
    extra_primitives: Optional[Dict[str, Tuple[callable,int]]] = None

    # Ephemeral 常数范围
    ephemeral_range: Tuple[float,float] = (-1.0, 1.0)

    # —— 遗传 / 评估参数（保持原样，可略） ——
    pop_size: int = 200
    ngen: int = 60
    cxpb: float = 0.5
    mutpb: float = 0.4
    tournsize: int = 5
    tree_len_max: int = 25
    init_depth_min: int = 1
    init_depth_max: int = 3

    substeps: int = 8
    alpha_complexity: float = 1e-2
    lambda_phys: float = 0.0
    lambda_cov: float = 0.0
    enable_nest_penalty: bool = True
    nest_op: str = "exp"
    nest_weight: float = 1e-3

    n_jobs: Optional[int] = None
    seed: int = 13

    # —— ODE 积分器选项 ——
    integrator: str = "rk4"       # 可选: "euler", "rk2", "rk4", "dopri5"
    substeps: int = 8             # 定步积分的基础子步数
    adapt_refine_max: int = 8     # 自适应最多二分 2**8 次
    dt_floor: float = 1e-6        # 最小步长（避免无穷细分）
    qcap_factor: float = 1.5      # qcap = qcap_factor * max(Q_obs)
    clamp_nonneg: bool = True     # 是否强制 Q >= 0
    # 误差容差（仅 dopri5 用）
    rtol: float = 1e-6
    atol: float = 1e-9

# =========================
# 构建 PrimitiveSet
# =========================
def build_pset_from_config(cfg: Config) -> gp.PrimitiveSet:
    """
    根据 Config.primitive_names + extra_primitives 构建 pset。
    提供变量命名与 Ephemeral 常数范围。
    """
    pset = gp.PrimitiveSet("MAIN", arity=len(cfg.var_names))
    # 参数重命名
    for i, v in enumerate(cfg.var_names):
        pset.renameArguments(**{f"ARG{i}": v})

    # 准备原子字典：注册表 + 用户自定义（若重名，用户定义覆盖）
    prims: Dict[str, Tuple[callable,int]] = dict(PRIMITIVE_REGISTRY)
    if cfg.extra_primitives:
        prims.update(cfg.extra_primitives)

    # 按名字选择并添加
    for name in cfg.primitive_names:
        if name not in prims:
            raise ValueError(f"Primitive '{name}' not found in registry. "
                             f"Available: {sorted(prims.keys())}")
        func, ar = prims[name]
        pset.addPrimitive(func, arity=ar, name=name)

    # Ephemeral 常数（用 partial，Windows 可 pickle）
    lo, hi = cfg.ephemeral_range
    pset.addEphemeralConstant("const", partial(random.uniform, float(lo), float(hi)))
    return pset

# =========================
# 图工具：覆盖检测 & 嵌套深度
# =========================
def uses_vars_graph(individual, must_have: Sequence[str],
                    var_names: Sequence[str]) -> bool:
    """基于图查看个体是否使用了 must_have 变量。"""
    nodes, edges, labels = gp.graph(individual)
    present = set()
    for i, lab in labels.items():
        if lab in var_names:
            present.add(lab)
    return all(v in present for v in must_have)

def max_sameop_chain(individual, op_name: str = "exp") -> int:
    """同名算子连续嵌套的最长长度（用于惩罚 exp(exp(x)) 这类结构）"""
    nodes, edges, labels = gp.graph(individual)
    children = {i: [] for i in nodes}
    for (p, c) in edges:
        children[p].append(c)
    from functools import lru_cache
    @lru_cache(None)
    def dfs(u):
        here = 1 if labels[u] == op_name else 0
        best = here
        for v in children[u]:
            child_chain = dfs(v)
            if labels[u] == op_name and labels[v] == op_name:
                best = max(best, 1 + child_chain)
            else:
                best = max(best, here, child_chain)
        return best
    longest = 0
    for u in nodes:
        longest = max(longest, dfs(u))
    return longest

# =========================
# 动态实参取值器：按 var_names 拼装 f(*args)
# =========================

def make_arg_getter(var_names: Sequence[str]):
    """
    返回 _args(Q, rec) -> tuple，与 var_names 顺序一致。
    - 'Q' 用当前状态 Q
    - 其它变量名优先从 rec.vars[name] 取，若无再尝试 rec.<name> 属性。
    """
    vn = tuple(var_names)

    def _args(Q, rec: Record):
        vals = []
        for name in vn:
            if name == "Q":
                vals.append(float(Q))
            else:
                if isinstance(rec.vars, dict) and (name in rec.vars):
                    vals.append(float(rec.vars[name]))
                elif hasattr(rec, name):
                    vals.append(float(getattr(rec, name)))
                else:
                    raise AttributeError(
                        f"Record missing variable '{name}'. "
                        f"Expected by var_names={vn}. "
                        f"Please put it in rec.vars['{name}']."
                    )
        return tuple(vals)
    return _args

# =========================
# 数值积分（多积分器 + 稳健性）
# =========================
def _clip_state(Q: float, qcap: float, cfg: "Config") -> float:
    if not math.isfinite(Q): return float("nan")
    if cfg.clamp_nonneg and Q < 0.0: Q = 0.0
    if Q > qcap: Q = qcap
    return Q

def _step_euler(f, Q, rec: Record, dt, qcap, cfg: "Config", arg_getter):
    k1 = _sat(f(*arg_getter(Q, rec)))
    Qn = _sat(Q + dt * k1)
    return _clip_state(Qn, qcap, cfg)

def _step_rk2_heun(f, Q, rec: Record, dt, qcap, cfg: "Config", arg_getter):
    k1 = _sat(f(*arg_getter(Q, rec)))
    k2 = _sat(f(*arg_getter(_sat(Q + dt * k1), rec)))
    Qn = _sat(Q + 0.5 * dt * (k1 + k2))
    return _clip_state(Qn, qcap, cfg)

def _step_rk4(f, Q, rec: Record, dt, qcap, cfg: "Config", arg_getter):
    k1 = _sat(f(*arg_getter(Q, rec)))
    k2 = _sat(f(*arg_getter(_sat(Q + 0.5*dt*k1), rec)))
    k3 = _sat(f(*arg_getter(_sat(Q + 0.5*dt*k2), rec)))
    k4 = _sat(f(*arg_getter(_sat(Q +     dt*k3), rec)))
    Qn = _sat(Q + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4))
    return _clip_state(Qn, qcap, cfg)

def _select_stepper(name: str):
    name = name.lower()
    if name in ("euler", "rk1"): return _step_euler
    if name in ("rk2", "heun", "improved_euler"): return _step_rk2_heun
    if name in ("rk4", "classic"): return _step_rk4
    if name in ("dopri5", "rk45", "solve_ivp"): return "dopri5"
    raise ValueError(f"Unknown integrator '{name}'")

def simulate_series(f, rec: Record, t_points, Q0=0.0,
                    cfg: Optional["Config"]=None, qcap: Optional[float]=None) -> np.ndarray:
    if cfg is None:
        cfg = Config()
    t = np.asarray(t_points, dtype=float)
    out = np.empty_like(t)
    Q = max(0.0, float(Q0)) if cfg.clamp_nonneg else float(Q0)
    out[0] = Q

    if qcap is None:
        qcap = SAT

    stepper = _select_stepper(cfg.integrator)
    arg_getter = make_arg_getter(cfg.var_names)

    # --- SciPy 自适应（Dormand–Prince RK45） ---
    if stepper == "dopri5":
        try:
            from scipy.integrate import solve_ivp
        except Exception:
            # 没装 scipy 就回退到 rk4
            stepper = _step_rk4
        else:
            Q = max(0.0, float(Q0)) if cfg.clamp_nonneg else float(Q0)
            out[0] = Q
            for k in range(1, len(t)):
                t0, t1 = float(t[k-1]), float(t[k])
                if t1 <= t0:
                    out[k] = Q
                    continue

                def rhs(_t, y):
                    # y: array([Q])
                    return _sat(f(*arg_getter(y[0], rec)))

                # 用 substeps 设一个合理的 max_step 上限，避免一步跨太大
                max_step = max((t1 - t0) / max(1, int(cfg.substeps)), cfg.dt_floor)

                sol = solve_ivp(
                    rhs, (t0, t1), y0=[Q],
                    method="RK45",
                    rtol=cfg.rtol, atol=cfg.atol,
                    max_step=max_step
                )
                if (not sol.success) or (not np.isfinite(sol.y[0, -1])):
                    return np.full_like(t, np.nan)

                Q = _clip_state(float(sol.y[0, -1]), qcap, cfg)
                out[k] = Q
            return out


    # 定步 + 自适应二分
    if callable(stepper):
        for k in range(1, len(t)):
            t0, t1 = float(t[k-1]), float(t[k])
            if t1 <= t0:
                out[k] = Q; continue
            dt_total = t1 - t0
            n0 = max(1, int(cfg.substeps))
            ok = False
            for refine in range(cfg.adapt_refine_max + 1):
                n = n0 * (2 ** refine)
                dt = max(dt_total / n, cfg.dt_floor)
                Qtry = Q
                finite = True
                for _ in range(n):
                    Qtry = stepper(f, Qtry, rec, dt, qcap, cfg, arg_getter)
                    if not math.isfinite(Qtry):
                        finite = False
                        break
                if finite:
                    Q = Qtry; ok = True; break
            if not ok:
                return np.full_like(t, np.nan)
            out[k] = Q
        return out

    raise RuntimeError("Integrator dispatch failed.")

# =========================
# 顶层评估（Windows 可并行）
# =========================
def eval_individual(individual, pset, dataset: Dataset, cfg: "Config"):
    # 覆盖硬约束（不变）
    if cfg.must_have and (not uses_vars_graph(individual, cfg.must_have, cfg.var_names)):
        return (1e6,)

    try:
        f = gp.compile(expr=individual, pset=pset)
    except Exception:
        return (1e9,)

    # —— 模拟 + 残差 ——
    se_sum, n_total = 0.0, 0
    for rec in dataset.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        Q_pred = simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        if not np.isfinite(Q_pred).all():
            return (1e6,)
        se_sum += float(np.sum((dataset.scaled(Q_pred) - dataset.scaled(rec.Q)) ** 2))
        n_total += rec.Q.size
    mse = se_sum / max(1, n_total)

    comp_pen = cfg.alpha_complexity * len(individual)

    # 嵌套惩罚/覆盖软罚（如你已有，保持）
    nest_pen = 0.0
    if cfg.enable_nest_penalty and cfg.nest_op:
        chain = max_sameop_chain(individual, cfg.nest_op)
        if chain > 1:
            nest_pen = cfg.nest_weight * (chain - 1)**2

    cov_pen = 0.0
    if cfg.lambda_cov > 0 and cfg.must_have:
        miss = sum(1 for v in cfg.must_have if v not in str(individual))
        cov_pen = cfg.lambda_cov * miss

    # —— 物理先验罚（可选）——
    phys_pen = 0.0
    if cfg.lambda_phys > 0.0 and dataset.records:
        arg_getter = make_arg_getter(cfg.var_names)
        eps = 1e-4
        sample_recs = dataset.records[:min(4, len(dataset.records))]
        for r in sample_recs:
            # qmax = float(np.max(r.Q)) if r.Q.size else 1.0
            # Q_grid = np.linspace(0.0, max(1.0, min(qmax*2.0, qmax + 1.0)), num=9)
            qmax = np.percentile(np.concatenate([r.Q for r in dataset.records]), 95)
            Q_grid = np.linspace(0, qmax, 11)
            for Q in Q_grid:
                try:
                    val = _sat(f(*arg_getter(Q, r)))
                    phys_pen += max(0.0, -val)**2
                    phys_pen += max(0.0, abs(val) - (1.0 + qmax/2.0))**2
                    vp = _sat(f(*arg_getter(Q+eps, r)))
                    vm = _sat(f(*arg_getter(max(Q-eps, 0.0), r)))
                    dfdQ = (vp - vm) / (2*eps)
                    phys_pen += max(0.0, dfdQ)**2
                except Exception:
                    phys_pen += 10.0

    total = mse + comp_pen + nest_pen + cov_pen + cfg.lambda_phys * phys_pen
    return (total,)

# =========================
# 构建 GP & 训练
# =========================
def build_gp(cfg: Config, pset: gp.PrimitiveSet):
    random.seed(cfg.seed)
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=cfg.init_depth_min, max_=cfg.init_depth_max)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=cfg.tree_len_max))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=cfg.tree_len_max))
    return toolbox

def train_symbolic_ode(dataset: Dataset, cfg: Config):
    pset = build_pset_from_config(cfg)   # ← 使用新的按名字构建
    toolbox = build_gp(cfg, pset)
    toolbox.register("evaluate", partial(eval_individual, pset=pset, dataset=dataset, cfg=cfg))

    # 并行（Windows 需放在 __main__ 下调用本函数）
    pool = None
    if cfg.n_jobs is None:
        import multiprocessing as mp
        cfg.n_jobs = max(1, mp.cpu_count()-1)
    if cfg.n_jobs > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=cfg.n_jobs)
        toolbox.register("map", pool.map)

    pop = toolbox.population(n=cfg.pop_size)
    hof = tools.HallOfFame(maxsize=5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean); stats.register("min", np.min); stats.register("max", np.max)

    # 轻量 eaSimple（避免额外依赖）
    pop, log = _ea_simple(pop, toolbox, cfg.cxpb, cfg.mutpb, cfg.ngen, stats, hof, verbose=True)

    if pool is not None:
        pool.close(); pool.join()
    best = hof[0] if len(hof) else tools.selBest(pop, 1)[0]
    return hof, log, best, pset

def _ea_simple(population, toolbox, cxpb, mutpb, ngen, stats, hof, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid = [ind for ind in population if not ind.fitness.valid]
    fits = list(toolbox.map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit
    if hof is not None:
        hof.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid), **record)
    if verbose: print(logbook.stream)

    for gen in range(1, ngen+1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                gp.cxOnePoint(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        if hof is not None:
            hof.update(offspring)
        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid), **record)
        if verbose: print(logbook.stream)
    return population, logbook

# =========================
# 便捷 API
# =========================
def compile_individual(individual, pset):
    """编译最佳个体为 Python 函数 f(Q,C1,C2,C3)"""
    return gp.compile(expr=individual, pset=pset)

def simulate_with_model(f, rec: Record, cfg: "Config") -> np.ndarray:
    qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
    return simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)

# =========================
def list_available_primitives() -> List[str]:
    """返回注册表内可用的原子名（按字母序）"""
    return sorted(PRIMITIVE_REGISTRY.keys())

def register_primitives(new_prims: Dict[str, Tuple[callable,int]]):
    """
    以模块级方式添加/覆盖原子（注意函数必须是顶层可 pickle）
    用法：register_primitives({"mysig": (my_sigmoid, 1)})
    """
    PRIMITIVE_REGISTRY.update(new_prims)
