# -*- coding: utf-8 -*-
"""
sr_ode_mod.py
-------------
DEAP-GP based symbolic regression for dQ/dt = f(Q, C1, C2, C3).
- Fitness = ODE simulation residual (strong form) + penalties (complexity, multi-op nesting)
- Hard physics filters: f >= 0, d f/dQ <= 0 (on sampled grid)
- Optional soft prior: formulation gradients (C1:-, C2:+, C3:+)
- Robust numerics: safe atoms, saturation, clipping, RK4 with substeps + adaptive refinement
- Readable export: infix string + SymPy + LaTeX
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Sequence
import math, random, numpy as np

# 可选进度条：有 tqdm 就用，没有就退化为 range
try:
    from tqdm import trange
except Exception:
    trange = range

import functools
import random
from deap import base, creator, tools, gp, algorithms

# =========================
# Saturation & safe atoms
# =========================
SAT = 1e6
DIV_MIN = 1e-8
_CLAMP_LOG1P = 0

def _sat(x: float, lo: float = -SAT, hi: float = SAT) -> float:
    if not math.isfinite(x): return 0.0
    if x > hi: return hi
    if x < lo: return lo
    return x

def p_add(a,b): return _sat(a+b)
def p_sub(a,b): return _sat(a-b)
def p_mul(a,b): a=_sat(a); b=_sat(b); return _sat(a*b)
def p_div(a,b):
    a=_sat(a); b=_sat(b)
    if not math.isfinite(b) or abs(b) < DIV_MIN: return 0.0
    return _sat(a/b)
def p_log1p(x):
    global _CLAMP_LOG1P
    x = _sat(x)
    if x < -0.999999:
        _CLAMP_LOG1P += 1
        x = -0.999999
    return _sat(math.log1p(x))
def p_exp(x):   # gentle cap before exp
    x = max(-50.0, min(50.0, _sat(x)))
    return _sat(math.exp(x))
def p_sqrt(x):  return _sat(math.sqrt(max(0.0, _sat(x))))
def p_softplus(x): # log(1+exp(x)) with stability
    x = _sat(x)
    if x > 20: return _sat(x)    # ~ x for large x
    if x < -20: return _sat(math.exp(x))
    return _sat(math.log1p(math.exp(x)))
def p_abs(x): return _sat(abs(_sat(x)))
def p_relu(x): return _sat(max(0.0, _sat(x)))
def p_tanh(x):
    x = max(-50.0, min(50.0, _sat(x)))
    return _sat(math.tanh(x))
def p_min(a,b): return _sat(min(_sat(a), _sat(b)))
def p_max(a,b): return _sat(max(_sat(a), _sat(b)))
def p_pow2(a):  a=_sat(a); return _sat(a*a)
def p_pow3(a):  a=_sat(a); return _sat(a*a*a)
def p_neg(x): return _sat(-_sat(x))

# registry: name -> (fn, arity)
PRIMITIVE_REGISTRY: Dict[str, Tuple[callable,int]] = {
    "add": (p_add, 2), "sub": (p_sub, 2), "mul": (p_mul, 2), "div": (p_div, 2),
    "log1p": (p_log1p, 1), "exp": (p_exp, 1), "sqrt": (p_sqrt, 1),
    "softplus": (p_softplus, 1), "abs": (p_abs, 1), "relu": (p_relu, 1), "tanh": (p_tanh, 1),
    "min": (p_min, 2), "max": (p_max, 2), "pow2": (p_pow2, 1), "pow3": (p_pow3, 1),
    "neg": (p_neg, 1),
}

# === DEBUG: 硬淘汰计数 ===
from collections import Counter
HARD_KO = Counter()

def ko(reason: str):
    """
    记录一次硬性淘汰原因，并返回 (1e6,) 作为个体的致命罚分。
    用法：把 return (1e6,) 改成 return ko("reason_key")
    """
    HARD_KO[reason] += 1
    return (1e6,)

# =========================
# Data containers
# =========================
class Record:
    __slots__ = ("t","Q","Q0","vars")
    def __init__(self, t: np.ndarray, Q: np.ndarray, Q0: float, vars: Dict[str,float]):
        self.t = np.asarray(t, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.Q0 = float(Q0)
        self.vars = dict(vars)

class Dataset:
    def __init__(self, records: List[Record]):
        self.records = records
        allQ = np.concatenate([r.Q for r in records]) if records else np.array([1.0])
        self.q_scale = float(np.nanmax(np.abs(allQ))) if allQ.size else 1.0
        self.normalized = True

# =========================
# Config
# =========================
@dataclass
class Config:
    var_names: Tuple[str, ...] = ("Q","C1","C2","C3")
    must_have: Tuple[str, ...] = ("Q","C1","C2","C3")

    primitive_names: Tuple[str, ...] = ("add","sub","mul","div","log1p","softplus","exp","sqrt","pow2")
    extra_primitives: Optional[Dict[str, Tuple[callable,int]]] = None
    ephemeral_range: Tuple[float,float] = (-2.0, 2.0)

    # GP
    pop_size: int = 400
    ngen: int = 100
    cxpb: float = 0.6
    mutpb: float = 0.5
    tournsize: int = 4
    tree_len_max: int = 25
    init_depth_min: int = 1
    init_depth_max: int = 4

    # Integrator
    integrator: str = "rk4"
    substeps: int = 8
    adapt_refine_max: int = 8
    dt_floor: float = 1e-6
    qcap_factor: float = 1.5
    clamp_nonneg: bool = True

    # Penalties
    alpha_complexity: float = 2e-4

    # Multi-op nesting penalties: {op: weight}; penalty = w * (L-1)^2 for longest chain length L>1
    nest_penalties: Optional[Dict[str, float]] = None  # e.g. {"exp":5e-3,"softplus":2e-3,"log1p":1e-3,"pow2":5e-4}

    # Hard physics filters
    hard_nonneg: bool = True
    hard_dfdQ_nonpos: bool = True
    hard_tol: float = 1e-9
    grad_check_points: int = 11
    grad_eps_rel: float = 1e-3

    # Soft prior on formulation gradients (C1:-, C2:+, C3:+)
    lambda_gradC: float = 0.0
    gradC_signs: Optional[Dict[str, int]] = None
    lambda_divQ: float = 1e-4   # 归一化适应度下；先开小点

    # Others
    n_jobs: Optional[int] = None
    seed: int = 13

    # Logging / progress
    verbosity: int = 2        # 0=静默, 1=checkpoint, 2=每代简报
    show_progress: bool = False  # 是否显示 tqdm 进度条
    log_best_every: int = 20  # 每多少代打印一次最优表达式预览

    # Logging 下方任意处追加
    hard_dfdQ_margin: float = 0.0   # 需要 d f/dQ ≤ -margin 的点数
    hard_dfdQ_min_hits: int = max(3, grad_check_points//2)      # 至少 K 个网格点满足
    
    # normalize fitness
    normalize_fitness: bool = True

    # 更硬的递减检查
    hard_check_all_points: bool = False      # True=所有网格点都要满足；False=只检查尾段
    hard_check_tail_frac: float = 0.5       # 当 all_points=False 时，检查后 50% 的网格点
    hard_check_nrecords: int = 12           # 每次评估最多抽 12 条记录（或全量，二选一）

    lambda_qsens: float = 1e-2
    qsens_min_ratio: float = 0.05

    lambda_gradC_mag: float = 1e-4  # 对 |df/dC|^2 的惩罚
    gradC_mag_min: float = 1e-3    # 期望的最小梯度幅值

    # === 在 Config 里新增（给默认值）===
    hard_gradC_required: bool = True
    hard_gradC_mag_min: float = 5e-3   # τ，按你变量已归一化到[0,1]的量级设
    hard_gradC_min_hits: int = 8      # K，在每条记录×若干Q点的抽样上统计

    hard_no_zero_denom: bool = True

    # 在 Config dataclass 里加两个字段（有默认值）
    debug_ko_every: int = 50
    debug_ko_reset_each_dump: bool = False
    debug_serial_eval: bool = False   # 调试期置 True 可关闭并行，确保计数器可见

    # —— Hall of Fame 快照 —— 
    hof_snapshot_every: int = 0            # 0=关闭；>0 表示每隔多少代保存一次
    hof_snapshot_dir: Optional[str] = None # 为空则用默认 artifacts 目录
    hof_snapshot_topk: int = 20            # 每次保存前 K 个

# =========================
# GP primitives (DEAP)
# =========================
from deap import gp, base, tools, creator
from functools import partial

def list_available_primitives() -> List[str]:
    return sorted(PRIMITIVE_REGISTRY.keys())

def register_primitives(new_prims: Dict[str, Tuple[callable,int]]):
    PRIMITIVE_REGISTRY.update(new_prims)

def build_pset_from_config(cfg: Config):
    pset = gp.PrimitiveSet("MAIN", len(cfg.var_names))
    # map variable order
    for i, name in enumerate(cfg.var_names):
        pset.renameArguments(**{f"ARG{i}": name})
    # add primitives
    reg = dict(PRIMITIVE_REGISTRY)
    if cfg.extra_primitives:
        reg.update(cfg.extra_primitives)
    for name in cfg.primitive_names:
        fn, ar = reg[name]
        pset.addPrimitive(fn, ar, name=name)
    # -------- FIX 1: Ephemeral 常数（兼容你这版 DEAP + 可 pickling）--------
    low, high = cfg.ephemeral_range
    pset.addEphemeralConstant("C", partial(random.uniform, low, high))
    # ↑ 仅传 (name, generator)。不要传类型参数；且用 partial 避免 Windows 多进程下 lambda 不能被 pickling。

    return pset

def build_gp(cfg: Config, pset):
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=cfg.init_depth_min, max_=cfg.init_depth_max)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=cfg.tree_len_max))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=cfg.tree_len_max))

    # -------- FIX 2: 高斯抖动新建 Terminal 的参数顺序（value, ret_type）--------
    def mut_gauss_const(individual, mu=0.0, sigma=0.2, p=0.3):
        for i, node in enumerate(individual):
            # 仅处理 Ephemeral 常数（终端，且有数值）
            if isinstance(node, gp.Terminal) and node.arity == 0 and (getattr(node, "value", None) is not None):
                if isinstance(node.value, (int, float)) and random.random() < p:
                    node.value = float(node.value) + random.gauss(mu, sigma)  # <- 直接改值
        return (individual,)
    toolbox.register("mutate_gauss", mut_gauss_const)
    return toolbox

# =========================
# Compile & argument getter
# =========================
def compile_individual(individual, pset):
    return gp.compile(expr=individual, pset=pset)

def make_arg_getter(var_names: Sequence[str]):
    vn = tuple(var_names)
    def _args(Q, rec: Record):
        vals = []
        for name in vn:
            if name == "Q":
                vals.append(float(Q))
            else:
                vals.append(float(rec.vars[name]))
        return tuple(vals)
    return _args

# =========================
# Integration (RK4/Heun/Euler)
# =========================
def _clip_state(Q: float, qcap: float, cfg: Config) -> float:
    if not math.isfinite(Q): return float("nan")
    if cfg.clamp_nonneg and Q < 0.0: Q = 0.0
    if Q > qcap: Q = qcap
    return Q

def step_euler(f, Q, rec, dt, qcap, cfg, args): # pragma: no cover
    return _clip_state(_sat(Q + dt * _sat(f(*args(Q, rec)))), qcap, cfg)

def step_heun(f, Q, rec, dt, qcap, cfg, args):  # improved Euler
    k1 = _sat(f(*args(Q, rec)))
    Qe = _sat(Q + dt * k1)
    k2 = _sat(f(*args(Qe, rec)))
    return _clip_state(_sat(Q + 0.5*dt*(k1 + k2)), qcap, cfg)

def step_rk4(f, Q, rec, dt, qcap, cfg, args):
    k1 = _sat(f(*args(Q, rec)))
    k2 = _sat(f(*args(Q + 0.5*dt*k1, rec)))
    k3 = _sat(f(*args(Q + 0.5*dt*k2, rec)))
    k4 = _sat(f(*args(Q + dt*k3, rec)))
    return _clip_state(_sat(Q + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)), qcap, cfg)

def _select_stepper(name: str):
    if name == "rk4": return step_rk4
    if name == "heun": return step_heun
    return step_euler

def simulate_series(f, rec: Record, t_points: np.ndarray, Q0: float, cfg: Config, qcap: float) -> np.ndarray:
    t = np.asarray(t_points, dtype=float)
    out = np.empty_like(t)
    Q = max(0.0, float(Q0)) if cfg.clamp_nonneg else float(Q0)
    stepper = _select_stepper(cfg.integrator)
    args = make_arg_getter(("Q","C1","C2","C3"))  # fixed order used in compile

    # If t[0] > 0.0, integrate 0 -> t[0] once (robustness)
    if t[0] > 0.0:
        t0, t1 = 0.0, float(t[0])
        dt_total = t1 - t0
        n0 = max(1, int(cfg.substeps))
        ok = False
        for refine in range(cfg.adapt_refine_max + 1):
            n = n0 * (2 ** refine)
            dt = max(dt_total / n, cfg.dt_floor)
            Qtry = Q; finite = True
            for _ in range(n):
                Qtry = stepper(f, Qtry, rec, dt, qcap, cfg, args)
                if not math.isfinite(Qtry):
                    finite = False; break
            if finite: Q = _clip_state(Qtry, qcap, cfg); ok = True; break
        if not ok: return np.full_like(t, np.nan)
        out[0] = Q
    else:
        out[0] = Q

    for k in range(1, len(t)):
        t0, t1 = float(t[k-1]), float(t[k])
        if t1 <= t0: out[k] = Q; continue
        dt_total = t1 - t0
        n0 = max(1, int(cfg.substeps))
        ok = False
        for refine in range(cfg.adapt_refine_max + 1):
            n = n0 * (2 ** refine)
            dt = max(dt_total / n, cfg.dt_floor)
            Qtry = Q; finite = True
            for _ in range(n):
                Qtry = stepper(f, Qtry, rec, dt, qcap, cfg, args)
                if not math.isfinite(Qtry):
                    finite = False; break
            if finite: Q = _clip_state(Qtry, qcap, cfg); ok = True; break
        if not ok: return np.full_like(t, np.nan)
        out[k] = Q
    return out

# =========================
# Graph helpers for nesting
# =========================
def max_sameop_chain(individual, op_name: str) -> int:
    nodes, edges, labels = gp.graph(individual)
    children = {i: [] for i in nodes}
    for p,c in edges:
        children[p].append(c)
    best = 0
    def dfs(u) -> int:
        nonlocal best
        here = 1 if labels[u]==op_name else 0
        longest = here
        for v in children[u]:
            child = dfs(v)
            if labels[u]==op_name and labels[v]==op_name:
                longest = max(longest, 1+child)
            else:
                longest = max(longest, here, child)
        best = max(best, longest)
        return longest
    for u in nodes: dfs(u)
    return best

def sameop_chain_lengths(individual, op_names: Sequence[str]) -> Dict[str,int]:
    nodes, edges, labels = gp.graph(individual)
    children = {i: [] for i in nodes}
    for p,c in edges: children[p].append(c)
    from functools import lru_cache
    @lru_cache(None)
    def dfs(u, target):
        here = 1 if labels[u]==target else 0
        best = here
        for v in children[u]:
            child = dfs(v, target)
            if labels[u]==target and labels[v]==target:
                best = max(best, 1+child)
            else:
                best = max(best, here, child)
        return best
    res={}
    for op in op_names:
        longest=0
        for u in nodes:
            longest=max(longest, dfs(u, op))
        res[op]=longest
    return res

# =========================
# Fitness evaluation
# =========================
def eval_individual(individual, pset, dataset: Dataset, cfg: Config):
    # must_have hard coverage
    nodes, edges, labels = gp.graph(individual)
    present = {lab for _,lab in labels.items()}
    for v in cfg.must_have:
        if v not in present:
            return ko("must_have_missing")   # 没包含 Q/C1/C2/C3 中的某个
        
    # ====== 补丁B：零分母结构硬筛 —— 卡掉 sub(Ci, Ci) 这类“分母=0”的投机体 ======
    if getattr(cfg, "hard_no_zero_denom", True):
        try:
            # 构造 children 邻接
            children = {i: [] for i in nodes}
            for p, c in edges:
                children[p].append(c)

            VARS = ("C1", "C2", "C3", "Q")

            # 1) 直接模式：div( … , sub(Ck, Ck) )
            for u in nodes:
                if labels[u] == "div" and len(children[u]) >= 2:
                    denom = children[u][1]
                    if labels[denom] == "sub" and len(children[denom]) == 2:
                        a, b = children[denom]
                        if labels.get(a) in VARS and labels.get(a) == labels.get(b):
                            # 典型“C3 - C3” → 分母恒为0，被 p_div 短路为 0，从而把变量“吃掉”
                            return ko("zero_denom_pattern")

            # 2)（可选，轻量扩展）div( … , x - x ) 且 x 是变量叶子以外的“同一引用”
            # 注：DEAP 的树节点是编号 + label，这里仅在 label 层面做快速筛；
            # 如果你后续想更严格，可以写一个“子树同构”比较，但这一步已经能挡住常见作弊。
        except Exception:
            # 安全兜底：结构解析失败就不在此处拦，交给后续编译/仿真逻辑处理
            pass
    
    # ====== 自抵消哨兵：卡掉 C_i 被代数消掉的典型模式 ======
    # 例如：sub(C1, sub(C1, g)) -> g   或   sub(sub(C1, g), C1) -> -g
    if True:  # 可加 cfg.hard_no_self_cancel 开关，默认 True
        try:
            children = {i: [] for i in nodes}
            for p, c in edges:
                children[p].append(c)
            VARS = ("C1", "C2", "C3")

            def is_var(node, name):
                return labels.get(node) == name

            for u in nodes:
                if labels[u] == "sub" and len(children[u]) == 2:
                    a, b = children[u]
                    # 1) sub(Ci, sub(Ci, *))
                    if labels.get(b) == "sub" and len(children[b]) == 2:
                        a1, a2 = children[b]
                        for name in VARS:
                            if is_var(a, name) and is_var(a1, name):
                                return ko("self_cancel_pattern")
                    # 2) sub(sub(Ci, *), Ci)
                    if labels.get(a) == "sub" and len(children[a]) == 2:
                        a1, a2 = children[a]
                        for name in VARS:
                            if is_var(a1, name) and is_var(b, name):
                                return ko("self_cancel_pattern")
        except Exception:
            pass

    # compile
    try:
        f = compile_individual(individual, pset)
    except Exception:
        return ko("runtime_compile_error")
    
    # ---- Shared sampling & arg getter (提前，供后续所有硬筛/检查复用) ----
    eps = cfg.grad_eps_rel
    arg_getter = make_arg_getter(("Q","C1","C2","C3"))

    # 记录抽样（统一在这里做一次），避免在后面出现“recs 未定义”的情况
    recs = dataset.records
    nrec = getattr(cfg, "hard_check_nrecords", 12)  # 你在 Config 里已有该字段；若无就按 12
    if nrec and len(recs) > nrec:
        import random as _rnd
        recs = _rnd.sample(recs, nrec)
    
    # ====== 翻转对比硬筛：每个 Ci 都必须真正影响 f ======
    # 不改你现有的 dQ 硬筛；这里只针对 C1/C2/C3。
    if True:  # 可加 cfg.hard_toggle_required 开关，默认 True
        tau = 1e-3       # 幅度门槛：|Δf| ≥ tau 视为“有效”
        min_hits = 3     # 命中次数下限（样本×Q点）
        delta = 0.4      # 翻转幅度（变量已归一到[0,1]时好用）
        eff_hits = {"C1": 0, "C2": 0, "C3": 0}

        for r in recs:
            Qmax = float(np.max(r.Q)) if r.Q.size else 1.0
            Q_grid = np.linspace(0.0, max(1.0, Qmax), cfg.grad_check_points)
            step = max(1, len(Q_grid)//5)
            Q_pick = Q_grid[::step]

            for name in ("C1", "C2", "C3"):
                base = dict(r.vars)
                v = base[name]
                v_lo = max(0.0, v - delta/2)
                v_hi = min(1.0, v + delta/2)  # 你的 C 已在读取时缩放到[0,1]

                r_lo = Record(r.t, r.Q, r.Q0, {**base, name: v_lo})
                r_hi = Record(r.t, r.Q, r.Q0, {**base, name: v_hi})

                for Q in Q_pick:
                    vlo = _sat(f(*arg_getter(Q, r_lo)))
                    vhi = _sat(f(*arg_getter(Q, r_hi)))
                    if abs(vhi - vlo) >= tau:
                        eff_hits[name] += 1

        for name in ("C1", "C2", "C3"):
            if eff_hits[name] < min_hits:
                return ko(f"toggle_{name}_fail")   # name ∈ {"C1","C2","C3"}

    # hard physics checks on small grid
    gradC_pen = 0.0
    gradC_mag_pen = 0.0
    if cfg.hard_nonneg or cfg.hard_dfdQ_nonpos or cfg.lambda_gradC>0:
        # 递减硬筛所需的阈值与命中点数（都可从 Config 里读）
        margin = getattr(cfg, "hard_dfdQ_margin", 1e-4)
        min_hits = getattr(cfg, "hard_dfdQ_min_hits", max(3, cfg.grad_check_points//2))

        # 默认方向先验
        sign_map = {"C1": -1, "C2": +1, "C3": +1}
        if cfg.gradC_signs: sign_map.update(cfg.gradC_signs)

        # ====== 硬筛：逐条记录、各自的 Q 网格 ======
        for r in recs:
            Qmax = float(np.max(r.Q)) if r.Q.size else 1.0
            Q_grid = np.linspace(0.0, max(1.0, Qmax), cfg.grad_check_points)

            # f >= 0
            if cfg.hard_nonneg:
                for Q in Q_grid:
                    try:
                        val = _sat(f(*arg_getter(Q, r)))
                    except Exception:
                        return ko("runtime_eval_error")
                    if val < -cfg.hard_tol:
                        return ko("hard_nonneg_violation")

            # df/dQ ≤ -margin ：本条记录的所有网格点都必须满足（更硬）
            if cfg.hard_dfdQ_nonpos:
                margin = cfg.hard_dfdQ_margin
                if getattr(cfg, "hard_check_all_points", True):
                    q_iter = Q_grid
                else:
                    start = int((1.0 - getattr(cfg, "hard_check_tail_frac", 0.5)) * len(Q_grid))
                    q_iter = Q_grid[start:]
                for Q in q_iter:
                    dQ = eps * max(1e-3, abs(Q))
                    vp = _sat(f(*arg_getter(Q + dQ, r)))
                    vm = _sat(f(*arg_getter(Q - dQ, r)))
                    dfdQ = (vp - vm) / (2*dQ)
                    if dfdQ > -margin:
                        return ko("dfdQ_hard_violation")
        
        # ====== 硬筛：配方变量“有效覆盖”（反幽灵变量） ======
        # 要求：每个 Ci 在若干 Q 网格点上都“能撬动 f”，即 |df/dCi| ≥ τ
        if getattr(cfg, "hard_gradC_required", False):
            tau = getattr(cfg, "hard_gradC_mag_min", 5e-3)
            min_hits = getattr(cfg, "hard_gradC_min_hits", max(3, cfg.grad_check_points//2))
            eff_hits = {"C1": 0, "C2": 0, "C3": 0}

            # 与前面硬筛相同的记录抽样 & 网格设置（recs/eps/arg_getter 已在上方定义）
            for r in recs:
                Qmax = float(np.max(r.Q)) if r.Q.size else 1.0
                Q_grid = np.linspace(0.0, max(1.0, Qmax), cfg.grad_check_points)

                for name in ("C1", "C2", "C3"):
                    base = dict(r.vars)
                    dC = eps * max(1e-3, abs(base[name]))
                    base[name] = r.vars[name] + dC
                    r_plus = Record(r.t, r.Q, r.Q0, base)
                    base[name] = r.vars[name] - dC
                    r_minus = Record(r.t, r.Q, r.Q0, base)

                    # 为了速度，仅抽 5 个网格点（与你 df/dQ 的检查粒度相近）
                    step = max(1, len(Q_grid)//5)
                    for Q in Q_grid[::step]:
                        vp = _sat(f(*arg_getter(Q, r_plus)))
                        vm = _sat(f(*arg_getter(Q, r_minus)))
                        dfdC = (vp - vm) / (2*dC)
                        if abs(dfdC) >= tau:
                            eff_hits[name] += 1

            # 命中不足 → 直接判大罚（与其它硬筛风格保持一致）
            for name in ("C1", "C2", "C3"):
                if eff_hits[name] < min_hits:
                    return ko(f"ghost_gradC_{name}")
                    
        # ====== 软先验：配方梯度方向（可选） ======
        if cfg.lambda_gradC > 0.0:
            for r in recs:
                Qmax = float(np.max(r.Q)) if r.Q.size else 1.0
                Q_grid = np.linspace(0.0, max(1.0, Qmax), cfg.grad_check_points)
                for name, sgn in sign_map.items():
                    base = dict(r.vars)
                    dC = eps * max(1e-3, abs(base[name]))
                    base[name] = r.vars[name] + dC
                    r_plus = Record(r.t, r.Q, r.Q0, base)
                    base[name] = r.vars[name] - dC
                    r_minus = Record(r.t, r.Q, r.Q0, base)
                    for Q in Q_grid[:: max(1, len(Q_grid)//5)]:
                        vp = _sat(f(*arg_getter(Q, r_plus)))
                        vm = _sat(f(*arg_getter(Q, r_minus)))
                        dfdC = (vp - vm) / (2*dC)
                        gradC_pen += max(0.0, -sgn*dfdC)**2
                        gradC_mag_pen += max(0.0, cfg.gradC_mag_min - abs(dfdC))**2

    # simulate and compute MSE + penalties
    mse_sum = 0.0; n_pts = 0
    clamp_pen = 0.0   # ← log1p 被钳到 <-0.999999 的次数累加
    comp_pen = cfg.alpha_complexity * float(len(individual))

    # multi-op nesting penalties
    nest_pen = 0.0
    if cfg.nest_penalties:
        chains = sameop_chain_lengths(individual, list(cfg.nest_penalties.keys()))
        for op, w in cfg.nest_penalties.items():
            L = chains.get(op, 1)
            if L > 1:
                nest_pen += w * (L - 1)**2

    qsens_pen = 0.0
    if getattr(cfg, "lambda_qsens", 0.0) > 0.0:
        for r in recs:
            Qmax = float(np.max(r.Q)) if r.Q.size else 1.0
            Q_grid = np.linspace(0.0, max(1.0, Qmax), cfg.grad_check_points)
            vals = [_sat(f(*arg_getter(Q, r))) for Q in Q_grid]
            m = np.mean(vals); sd = np.std(vals)
            ratio = sd / (abs(m) + 1e-8)
            qsens_pen += max(0.0, getattr(cfg, "qsens_min_ratio", 0.05) - ratio)**2

    
    # 结构罚：若根是 div 且“分母子树包含 Q”，加一笔 divQ_pen
    divQ_pen = 0.0
    try:
        nodes, edges, labels = gp.graph(individual)
        children = {i: [] for i in nodes}
        for p,c in edges: children[p].append(c)
        def hasQ(u):
            return labels[u]=="Q" or any(hasQ(v) for v in children[u])
        for u in nodes:
            if labels[u]=="div" and len(children[u])>=2:
                denom = children[u][1]
                if hasQ(denom):
                    divQ_pen += 1.0
    except Exception:
        pass

    for rec in dataset.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        # 清零本条样本的 log1p 钳制计数
        global _CLAMP_LOG1P
        _CLAMP_LOG1P = 0
        try:
            pred = simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        except Exception:
            return ko("runtime_simulate_error")
        if not np.all(np.isfinite(pred)): return ko("runtime_simulate_error")
        clamp_pen += float(_CLAMP_LOG1P)  # ← 本样本里被钳的总次数
        qnorm = dataset.q_scale if cfg.normalize_fitness else 1.0
        diff = (pred - rec.Q) / qnorm
        mse_sum += float(np.dot(diff, diff))
        n_pts += diff.size

    mse = mse_sum / max(1, n_pts)
    total = mse + comp_pen + nest_pen + cfg.lambda_gradC * gradC_pen + 1e-4*clamp_pen + cfg.lambda_divQ * divQ_pen + cfg.lambda_qsens * qsens_pen + cfg.lambda_gradC_mag * gradC_mag_pen
    return (total,)

# =========================
# Training loop
# =========================
def _dump_hard_ko(gen: int, cfg):
    every = getattr(cfg, "debug_ko_every", 50)  # 可在 Config 里加默认
    if (gen % every == 0) or (gen == getattr(cfg, "ngen", gen)):
        if HARD_KO:
            top = HARD_KO.most_common(12)
            msg = " ".join(f"{k}:{v}" for k, v in top)
            print(f"[HARD_KO up to gen {gen}] {msg}", flush=True)
            if getattr(cfg, "debug_ko_reset_each_dump", False):
                HARD_KO.clear()
        else:
            print(f"[HARD_KO up to gen {gen}] (no hard kills)", flush=True)

from pathlib import Path
import csv
def _save_hof_snapshot(hof, gen: int, cfg: Config, dataset: Dataset, pset):
    """
    将当前 HOF 的前 K 个个体落盘到 CSV：hof_top_gen{gen}.csv
    字段：rank, size, fitness_total, mse_train_norm, infix
    """
    # 目标目录
    if cfg.hof_snapshot_dir:
        out_dir = Path(cfg.hof_snapshot_dir)
    else:
        # 与你的主程序保持一致的默认目录
        out_dir = Path("Symbolic Differential/SymODE/artifacts/ivrt_pair")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 文件名：带世代号
    path = out_dir / f"hof_top_gen{gen}.csv"

    # 归一化尺度（与训练时一致）
    q_scale_train = dataset.q_scale if getattr(cfg, "normalize_fitness", True) else 1.0

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank","size","fitness_total","mse_train_norm","infix"])
        topk = min(len(hof), max(1, int(cfg.hof_snapshot_topk)))
        for k, ind in enumerate(hof[:topk], start=1):
            fit_total = float(ind.fitness.values[0])
            # 计算按训练集 q_scale 归一的 MSE（与 main 的导出一致）
            try:
                f_comp = compile_individual(ind, pset)
                sse, npts = 0.0, 0
                for rec in dataset.records:
                    qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
                    pred = simulate_series(f_comp, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
                    d = (pred - rec.Q) / max(1.0, q_scale_train)
                    sse += float(d @ d)
                    npts += d.size
                mse_train_norm = sse / max(1, npts)
            except Exception:
                mse_train_norm = float("nan")
            writer.writerow([k, len(ind), f"{fit_total:.6g}", f"{mse_train_norm:.6g}", to_infix_str(ind)[:500]])
    print(f"[snapshot] saved HOF top{topk} at gen {gen} -> {path}")

def train_symbolic_ode(dataset: Dataset, cfg: Config):
    pset = build_pset_from_config(cfg)
    toolbox = build_gp(cfg, pset)
    toolbox.register("evaluate", partial(eval_individual, pset=pset, dataset=dataset, cfg=cfg))

    # === 调试：清零硬淘汰计数 ===
    HARD_KO.clear()

    # === 评估方式：调试期强制串行 ===
    pool = None
    if getattr(cfg, "debug_serial_eval", False):
        toolbox.map = map
        cfg.n_jobs = 1
        print("[debug] Using SERIAL evaluation; HARD_KO will be visible.", flush=True)
    else:
        # parallel
        if cfg.n_jobs is None:
            import multiprocessing as mp
            cfg.n_jobs = max(1, mp.cpu_count() - 1)
        if cfg.n_jobs > 1:
            import multiprocessing as mp
            pool = mp.Pool(processes=cfg.n_jobs)
            toolbox.register("map", pool.map)
            print(f"[info] Using PARALLEL evaluation (n_jobs={cfg.n_jobs}); "
                  "HARD_KO from workers won't show in main.", flush=True)
        else:
            toolbox.map = map

    # # Parallel
    # pool = None
    # if cfg.n_jobs is None:
    #     import multiprocessing as mp
    #     cfg.n_jobs = max(1, mp.cpu_count()-1)
    # if cfg.n_jobs > 1:
    #     import multiprocessing as mp
    #     pool = mp.Pool(processes=cfg.n_jobs)
    #     toolbox.register("map", pool.map)

    pop = toolbox.population(n=cfg.pop_size)
    random.seed(cfg.seed)

    # stats
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(20)

    # 统计器（每代重置）
    cross_try = cross_ok = mut_try = mut_ok = 0
    it = trange(cfg.ngen, desc="Evolving", leave=True) if cfg.show_progress else range(cfg.ngen)
    for gen in it:
        offspring = tools.selTournament(pop, len(pop), tournsize=cfg.tournsize)
        offspring = list(map(toolbox.clone, offspring))

        for i in range(len(offspring)):
            if random.random() < cfg.cxpb:
                cross_try += 1
                j = random.randrange(len(offspring))
                if j == i:
                    j = (j + 1) % len(offspring)  # 避免同个体交叉
                if len(offspring[i]) > 1 and len(offspring[j]) > 1:
                    try:
                        offspring[i], offspring[j] = toolbox.mate(offspring[i], offspring[j])
                        del offspring[i].fitness.values
                        del offspring[j].fitness.values
                        cross_ok += 1
                    except (IndexError, ValueError):
                        pass

            if random.random() < cfg.mutpb:
                mut_try += 1
                try:
                    if random.random() < 0.5:
                        offspring[i], = toolbox.mutate(offspring[i])
                    else:
                        offspring[i], = toolbox.mutate_gauss(offspring[i])
                    del offspring[i].fitness.values
                    mut_ok += 1
                except (IndexError, ValueError):
                    pass
                
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        # 评估完毕，更新统计与日志
        fits_all = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
        if cfg.verbosity >= 2 and fits_all:
            print(f"[gen {gen+1:03d}] "
                  f"best={hof[0].fitness.values[0]:.6g} "
                  f"avg={np.mean(fits_all):.6g} "
                  f"std={np.std(fits_all):.6g} "
                  f"min={np.min(fits_all):.6g} "
                  f"size={len(hof[0])} "
                  f"cross {cross_ok}/{cross_try}, mut {mut_ok}/{mut_try}")
        _dump_hard_ko(gen, cfg)
        # === HOF 快照：每隔 hof_snapshot_every 代保存一次 ===
        if getattr(cfg, "hof_snapshot_every", 0):
            if (gen + 1) % cfg.hof_snapshot_every == 0:
                _save_hof_snapshot(hof, gen + 1, cfg, dataset, pset)

        if cfg.verbosity >= 1 and (gen+1) % cfg.log_best_every == 0:
            try:
                preview = to_infix_str(hof[0])
                print("[best expr]", preview[:180])
            except Exception:
                pass
            if cfg.nest_penalties:
                try:
                    chains = sameop_chain_lengths(hof[0], cfg.nest_penalties.keys())
                    print("[nesting]", chains)
                except Exception:
                    pass

        # 重置计数器（下一代）
        cross_try = cross_ok = mut_try = mut_ok = 0

        if (gen+1) % max(1, cfg.ngen//5) == 0:
            best = hof[0]
            print(f"[gen {gen+1:03d}] best = {best.fitness.values[0]:.6g}, size={len(best)}")

    if pool is not None:
        pool.close(); pool.join()
    
    # === 训练结束，最后再汇总一次 ===
    _dump_hard_ko(getattr(cfg, "ngen", -1), cfg)

    return hof[0], hof, stats

# =========================
# Exports
# =========================
def to_infix_str(individual) -> str:
    from deap import gp
    def fmt(name, args):
        if name=="add": return f"({args[0]} + {args[1]})"
        if name=="sub": return f"({args[0]} - {args[1]})"
        if name=="mul": return f"({args[0]} * {args[1]})"
        if name=="div": return f"({args[0]} / {args[1]})"
        if name=="log1p": return f"log(1+{args[0]})"
        if name=="exp": return f"exp({args[0]})"
        if name=="sqrt": return f"sqrt({args[0]})"
        if name=="softplus": return f"log(1+exp({args[0]}))"
        if name=="pow2": return f"({args[0]})**2"
        if name=="pow3": return f"({args[0]})**3"
        return f"{name}({', '.join(args)})"
    stack=[]
    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            txt = node.name if node.value is None else (str(node.value))
            stack.append(txt)
        else:
            ar = node.arity
            args = stack[-ar:]; stack[-ar:] = []
            stack.append(fmt(node.name, args))
    return stack[0]

def to_sympy(individual):
    import sympy as sp
    from deap import gp
    syms = {k: sp.Symbol(k) for k in ["Q","C1","C2","C3"]}
    def fmt(name, args):
        a = args
        if name=="add": return a[0] + a[1]
        if name=="sub": return a[0] - a[1]
        if name=="mul": return a[0] * a[1]
        if name=="div": return a[0] / a[1]
        if name=="log1p": return sp.log(1 + a[0])
        if name=="exp": return sp.exp(a[0])
        if name=="sqrt": return sp.sqrt(sp.Abs(a[0]))
        if name=="softplus": return sp.log(1 + sp.exp(a[0]))
        if name=="pow2": return a[0]**2
        if name=="pow3": return a[0]**3
        return sp.Function(name)(*a)
    stack=[]
    for node in reversed(individual):
        if isinstance(node, gp.Terminal):
            val = getattr(node, "value", None)
            # 把「变量」识别得更稳：名字在 {Q,C1,C2,C3} 或 value 是字符串 或 value 为 None
            is_var = (node.name in ("Q","C1","C2","C3")) or isinstance(val, str) or (val is None)
            if is_var:
                stack.append(syms.get(node.name, sp.Symbol(node.name)))
            else:
                stack.append(sp.Float(str(val)))
        else:
            ar = node.arity
            args = stack[-ar:]; stack[-ar:] = []
            stack.append(fmt(node.name, args))
    expr = stack[0]
    return expr, syms

# =========================
# Plotting / Export utils
# =========================
def predict_dataset(f, ds: Dataset, cfg: Config) -> List[np.ndarray]:
    """对整个数据集做前向，逐 Record 返回 yhat（与 rec.t 对齐），与训练 evaluate 完全同构。"""
    preds = []
    for rec in ds.records:
        qcap = max(1.0, cfg.qcap_factor * float(np.max(rec.Q))) if rec.Q.size else 1.0
        yhat = simulate_series(f, rec, rec.t, Q0=rec.Q0, cfg=cfg, qcap=qcap)
        preds.append(yhat)
    return preds

def export_predictions_csv(ds: Dataset, preds: List[np.ndarray], csv_path):
    """导出逐样本逐时间点的观测/预测（方便做外部敏感度/回归诊断）。"""
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_idx", "time_h", "Q_obs", "Q_pred"])
        for i, rec in enumerate(ds.records):
            yhat = preds[i]
            for t, y, yh in zip(rec.t, rec.Q, yhat):
                w.writerow([i, float(t), float(y), float(yh)])

# def plot_scatter(ds: Dataset, preds: List[np.ndarray], png_path):
#     """观测 vs 预测散点（所有点拼在一起），y=x 参考线。"""
#     import numpy as np, matplotlib.pyplot as plt
#     y = np.concatenate([rec.Q for rec in ds.records])
#     yhat = np.concatenate(preds)
#     lim_hi = float(max(1.0, np.nanmax(y), np.nanmax(yhat)))
#     plt.figure(figsize=(5,5))
#     plt.scatter(y, yhat, s=8, alpha=0.6)
#     plt.plot([0,lim_hi],[0,lim_hi], "--")
#     plt.xlabel("Observed Q (μg/cm²)")
#     plt.ylabel("Predicted Q (μg/cm²)")
#     plt.title("Observed vs Predicted")
#     plt.tight_layout()
#     plt.savefig(png_path, dpi=160)
#     plt.close()

# def plot_series(ds: Dataset, preds: List[np.ndarray], png_path, max_examples: int = 6):
#     """抽样若干 Record 的时间序列对比图。"""
#     import numpy as np, matplotlib.pyplot as plt
#     n = min(max_examples, len(ds.records))
#     idx = np.linspace(0, len(ds.records)-1, n, dtype=int)
#     plt.figure(figsize=(9,6))
#     for i in idx:
#         rec = ds.records[i]
#         plt.plot(rec.t, rec.Q, "o-", label=f"obs#{i}", alpha=0.9)
#         plt.plot(rec.t, preds[i], "--", label=f"pred#{i}", alpha=0.9)
#     plt.xlabel("Time (h)"); plt.ylabel("Q (μg/cm²)")
#     plt.title("Time series (sampled)")
#     plt.legend(ncol=2, fontsize=8)
#     plt.tight_layout()
#     plt.savefig(png_path, dpi=160)
#     plt.close()

def plot_scatter(ds: Dataset, preds: List[np.ndarray], png_path):
    """观测 vs 预测散点（所有点拼在一起），y=x 参考线。"""
    import numpy as np, matplotlib.pyplot as plt
    y = np.concatenate([rec.Q for rec in ds.records])
    yhat = np.concatenate(preds)
    # 仅保留有效点
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        raise ValueError("No finite points to plot")
    y = y[mask]; yhat = yhat[mask]

    # 评价指标：RMSE, R2
    mse = float(np.mean((y - yhat) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    lim_hi = float(max(1.0, np.nanmax(y), np.nanmax(yhat)))
    plt.figure(figsize=(5,5))
    plt.scatter(y, yhat, s=8, alpha=0.6)
    plt.plot([0,lim_hi],[0,lim_hi], "--")
    plt.xlabel("Observed Q (μg/cm²)")
    plt.ylabel("Predicted Q (μg/cm²)")
    plt.title("Observed vs Predicted")
    # put metrics on the plot
    txt = f"R² = {r2:.3f}\nRMSE = {rmse:.3g}"
    plt.gca().text(0.98, 0.02, txt, ha="right", va="bottom", transform=plt.gca().transAxes,
                   fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

def plot_series(ds: Dataset, preds: List[np.ndarray], png_path, max_examples: int = 6):
    """抽样若干 Record 的时间序列对比图。"""
    import numpy as np, matplotlib.pyplot as plt
    n = min(max_examples, len(ds.records))
    idx = np.linspace(0, len(ds.records)-1, n, dtype=int)
    plt.figure(figsize=(9,6))
    # collect all points to compute global metrics
    all_y = []
    all_yhat = []
    for i in idx:
        rec = ds.records[i]
        all_y.append(rec.Q)
        all_yhat.append(preds[i])
        plt.plot(rec.t, rec.Q, "o-", label=f"obs#{i}", alpha=0.9)
        plt.plot(rec.t, preds[i], "--", label=f"pred#{i}", alpha=0.9)
    if all_y:
        y = np.concatenate(all_y)
        yhat = np.concatenate(all_yhat)
        mask = np.isfinite(y) & np.isfinite(yhat)
        if np.any(mask):
            y = y[mask]; yhat = yhat[mask]
            mse = float(np.mean((y - yhat) ** 2))
            rmse = float(np.sqrt(mse))
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
            txt = f"R² = {r2:.3f}    RMSE = {rmse:.3g}"
            plt.gca().text(0.98, 0.02, txt, ha="right", va="bottom", transform=plt.gca().transAxes,
                           fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    plt.xlabel("Time (h)"); plt.ylabel("Q (μg/cm²)")
    plt.title("Time series (sampled)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()