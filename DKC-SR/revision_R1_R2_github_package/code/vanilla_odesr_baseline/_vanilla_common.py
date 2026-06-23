from __future__ import annotations

import json
import math
import random
import re
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from deap import base, creator, gp, tools
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_validation_vanilla_odesr_24train6test"
CONFIG_DIR = OUT / "config"
RESULTS_DIR = OUT / "results"
REPORT_DIR = OUT / "reports"
FIGURES_DIR = OUT / "figures"
RUNS_DIR = OUT / "runs"

SOURCE_24_DATA = ROOT / "revision_validation_24train6test" / "data"
SOURCE_DKCSR = ROOT / "revision_validation_24train6test_dkcsr"
SOURCE_STATIC = ROOT / "revision_validation_24train6test"
SOURCE_UCSR = ROOT / "revision_validation_unconstrained_sr_24train6test"
SOURCE_UCSR_STRUCT = ROOT / "revision_validation_unconstrained_sr_structural_audit"
SELECTED_ARTIFACT_DIR = ROOT / "artifacts" / "archive" / "ivrt-pair-251007"
BEST_SYMPY = SELECTED_ARTIFACT_DIR / "best_sympy.txt"

Q_SCALE = 3008.198194823261
TIME_POINTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
C_BOUNDS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}
VAR_NAMES = ("Q", "C1", "C2", "C3")
FORMULATION_VARS = ("C1", "C2", "C3")
PRIMITIVE_NAMES = ("add", "sub", "mul", "div", "pow2")
SAT = 1e6
DIV_MIN = 1e-8


@dataclass
class VanillaConfig:
    mode: str
    seeds: list[int]
    pop_size: int
    ngen: int
    hall_of_fame_size: int
    cxpb: float = 0.6
    mutpb: float = 0.4
    tournsize: int = 5
    tree_len_max: int = 25
    init_depth_min: int = 1
    init_depth_max: int = 4
    ephemeral_range: tuple[float, float] = (-2.0, 2.0)
    substeps: int = 8
    adapt_refine_max: int = 6
    dt_floor: float = 1e-6
    qcap_factor: float = 1.5
    alpha_complexity: float = 5e-4
    primitive_names: tuple[str, ...] = PRIMITIVE_NAMES


def ensure_dirs() -> None:
    for path in [OUT, CONFIG_DIR, RESULTS_DIR, REPORT_DIR, FIGURES_DIR, RUNS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def default_config_dict() -> dict[str, Any]:
    return {
        "q_scale": Q_SCALE,
        "q_scale_source": "fixed by vanilla ODE-SR task; artifact q_scale is not read or used",
        "dataset_files": {
            "curve_train": rel(SOURCE_24_DATA / "canonical_curve_train_24.csv"),
            "curve_test": rel(SOURCE_24_DATA / "canonical_curve_test_6.csv"),
            "endpoint_train": rel(SOURCE_24_DATA / "canonical_endpoint_train_24.csv"),
            "endpoint_test": rel(SOURCE_24_DATA / "canonical_endpoint_test_6.csv"),
        },
        "excluded_rows_not_used": ["S10", "Opt-2", "Opt-4", "Opt-6", "Opt-7", "Opt-10"],
        "variables_available": list(VAR_NAMES),
        "primitive_names": list(PRIMITIVE_NAMES),
        "softplus_absent": True,
        "variable_inclusion_constraint": "none; variable retention is audited after fitting",
        "retained_components": [
            "Q, C1, C2, C3 as available variables",
            "ODE replay framework",
            "same 24 train + 6 held-out test split",
            "fixed q_scale = 3008.198194823261",
            "protected division",
            "finite-value failure handling",
            "RK4 integration with substeps",
            "complexity penalty",
            "tree-length limit",
        ],
        "removed_domain_knowledge_components": [
            "softplus primitive",
            "hard nonnegative RHS",
            "hard nonpositive df/dQ",
            "formulation-gradient sign prior",
            "gradient magnitude/effect filters",
            "Q-sensitivity penalty",
            "denominator-Q physical penalty",
            "nonnegative release prediction clamp",
        ],
        "full": {
            "mode": "full",
            "seeds": [0, 1, 2, 3, 4],
            "pop_size": 300,
            "ngen": 30,
            "hall_of_fame_size": 20,
        },
        "dryrun": {
            "mode": "dryrun",
            "seeds": [0],
            "pop_size": 80,
            "ngen": 3,
            "hall_of_fame_size": 10,
        },
        "gp": {
            "cxpb": 0.6,
            "mutpb": 0.4,
            "tournsize": 5,
            "tree_len_max": 25,
            "init_depth_min": 1,
            "init_depth_max": 4,
            "ephemeral_range": [-2.0, 2.0],
            "substeps": 8,
            "adapt_refine_max": 6,
            "dt_floor": 1e-6,
            "qcap_factor": 1.5,
            "alpha_complexity": 5e-4,
        },
        "default_run_mode": "full",
    }


def config_from_dict(cfg: dict[str, Any], mode: str) -> VanillaConfig:
    block = cfg[mode]
    gp_block = cfg.get("gp", {})
    return VanillaConfig(
        mode=str(block["mode"]),
        seeds=list(map(int, block["seeds"])),
        pop_size=int(block["pop_size"]),
        ngen=int(block["ngen"]),
        hall_of_fame_size=int(block["hall_of_fame_size"]),
        cxpb=float(gp_block.get("cxpb", 0.6)),
        mutpb=float(gp_block.get("mutpb", 0.4)),
        tournsize=int(gp_block.get("tournsize", 5)),
        tree_len_max=int(gp_block.get("tree_len_max", 25)),
        init_depth_min=int(gp_block.get("init_depth_min", 1)),
        init_depth_max=int(gp_block.get("init_depth_max", 4)),
        ephemeral_range=tuple(gp_block.get("ephemeral_range", [-2.0, 2.0])),
        substeps=int(gp_block.get("substeps", 8)),
        adapt_refine_max=int(gp_block.get("adapt_refine_max", 6)),
        dt_floor=float(gp_block.get("dt_floor", 1e-6)),
        qcap_factor=float(gp_block.get("qcap_factor", 1.5)),
        alpha_complexity=float(gp_block.get("alpha_complexity", 5e-4)),
    )


def load_config(mode: str | None = None) -> VanillaConfig:
    path = CONFIG_DIR / "vanilla_odesr_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} not found. Run 00_prepare_vanilla_odesr_config.py first.")
    cfg = read_json(path)
    selected = mode or cfg.get("default_run_mode", "full")
    return config_from_dict(cfg, selected)


def load_canonical() -> dict[str, pd.DataFrame]:
    return {
        "curve_train": pd.read_csv(SOURCE_24_DATA / "canonical_curve_train_24.csv"),
        "curve_test": pd.read_csv(SOURCE_24_DATA / "canonical_curve_test_6.csv"),
        "endpoint_train": pd.read_csv(SOURCE_24_DATA / "canonical_endpoint_train_24.csv"),
        "endpoint_test": pd.read_csv(SOURCE_24_DATA / "canonical_endpoint_test_6.csv"),
    }


def records_from_curve(curve: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, g in curve.groupby(["dataset", "record_id"], sort=False):
        gg = g.sort_values("time_h").reset_index(drop=True)
        records.append(
            {
                "dataset": str(gg["dataset"].iloc[0]),
                "record_id": str(gg["record_id"].iloc[0]),
                "record_index": int(gg["record_index"].iloc[0]),
                "run_no": str(gg["run_no"].iloc[0]),
                "time_h": gg["time_h"].to_numpy(float),
                "Q_obs": gg["Q_obs"].to_numpy(float),
                "Qtilde_obs": gg["Q_obs"].to_numpy(float) / Q_SCALE,
                "C1": float(gg["C1"].iloc[0]),
                "C2": float(gg["C2"].iloc[0]),
                "C3": float(gg["C3"].iloc[0]),
                "C1n": float(gg["C1n"].iloc[0]),
                "C2n": float(gg["C2n"].iloc[0]),
                "C3n": float(gg["C3n"].iloc[0]),
            }
        )
    return records


def _sat(x: float, lo: float = -SAT, hi: float = SAT) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return min(max(x, lo), hi)


def p_add(a, b):
    return _sat(a + b)


def p_sub(a, b):
    return _sat(a - b)


def p_mul(a, b):
    return _sat(_sat(a) * _sat(b))


def p_div(a, b):
    a = _sat(a)
    b = _sat(b)
    if abs(b) < DIV_MIN:
        return 0.0
    return _sat(a / b)


def p_pow2(a):
    a = _sat(a)
    return _sat(a * a)


PRIMS: dict[str, tuple[Callable[..., float], int]] = {
    "add": (p_add, 2),
    "sub": (p_sub, 2),
    "mul": (p_mul, 2),
    "div": (p_div, 2),
    "pow2": (p_pow2, 1),
}


def build_pset(cfg: VanillaConfig):
    pset = gp.PrimitiveSet("MAIN", len(VAR_NAMES))
    for idx, name in enumerate(VAR_NAMES):
        pset.renameArguments(**{f"ARG{idx}": name})
    for name in cfg.primitive_names:
        fn, arity = PRIMS[name]
        pset.addPrimitive(fn, arity, name=name)
    low, high = cfg.ephemeral_range
    pset.addEphemeralConstant("C", partial(random.uniform, low, high))
    return pset


def ensure_deap_types() -> None:
    if "FitnessMin_VODE" not in creator.__dict__:
        creator.create("FitnessMin_VODE", base.Fitness, weights=(-1.0,))
    if "Individual_VODE" not in creator.__dict__:
        creator.create("Individual_VODE", gp.PrimitiveTree, fitness=creator.FitnessMin_VODE)


def compile_expr(expr_text: str, cfg: VanillaConfig | None = None):
    cfg = cfg or VanillaConfig(mode="eval", seeds=[], pop_size=0, ngen=0, hall_of_fame_size=0)
    pset = build_pset(cfg)
    tree = gp.PrimitiveTree.from_string(expr_text, pset)
    return gp.compile(tree, pset), tree, pset


def simulate_record(
    f: Callable[[float, float, float, float], float],
    rec: dict[str, Any],
    cfg: VanillaConfig,
    qcap_norm: float | None = None,
) -> np.ndarray:
    t = np.asarray(rec["time_h"], dtype=float)
    out = np.empty_like(t, dtype=float)
    q = float(rec["Qtilde_obs"][0])
    c1, c2, c3 = float(rec["C1n"]), float(rec["C2n"]), float(rec["C3n"])
    if qcap_norm is None:
        qcap_norm = max(1.0, cfg.qcap_factor * float(np.nanmax(np.abs(rec["Qtilde_obs"]))))
    qcap_norm = max(1.0, float(qcap_norm))
    out[0] = q
    for k in range(1, len(t)):
        dt_total = float(t[k] - t[k - 1])
        if dt_total <= 0:
            out[k] = q
            continue
        ok = False
        for refine in range(cfg.adapt_refine_max + 1):
            n_steps = cfg.substeps * (2**refine)
            dt = max(dt_total / n_steps, cfg.dt_floor)
            q_try = q
            finite = True
            for _ in range(n_steps):
                try:
                    k1 = _sat(f(q_try, c1, c2, c3))
                    k2 = _sat(f(q_try + 0.5 * dt * k1, c1, c2, c3))
                    k3 = _sat(f(q_try + 0.5 * dt * k2, c1, c2, c3))
                    k4 = _sat(f(q_try + dt * k3, c1, c2, c3))
                    q_try = _sat(q_try + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), -qcap_norm, qcap_norm)
                except Exception:
                    finite = False
                    break
                if not math.isfinite(q_try):
                    finite = False
                    break
            if finite:
                q = q_try
                ok = True
                break
        if not ok:
            return np.full_like(t, np.nan, dtype=float)
        out[k] = q
    return out


def eval_individual(individual, pset, records: list[dict[str, Any]], cfg: VanillaConfig):
    try:
        f = gp.compile(individual, pset)
    except Exception:
        return (10.0,)
    sse = 0.0
    n = 0
    failure = 0
    for rec in records:
        pred = simulate_record(f, rec, cfg)
        obs = np.asarray(rec["Qtilde_obs"], dtype=float)
        if pred.shape != obs.shape or not np.all(np.isfinite(pred)):
            failure += 1
            continue
        diff = pred - obs
        sse += float(diff @ diff)
        n += int(diff.size)
    if n == 0:
        return (10.0 + failure,)
    return (sse / n + cfg.alpha_complexity * float(len(individual)) + 0.1 * failure,)


def build_toolbox(cfg: VanillaConfig, pset, train_records: list[dict[str, Any]]):
    ensure_deap_types()
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=cfg.init_depth_min, max_=cfg.init_depth_max)
    toolbox.register("individual", tools.initIterate, creator.Individual_VODE, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=cfg.tree_len_max))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=cfg.tree_len_max))
    toolbox.register("evaluate", eval_individual, pset=pset, records=train_records, cfg=cfg)
    return toolbox


def train_seed(seed: int, cfg: VanillaConfig, train_records: list[dict[str, Any]]):
    random.seed(seed)
    np.random.seed(seed)
    pset = build_pset(cfg)
    toolbox = build_toolbox(cfg, pset, train_records)
    pop = toolbox.population(n=cfg.pop_size)
    hof = tools.HallOfFame(cfg.hall_of_fame_size, similar=lambda a, b: str(a) == str(b))
    stats_rows: list[dict[str, Any]] = []
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    hof.update(pop)
    for gen in range(1, cfg.ngen + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for i in range(1, len(offspring), 2):
            if random.random() < cfg.cxpb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values
        for i in range(len(offspring)):
            if random.random() < cfg.mutpb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        pop[:] = offspring
        hof.update(pop)
        fits = [float(ind.fitness.values[0]) for ind in pop if ind.fitness.valid]
        if fits:
            stats_rows.append(
                {
                    "generation": gen,
                    "min_fitness": float(np.min(fits)),
                    "mean_fitness": float(np.mean(fits)),
                    "std_fitness": float(np.std(fits)),
                    "hof_best_fitness": float(hof[0].fitness.values[0]) if len(hof) else np.nan,
                    "hof_best_size": int(len(hof[0])) if len(hof) else 0,
                }
            )
    return hof[0], hof, pd.DataFrame(stats_rows), pset


def individual_to_sympy(individual) -> sp.Expr:
    iterator = iter(individual)
    symbols = {name: sp.Symbol(name) for name in VAR_NAMES}

    def rec():
        node = next(iterator)
        if isinstance(node, gp.Primitive):
            args = [rec() for _ in range(node.arity)]
            if node.name == "add":
                return args[0] + args[1]
            if node.name == "sub":
                return args[0] - args[1]
            if node.name == "mul":
                return args[0] * args[1]
            if node.name == "div":
                return args[0] / args[1]
            if node.name == "pow2":
                return args[0] ** 2
            return sp.Function(node.name)(*args)
        value = getattr(node, "value", None)
        name = getattr(node, "name", "")
        if name in symbols:
            return symbols[name]
        if isinstance(value, str) and value in symbols:
            return symbols[value]
        try:
            return sp.Float(float(value), 12)
        except Exception:
            return sp.Symbol(str(name))

    return sp.simplify(rec())


def expression_complexity(expr: str) -> dict[str, Any]:
    operators = re.findall(r"\b(add|sub|mul|div|pow2)\s*\(", expr)
    constants = re.findall(r"(?<![A-Za-z_])-?\d+\.\d+(?:e[+-]?\d+)?", expr, flags=re.I)
    try:
        _, tree, _ = compile_expr(expr)
        depth = int(tree.height)
        size = int(len(tree))
    except Exception:
        depth = np.nan
        size = np.nan
    return {
        "expression_length": len(expr),
        "number_of_operators": len(operators),
        "tree_depth": depth,
        "tree_size": size,
        "number_of_constants": len(constants),
    }


def predictions_for_expr(expr: str, curve: pd.DataFrame, dataset_name: str, model_label: str, seed: int | str = "") -> tuple[pd.DataFrame, int]:
    f, _, _ = compile_expr(expr)
    cfg = VanillaConfig(mode="eval", seeds=[], pop_size=0, ngen=0, hall_of_fame_size=0)
    records = records_from_curve(curve)
    rows: list[dict[str, Any]] = []
    failures = 0
    for rec in records:
        pred_norm = simulate_record(f, rec, cfg)
        if not np.all(np.isfinite(pred_norm)):
            failures += 1
        for t, q_obs, q_pred_norm in zip(rec["time_h"], rec["Q_obs"], pred_norm):
            rows.append(
                {
                    "model": model_label,
                    "seed": seed,
                    "dataset": dataset_name,
                    "record_id": rec["record_id"],
                    "record_index": rec["record_index"],
                    "run_no": rec["run_no"],
                    "time_h": float(t),
                    "Q_obs": float(q_obs),
                    "Q_pred": float(q_pred_norm * Q_SCALE) if np.isfinite(q_pred_norm) else np.nan,
                    "C1": rec["C1"],
                    "C2": rec["C2"],
                    "C3": rec["C3"],
                    "C1n": rec["C1n"],
                    "C2n": rec["C2n"],
                    "C3n": rec["C3n"],
                }
            )
    return pd.DataFrame(rows), failures


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else np.nan
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, q_scale: float | None = None) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "MSE": np.nan, "MSE_normalized_by_q_scale": np.nan}
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": r2_score_safe(y_true, y_pred),
        "MSE": mse,
        "MSE_normalized_by_q_scale": float(mse / (q_scale**2)) if q_scale else np.nan,
    }


def curve_metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, dataset), g in pred.groupby(["model", "seed", "dataset"], dropna=False, sort=False):
        for scope, sub in [("overall_excluding_t0", g.loc[~np.isclose(g["time_h"], 0.0)]), ("overall_including_t0", g)]:
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "dataset": dataset,
                    "scope": scope,
                    "time_h": "",
                    "n_points": int(len(sub)),
                    **regression_metrics(sub["Q_obs"].to_numpy(float), sub["Q_pred"].to_numpy(float), q_scale=Q_SCALE),
                }
            )
    return pd.DataFrame(rows)


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int]:
    correct = 0
    total = 0
    for i in range(len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            st = np.sign(y_true[i] - y_true[j])
            sp = np.sign(y_pred[i] - y_pred[j])
            if st == 0 or sp == 0:
                continue
            total += 1
            correct += int(st == sp)
    return (float(correct / total) if total else np.nan, correct, total)


def top_hit_sets(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    order_true = np.argsort(-np.asarray(y_true, dtype=float))
    order_pred = np.argsort(-np.asarray(y_pred, dtype=float))
    k2 = min(2, len(labels))
    true_top1 = [labels[int(order_true[0])]] if labels else []
    pred_top1 = [labels[int(order_pred[0])]] if labels else []
    true_top2 = [labels[int(i)] for i in order_true[:k2]]
    pred_top2 = [labels[int(i)] for i in order_pred[:k2]]
    return {
        "top1_hit": int(bool(true_top1) and true_top1[0] == pred_top1[0]),
        "top2_hit": int(bool(set(true_top2) & set(pred_top2))),
        "true_top1": json.dumps(true_top1),
        "pred_top1": json.dumps(pred_top1),
        "true_top2": json.dumps(true_top2),
        "pred_top2": json.dumps(pred_top2),
    }


def rank_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    with np.errstate(all="ignore"):
        rho = spearmanr(y_true, y_pred).correlation if len(y_true) >= 2 else np.nan
        tau = kendalltau(y_true, y_pred).correlation if len(y_true) >= 2 else np.nan
    pair_acc, pair_correct, pair_total = pairwise_accuracy(y_true, y_pred)
    return {
        "Spearman": float(rho) if rho is not None else np.nan,
        "Kendall": float(tau) if tau is not None else np.nan,
        "pairwise_accuracy": pair_acc,
        "pairwise_correct": pair_correct,
        "pairwise_total": pair_total,
        **top_hit_sets(y_true, y_pred, labels),
    }


def endpoint_from_curve(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, dataset, record_id, run_no), g in pred.groupby(["model", "seed", "dataset", "record_id", "run_no"], dropna=False, sort=False):
        gg = g.sort_values("time_h")
        first = gg.iloc[0]
        q6_obs = gg.loc[np.isclose(gg["time_h"], 6.0), "Q_obs"].iloc[0]
        q6_pred = gg.loc[np.isclose(gg["time_h"], 6.0), "Q_pred"].iloc[0]
        rows.append(
            {
                "model": model,
                "seed": seed,
                "dataset": dataset,
                "record_id": record_id,
                "record_index": int(first["record_index"]),
                "run_no": run_no,
                "C1": float(first["C1"]),
                "C2": float(first["C2"]),
                "C3": float(first["C3"]),
                "C1n": float(first["C1n"]),
                "C2n": float(first["C2n"]),
                "C3n": float(first["C3n"]),
                "Q6_obs": float(q6_obs),
                "Q6_pred": float(q6_pred),
                "AUC_obs": float(np.trapz(gg["Q_obs"].to_numpy(float), gg["time_h"].to_numpy(float))),
                "AUC_pred": float(np.trapz(gg["Q_pred"].to_numpy(float), gg["time_h"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows)


def endpoint_metrics_table(endpoint_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, dataset), g in endpoint_pred.groupby(["model", "seed", "dataset"], dropna=False, sort=False):
        labels = g["run_no"].astype(str).tolist()
        row: dict[str, Any] = {"model": model, "seed": seed, "dataset": dataset, "n_formulations": int(len(g))}
        for prefix, obs_col, pred_col in [("Q6", "Q6_obs", "Q6_pred"), ("AUC", "AUC_obs", "AUC_pred")]:
            reg = regression_metrics(g[obs_col].to_numpy(float), g[pred_col].to_numpy(float), q_scale=Q_SCALE if prefix == "Q6" else None)
            rank = rank_metrics(g[obs_col].to_numpy(float), g[pred_col].to_numpy(float), labels)
            row[f"RMSE_{prefix}"] = reg["RMSE"]
            row[f"MAE_{prefix}"] = reg["MAE"]
            row[f"R2_{prefix}"] = reg["R2"]
            row[f"Spearman_{prefix}"] = rank["Spearman"]
            row[f"Kendall_{prefix}"] = rank["Kendall"]
            row[f"pairwise_accuracy_{prefix}"] = rank["pairwise_accuracy"]
            row[f"pairwise_correct_{prefix}"] = rank["pairwise_correct"]
            row[f"pairwise_total_{prefix}"] = rank["pairwise_total"]
            row[f"top1_hit_{prefix}"] = rank["top1_hit"]
            row[f"top2_hit_{prefix}"] = rank["top2_hit"]
            row[f"true_top1_{prefix}"] = rank["true_top1"]
            row[f"pred_top1_{prefix}"] = rank["pred_top1"]
            row[f"true_top2_{prefix}"] = rank["true_top2"]
            row[f"pred_top2_{prefix}"] = rank["pred_top2"]
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    view = df if max_rows is None else df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                val = f"{float(val):.6g}" if np.isfinite(val) else "nan"
            vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def get_successful_seed_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return [
        path
        for path in sorted(RUNS_DIR.glob("seed_*"), key=lambda p: int(p.name.split("_")[-1]))
        if (path / "best_expression_infix.txt").exists() and not (path / "run_failed.json").exists()
    ]


def make_grid(n: int) -> pd.DataFrame:
    rows = []
    grid_id = 0
    for c1 in np.linspace(C_BOUNDS["C1"][0], C_BOUNDS["C1"][1], n):
        for c2 in np.linspace(C_BOUNDS["C2"][0], C_BOUNDS["C2"][1], n):
            for c3 in np.linspace(C_BOUNDS["C3"][0], C_BOUNDS["C3"][1], n):
                c1n = (c1 - C_BOUNDS["C1"][0]) / (C_BOUNDS["C1"][1] - C_BOUNDS["C1"][0])
                c2n = (c2 - C_BOUNDS["C2"][0]) / (C_BOUNDS["C2"][1] - C_BOUNDS["C2"][0])
                c3n = (c3 - C_BOUNDS["C3"][0]) / (C_BOUNDS["C3"][1] - C_BOUNDS["C3"][0])
                for t in TIME_POINTS:
                    rows.append({"grid_id": grid_id, "C1": c1, "C2": c2, "C3": c3, "C1n": c1n, "C2n": c2n, "C3n": c3n, "time_h": t})
                grid_id += 1
    return pd.DataFrame(rows)


def grid_records(grid: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for gid, g in grid.groupby("grid_id", sort=True):
        gg = g.sort_values("time_h")
        records.append(
            {
                "record_index": int(gid),
                "run_no": f"grid_{int(gid):05d}",
                "time_h": gg["time_h"].to_numpy(float),
                "Q_obs": np.zeros(len(gg), dtype=float),
                "Qtilde_obs": np.zeros(len(gg), dtype=float),
                "C1": float(gg["C1"].iloc[0]),
                "C2": float(gg["C2"].iloc[0]),
                "C3": float(gg["C3"].iloc[0]),
                "C1n": float(gg["C1n"].iloc[0]),
                "C2n": float(gg["C2n"].iloc[0]),
                "C3n": float(gg["C3n"].iloc[0]),
            }
        )
    return records


def grid_predictions_for_model(label: str, seed: Any, f: Callable, grid: pd.DataFrame, qcap_norm: float) -> pd.DataFrame:
    cfg = VanillaConfig(mode="grid", seeds=[], pop_size=0, ngen=0, hall_of_fame_size=0)
    rows = []
    for rec in grid_records(grid):
        pred_norm = simulate_record(f, rec, cfg, qcap_norm=qcap_norm)
        for t, qn in zip(rec["time_h"], pred_norm):
            rows.append(
                {
                    "model": label,
                    "seed": seed,
                    "grid_id": rec["record_index"],
                    "C1": rec["C1"],
                    "C2": rec["C2"],
                    "C3": rec["C3"],
                    "C1n": rec["C1n"],
                    "C2n": rec["C2n"],
                    "C3n": rec["C3n"],
                    "time_h": float(t),
                    "Q_pred": float(qn * Q_SCALE) if np.isfinite(qn) else np.nan,
                    "Qtilde_pred": float(qn) if np.isfinite(qn) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def qcap_norm_from_observed() -> tuple[float, float]:
    canon = load_canonical()
    endpoints = pd.concat([canon["endpoint_train"], canon["endpoint_test"]], ignore_index=True)
    observed_q6_upper = 1.5 * float(endpoints["Q6_obs"].max())
    return max(10.0, 2.0 * observed_q6_upper / Q_SCALE), observed_q6_upper


def is_boundary(c1: float, c2: float, c3: float) -> bool:
    return any(
        np.isclose(v, C_BOUNDS[k][0]) or np.isclose(v, C_BOUNDS[k][1])
        for k, v in {"C1": c1, "C2": c2, "C3": c3}.items()
    )


def variables_in_text(text: str) -> list[str]:
    found = {var for var in VAR_NAMES if re.search(rf"\b{re.escape(var)}\b", text)}
    for arg, var in {"ARG0": "Q", "ARG1": "C1", "ARG2": "C2", "ARG3": "C3"}.items():
        if re.search(rf"\b{arg}\b", text):
            found.add(var)
    return sorted(found, key=VAR_NAMES.index)


def simplify_expression(text: str) -> tuple[str, list[str]]:
    locals_dict = {name: sp.Symbol(name) for name in VAR_NAMES}
    try:
        expr = sp.sympify(text, locals=locals_dict)
        simplified = sp.simplify(expr)
        present = sorted([str(s) for s in simplified.free_symbols if str(s) in VAR_NAMES], key=VAR_NAMES.index)
        return str(simplified), present
    except Exception:
        return "", variables_in_text(text)


def dkc_callable() -> Callable[[float, float, float, float], float]:
    expr = BEST_SYMPY.read_text(encoding="utf-8").strip()
    normalized = expr
    for old, new in {"ARG0": "Q", "ARG1": "C1", "ARG2": "C2", "ARG3": "C3"}.items():
        normalized = re.sub(rf"\b{old}\b", new, normalized)
    code = compile(normalized, "<dkcsr_selected_qscale3008>", "eval")

    def safe_exp(x):
        return math.exp(max(-50.0, min(50.0, _sat(x))))

    safe_env = {"__builtins__": {}, "log": math.log, "exp": safe_exp, "pow": pow}

    def f(q, c1, c2, c3):
        c3s = float(c3)
        if abs(c3s) < DIV_MIN:
            c3s = DIV_MIN if c3s >= 0 else -DIV_MIN
        return float(eval(code, safe_env, {"Q": max(0.0, float(q)), "C1": float(c1), "C2": float(c2), "C3": c3s}))

    return f


def finite_dfdq(f: Callable, q: float, c1: float, c2: float, c3: float) -> float:
    eps = 1e-5 * max(1.0, abs(float(q)))
    try:
        return float((f(q + eps, c1, c2, c3) - f(q - eps, c1, c2, c3)) / (2.0 * eps))
    except Exception:
        return np.nan


def safety_rows_from_grid(grid_pred: pd.DataFrame, observed_q6_upper: float, f_lookup: dict[tuple[str, Any], Callable]) -> pd.DataFrame:
    rows = []
    for (model, seed), g in grid_pred.groupby(["model", "seed"], dropna=False, sort=False):
        q = g["Q_pred"].to_numpy(float)
        finite = np.isfinite(q)
        negative_q_rate = float(np.mean((q < -1e-12) & finite)) if len(q) else np.nan
        failure_rate = float(np.mean(~finite)) if len(q) else np.nan
        nonmono = 0
        extreme = 0
        ncurves = 0
        rhs_values = []
        dfdq_values = []
        f = f_lookup.get((model, seed))
        for _, gg in g.groupby("grid_id", sort=True):
            s = gg.sort_values("time_h")
            vals = s["Q_pred"].to_numpy(float)
            vals_norm = s["Qtilde_pred"].to_numpy(float) if "Qtilde_pred" in s.columns else vals / Q_SCALE
            ncurves += 1
            if np.all(np.isfinite(vals)) and np.any(np.diff(vals) < -1e-9):
                nonmono += 1
            q6 = s.loc[np.isclose(s["time_h"], 6.0)].iloc[0]
            q6_val = float(q6["Q_pred"])
            if np.isfinite(q6_val) and (q6_val < 0.0 or q6_val > observed_q6_upper):
                extreme += 1
            if f is not None:
                for qn in vals_norm[np.isfinite(vals_norm)]:
                    try:
                        rhs_values.append(float(f(float(qn), float(q6["C1n"]), float(q6["C2n"]), float(q6["C3n"]))))
                        dfdq_values.append(finite_dfdq(f, float(qn), float(q6["C1n"]), float(q6["C2n"]), float(q6["C3n"])))
                    except Exception:
                        rhs_values.append(np.nan)
                        dfdq_values.append(np.nan)
        rhs = np.asarray(rhs_values, dtype=float)
        rhs_finite = rhs[np.isfinite(rhs)]
        dfdq = np.asarray(dfdq_values, dtype=float)
        dfdq_finite = dfdq[np.isfinite(dfdq)]
        rows.append(
            {
                "model": model,
                "seed": seed,
                "negative_Q_prediction_rate": negative_q_rate,
                "non_monotonic_curve_rate": float(nonmono / ncurves) if ncurves else np.nan,
                "extreme_Q6_rate": float(extreme / ncurves) if ncurves else np.nan,
                "numerical_failure_rate": failure_rate,
                "negative_RHS_rate": float(np.mean(rhs_finite < -1e-12)) if rhs_finite.size else np.nan,
                "positive_dfdQ_rate": float(np.mean(dfdq_finite > 1e-12)) if dfdq_finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def optimisation_rows_from_grid(grid_pred: pd.DataFrame, observed_q6_upper: float) -> pd.DataFrame:
    rows = []
    for (model, seed), g in grid_pred.groupby(["model", "seed"], dropna=False, sort=False):
        best_row = None
        failed = 0
        extreme = 0
        total = 0
        best_neg = False
        best_nonmono = False
        best_fail = False
        for _, gg in g.groupby("grid_id", sort=True):
            s = gg.sort_values("time_h")
            vals = s["Q_pred"].to_numpy(float)
            total += 1
            has_fail = not np.all(np.isfinite(vals))
            has_neg = bool(np.any(vals[np.isfinite(vals)] < -1e-12))
            has_nonmono = bool(np.all(np.isfinite(vals)) and np.any(np.diff(vals) < -1e-9))
            if has_fail:
                failed += 1
            q6_row = s.loc[np.isclose(s["time_h"], 6.0)].iloc[0]
            q6 = float(q6_row["Q_pred"])
            if np.isfinite(q6) and (q6 < 0.0 or q6 > observed_q6_upper):
                extreme += 1
            if np.isfinite(q6) and (best_row is None or q6 > float(best_row["Q_pred"])):
                best_row = q6_row
                best_neg = has_neg
                best_nonmono = has_nonmono
                best_fail = has_fail
        if best_row is None:
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "best_C1": np.nan,
                    "best_C2": np.nan,
                    "best_C3": np.nan,
                    "best_Q6": np.nan,
                    "best_is_boundary": False,
                    "best_has_numerical_failure": True,
                    "best_has_negative_Q": False,
                    "best_has_non_monotonic_curve": False,
                    "failure_rate": failed / max(1, total),
                    "extreme_Q6_rate_optimisation_grid": extreme / max(1, total),
                }
            )
            continue
        c1, c2, c3 = float(best_row["C1"]), float(best_row["C2"]), float(best_row["C3"])
        rows.append(
            {
                "model": model,
                "seed": seed,
                "best_C1": c1,
                "best_C2": c2,
                "best_C3": c3,
                "best_Q6": float(best_row["Q_pred"]),
                "best_is_boundary": bool(is_boundary(c1, c2, c3)),
                "best_has_numerical_failure": bool(best_fail),
                "best_has_negative_Q": bool(best_neg),
                "best_has_non_monotonic_curve": bool(best_nonmono),
                "failure_rate": failed / max(1, total),
                "extreme_Q6_rate_optimisation_grid": extreme / max(1, total),
            }
        )
    return pd.DataFrame(rows)


def run_failed(path: Path, seed: int, stage: str, exc: BaseException) -> None:
    write_json(
        path / "run_failed.json",
        {
            "seed": seed,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_excerpt": traceback.format_exc(limit=8),
        },
    )


def elapsed(start: float) -> str:
    return f"{time.time() - start:.3f}"
