from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Benchmark" / "Improvement"
SELECTED_BATCHES = OUT_DIR / "selected_batches.csv"
LOG_PATH = OUT_DIR / "batch_diversity_sweep.log"
STATUS_PATH = OUT_DIR / "batch_diversity_sweep_status.csv"
DISCARD_INTERNAL_LOG = False
WORK_ROOT: Path | None = None

BENCHMARKS = {
    "ackley": {
        "folder": "Package Module-III",
        "runner": "run_moo_loop-V1.py",
        "bounds": (-32.768, 32.768),
        "lower_bound": 1e-1,
        "upper_bound": 1e1,
    },
    "rastrigin": {
        "folder": "Package Module-III-rastrigin",
        "runner": "run_moo_loop-v1.py",
        "bounds": (-5.12, 5.12),
        "lower_bound": 1e-1,
        "upper_bound": 1e1,
    },
    "zakharov": {
        "folder": "Package Module-IIII",
        "runner": "run_moo_loop-v1.py",
        "bounds": (-10.0, 10.0),
        "lower_bound": 1e-1,
        "upper_bound": 1e1,
    },
    "griewank": {
        "folder": "Package Module-III-griewank",
        "runner": "run_moo_loop-V1.py",
        "bounds": (-600.0, 600.0),
        "lower_bound": 1e-1,
        "upper_bound": 1e1,
    },
    "sphere": {
        "folder": "Package Module-III-sphere",
        "runner": "run_moo_loop-v1.py",
        "bounds": (-60.0, 60.0),
        "lower_bound": 1e-2,
        "upper_bound": 1e2,
    },
}

METHOD_LABELS = {
    "EI": "EI-Pareto",
    "HV": "HV-Pareto",
    "PROPOSED": "Adaptive",
}

MODULES_TO_CLEAR = [
    "lhs_sample",
    "ackley_func",
    "rastrigin_func",
    "zakharov_func",
    "griewank_func",
    "sphere_func",
    "multi_objective_optimisation",
    "acquisition_function",
    "adaptive_weight_func",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run batch methods and log selected batches for diversity analysis."
    )
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--methods", nargs="+", default=["EI", "HV", "PROPOSED"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30, 40)))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--append-selected", action="store_true")
    parser.add_argument("--selected-output", type=Path, default=None)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--discard-internal-log", action="store_true")
    return parser.parse_args()


@contextlib.contextmanager
def internal_output():
    if DISCARD_INTERNAL_LOG:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
    else:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                yield


def clear_local_imports() -> None:
    for name in MODULES_TO_CLEAR:
        sys.modules.pop(name, None)


def load_runner(folder: Path, runner_name: str):
    clear_local_imports()
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            f"batch_runner_{folder.name.replace('-', '_')}",
            folder / runner_name,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {folder / runner_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(folder))
        except ValueError:
            pass


def settings_for_method(method_base: str) -> tuple[int, int]:
    if method_base == "PROPOSED":
        return 50, 100
    return 10, 50


def append_status(row: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        STATUS_PATH,
        mode="a",
        header=not STATUS_PATH.exists(),
        index=False,
    )


def update_adaptive_accuracies(
    module,
    path: str,
    path_data_run: str,
    method: str,
    benchmark_col: str,
    selected_sheetname,
    iteration_index: int,
    accuracies: dict,
) -> None:
    sheet_name_ei, sheet_name_hv = selected_sheetname[0], selected_sheetname[1]
    result_path = Path(path_data_run) / f"Top-RUN{iteration_index + 1}-{method}_{benchmark_col}_result.xlsx"
    pareto_front_ei = pd.read_excel(result_path, sheet_name=sheet_name_ei)
    pareto_front_hv = pd.read_excel(result_path, sheet_name=sheet_name_hv)
    latest_df = pd.read_excel(path, sheet_name=f"RUN-{iteration_index + 1}-{method}")
    y_current_best = latest_df[benchmark_col].max()
    accuracies["ei"] = module.calculate_accuracy(pareto_front_ei[benchmark_col].values, y_current_best)
    accuracies["hv"] = module.calculate_accuracy(pareto_front_hv[benchmark_col].values, y_current_best)


def run_seed(
    module,
    *,
    path: str,
    dataset_dir: Path,
    results_dir: Path,
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    method: str,
    lower_bound: float,
    upper_bound: float,
    popsize: int,
    gen: int,
    seed: int,
    iterations: int,
    prev_weights: dict,
    accuracies: dict,
) -> tuple[dict | None, dict]:
    benchmark_col = "Ackley"
    total_selection = 6
    alpha = 0.5
    rounding = "floor"
    savefig = False

    for i in range(iterations):
        new_weights, allocation, path_data_run, selected_sheetname = module.run_moo_loop(
            path=path,
            path_data=str(dataset_dir),
            path_df=str(results_dir),
            d=dim,
            lb=lb,
            ub=ub,
            run_num=i,
            method=method,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            popsize=popsize,
            gen=gen,
            total_selection=total_selection,
            prev_weights=prev_weights,
            accuracies=accuracies,
            alpha=alpha,
            rounding=rounding,
            benchmark=benchmark_col,
            savefig=savefig,
            random_seed=seed,
        )
        prev_weights = new_weights
        if method.startswith("PROPOSED"):
            update_adaptive_accuracies(
                module,
                path,
                path_data_run,
                method,
                benchmark_col,
                selected_sheetname,
                i,
                accuracies,
            )

    return prev_weights, accuracies


def run_cell(
    benchmark: str,
    config: dict,
    dim: int,
    method_base: str,
    seeds: list[int],
    iterations: int,
) -> None:
    folder = ROOT / "Benchmark" / config["folder"]
    module = load_runner(folder, config["runner"])

    method = f"{method_base}-{dim}D"
    if WORK_ROOT is None:
        dataset_dir = folder / "Dataset"
        results_dir = folder / "Results"
    else:
        cell_root = WORK_ROOT / benchmark / f"{dim}D" / method_base
        dataset_dir = cell_root / "Dataset"
        results_dir = cell_root / "Results"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    module.directory = str(dataset_dir)

    lb_value, ub_value = config["bounds"]
    lb = np.array([lb_value] * dim)
    ub = np.array([ub_value] * dim)
    popsize, gen = settings_for_method(method_base)

    prev_weights = {"ei": 0.5, "hv": 0.5}
    accuracies = {"ei": 0.0, "hv": 0.0}

    with internal_output():
        path = module.run_moo_initial_experiment(
            10,
            dim,
            lb,
            ub,
            str(dataset_dir),
            method,
            "Ackley",
        )

    for seed in seeds:
        start = time.time()
        print(
            f"{benchmark} {dim}D {METHOD_LABELS[method_base]} seed={seed} "
            f"({iterations} iterations, popsize={popsize}, gen={gen})",
            flush=True,
        )
        with internal_output():
            prev_weights, accuracies = run_seed(
                module,
                path=path,
                dataset_dir=dataset_dir,
                results_dir=results_dir,
                dim=dim,
                lb=lb,
                ub=ub,
                method=method,
                lower_bound=config["lower_bound"],
                upper_bound=config["upper_bound"],
                popsize=popsize,
                gen=gen,
                seed=seed,
                iterations=iterations,
                prev_weights=prev_weights,
                accuracies=accuracies,
            )
        elapsed = time.time() - start
        append_status(
            {
                "benchmark": benchmark,
                "dim": dim,
                "method": METHOD_LABELS[method_base],
                "seed": seed,
                "iterations": iterations,
                "elapsed_seconds": round(elapsed, 3),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )


def main() -> None:
    global SELECTED_BATCHES, STATUS_PATH, LOG_PATH, DISCARD_INTERNAL_LOG, WORK_ROOT

    args = parse_args()
    DISCARD_INTERNAL_LOG = args.discard_internal_log
    if args.selected_output is not None:
        SELECTED_BATCHES = args.selected_output.resolve()
        os.environ["SELECTED_BATCHES_PATH"] = str(SELECTED_BATCHES)
    if args.status_output is not None:
        STATUS_PATH = args.status_output.resolve()
    if args.log_output is not None:
        LOG_PATH = args.log_output.resolve()
    if args.work_root is not None:
        WORK_ROOT = args.work_root.resolve()

    unknown_benchmarks = sorted(set(args.benchmarks) - set(BENCHMARKS))
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmark(s): {unknown_benchmarks}")
    unknown_methods = sorted(set(args.methods) - set(METHOD_LABELS))
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {unknown_methods}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.append_selected:
        SELECTED_BATCHES.unlink(missing_ok=True)
        STATUS_PATH.unlink(missing_ok=True)
        LOG_PATH.unlink(missing_ok=True)

    total_cells = len(args.benchmarks) * len(args.dims) * len(args.methods)
    print(
        f"Running {total_cells} benchmark-dim-method cells, "
        f"{len(args.seeds)} seeds each, {args.iterations} iterations per seed."
    )
    for benchmark in args.benchmarks:
        for dim in args.dims:
            for method_base in args.methods:
                run_cell(
                    benchmark,
                    BENCHMARKS[benchmark],
                    dim,
                    method_base,
                    args.seeds,
                    args.iterations,
                )


if __name__ == "__main__":
    main()
