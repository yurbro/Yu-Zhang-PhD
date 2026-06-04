from __future__ import annotations

import contextlib
import importlib.util
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "Benchmark"
N_INIT = 10
N_ITER = 180
SEEDS = list(range(30, 40))
AFS = ["ei", "ucb", "poi"]
DIMS = [3, 5]

BENCHMARKS = [
    {
        "name": "ackley",
        "label": "Ackley",
        "path": BENCHMARK_DIR / "Package Module-III" / "single_bo_custom.py",
        "bounds": (-32.768, 32.768),
        "passes_benchmark": False,
    },
    {
        "name": "rastrigin",
        "label": "Rastrigin",
        "path": BENCHMARK_DIR / "Package Module-III-rastrigin" / "single_bo_custom.py",
        "bounds": (-5.12, 5.12),
        "passes_benchmark": True,
    },
    {
        "name": "zakharov",
        "label": "Zakharov",
        "path": BENCHMARK_DIR / "Package Module-IIII" / "single_bo_custom.py",
        "bounds": (-10.0, 10.0),
        "passes_benchmark": True,
    },
    {
        "name": "griewank",
        "label": "Griewank",
        "path": BENCHMARK_DIR / "Package Module-III-griewank" / "single_bo_custom.py",
        "bounds": (-600.0, 600.0),
        "passes_benchmark": True,
    },
    {
        "name": "sphere",
        "label": "Sphere",
        "path": BENCHMARK_DIR / "Package Module-III-sphere" / "single_bo_custom.py",
        "bounds": (-60.0, 60.0),
        "passes_benchmark": True,
    },
]

LOCAL_MODULES = [
    "lhs_sample",
    "ackley_func",
    "zakharov_func",
    "rastrigin_func",
    "griewank_func",
    "sphere_func",
    "rosenbrock_func",
]


def load_bo_module(path: Path, module_name: str):
    folder = str(path.parent)
    for name in LOCAL_MODULES:
        sys.modules.pop(name, None)

    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(folder)
        except ValueError:
            pass


def run_cell(cell: dict, log_dir: str, progress_queue):
    warnings.filterwarnings("ignore")
    bench = cell["benchmark"]
    af = cell["af"]
    dim = cell["dim"]
    log_path = Path(log_dir) / f"{bench['name']}_{dim}D_{af}.log"
    result = {
        "benchmark": bench["name"],
        "dim": dim,
        "af": af,
        "bests": [],
        "status": "ok",
        "error": None,
    }

    try:
        module = load_bo_module(
            bench["path"],
            f"single_bo_custom_{bench['name']}_{dim}_{af}_{os.getpid()}",
        )
        acquisitions = {
            "ei": module.expected_improvement,
            "ucb": module.upper_confidence_bound,
            "poi": module.probability_of_improvement,
        }
        lb_val, ub_val = bench["bounds"]
        lb = np.array([lb_val] * dim)
        ub = np.array([ub_val] * dim)
        bounds = [(lb_val, ub_val) for _ in range(dim)]

        with log_path.open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                print(f"START {bench['name']} {dim}D {af}", flush=True)
                X_init, Y_init = module.run_moo_initial_experiment(
                    N_INIT, lb, ub, benchmark=bench["label"]
                )

                for seed in SEEDS:
                    seed_start = time.time()
                    np.random.seed(seed)
                    if bench["passes_benchmark"]:
                        X_opt, Y_opt = module.bayesian_optimization(
                            module.array_to_ackley,
                            X_init,
                            Y_init,
                            bounds,
                            n_iter=N_ITER,
                            af_func=af,
                            benchmark=bench["label"],
                            acquisition_func=acquisitions[af],
                            dim=dim,
                            random_seed=seed,
                        )
                    else:
                        X_opt, Y_opt = module.bayesian_optimization(
                            module.array_to_ackley,
                            X_init,
                            Y_init,
                            bounds,
                            n_iter=N_ITER,
                            af_func=af,
                            acquisition_func=acquisitions[af],
                            dim=dim,
                            random_seed=seed,
                        )
                    best = float(np.max(np.asarray(Y_opt, dtype=float).ravel()))
                    result["bests"].append(best)
                    elapsed = time.time() - seed_start
                    progress_queue.put(
                        f"{bench['name']} {dim}D {af} seed {seed} complete: "
                        f"best={best:.6g}, elapsed={elapsed / 60:.1f} min"
                    )
                print(f"END {bench['name']} {dim}D {af}", flush=True)
        return result
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
        progress_queue.put(f"ERROR {bench['name']} {dim}D {af}")
        return result


def format_value(mean: float, std: float) -> str:
    return f"{mean:.6g} ± {std:.6g}"


def main():
    max_workers = min(8, os.cpu_count() or 1)
    log_dir = BENCHMARK_DIR / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cells = [
        {"benchmark": bench, "dim": dim, "af": af}
        for bench in BENCHMARKS
        for dim in DIMS
        for af in AFS
    ]

    print(f"Starting full sweep: {len(cells)} cells, {len(SEEDS)} seeds each")
    print(f"Workers: {max_workers}")
    print(f"Logs: {log_dir}")
    start = time.time()
    results = []

    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_cell, cell, str(log_dir), progress_queue): cell
                for cell in cells
            }
            pending = set(futures)
            while pending:
                try:
                    message = progress_queue.get(timeout=30)
                    print(message, flush=True)
                except queue.Empty:
                    elapsed = (time.time() - start) / 60
                    print(
                        f"heartbeat: {len(cells) - len(pending)}/{len(cells)} cells done, "
                        f"elapsed={elapsed:.1f} min",
                        flush=True,
                    )

                done = {future for future in pending if future.done()}
                for future in done:
                    pending.remove(future)
                    cell = futures[future]
                    result = future.result()
                    if result["status"] != "ok":
                        print(result["error"], flush=True)
                        raise RuntimeError(
                            f"Cell failed: {cell['benchmark']['name']} "
                            f"{cell['dim']}D {cell['af']}"
                        )
                    results.append(result)
                    print(
                        f"CELL_DONE {result['benchmark']} {result['dim']}D "
                        f"{result['af']}",
                        flush=True,
                    )

    result_map = {
        (result["benchmark"], result["dim"], result["af"]): result["bests"]
        for result in results
    }

    print("\n| benchmark | dim | af | mean_best ± std_best |")
    print("|---|---:|---|---:|")
    for bench in BENCHMARKS:
        for dim in DIMS:
            for af in AFS:
                bests = np.array(result_map[(bench["name"], dim, af)], dtype=float)
                mean = float(np.mean(bests))
                std = float(np.std(bests, ddof=1))
                print(
                    f"| {bench['name']} | {dim} | {af} | {format_value(mean, std)} |"
                )

    elapsed = time.time() - start
    print(f"\nTotal elapsed: {elapsed / 3600:.2f} hours ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
