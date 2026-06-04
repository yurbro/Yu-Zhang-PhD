from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Benchmark" / "Improvement"
DRIVER = OUT_DIR / "run_batch_diversity_sweep.py"
PART_DIR = OUT_DIR / "selected_batches_parts"
SELECTED_BATCHES = OUT_DIR / "selected_batches.csv"
STATUS_PATH = OUT_DIR / "batch_diversity_sweep_status.csv"
WORK_ROOT = Path(tempfile.gettempdir()) / "moo_batch_diversity_work"

BENCHMARKS = ["ackley", "rastrigin", "zakharov", "griewank", "sphere"]
DIMS = [3, 5]
METHODS = ["EI", "HV", "PROPOSED"]
METHOD_LABELS = {"EI": "EI-Pareto", "HV": "HV-Pareto", "PROPOSED": "Adaptive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel wrapper for the batch diversity sweep.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--dims", nargs="+", type=int, default=DIMS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30, 40)))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    return parser.parse_args()


def cell_name(benchmark: str, dim: int, method: str) -> str:
    return f"{benchmark}_{dim}D_{method}"


def build_tasks(args: argparse.Namespace) -> list[dict]:
    tasks = []
    for benchmark in args.benchmarks:
        for dim in args.dims:
            for method in args.methods:
                name = cell_name(benchmark, dim, method)
                tasks.append(
                    {
                        "name": name,
                        "benchmark": benchmark,
                        "dim": dim,
                        "method": method,
                        "selected": PART_DIR / f"{name}.csv",
                        "status": PART_DIR / f"{name}_status.csv",
                        "log": PART_DIR / f"{name}.log",
                        "stdout": PART_DIR / f"{name}_stdout.log",
                    }
                )
    return tasks


def launch_task(task: dict, args: argparse.Namespace) -> tuple[subprocess.Popen, object]:
    stdout_handle = task["stdout"].open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(DRIVER),
        "--benchmarks",
        task["benchmark"],
        "--dims",
        str(task["dim"]),
        "--methods",
        task["method"],
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--iterations",
        str(args.iterations),
        "--selected-output",
        str(task["selected"]),
        "--status-output",
        str(task["status"]),
        "--log-output",
        str(task["log"]),
        "--work-root",
        str(args.work_root),
        "--discard-internal-log",
    ]
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, stdout_handle


def tail(path: Path, n: int = 30) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def combine_outputs(tasks: list[dict]) -> None:
    frames = []
    statuses = []
    for task in tasks:
        if not task["selected"].exists():
            raise FileNotFoundError(f"Missing selected-batch part: {task['selected']}")
        frames.append(pd.read_csv(task["selected"]))
        if task["status"].exists():
            statuses.append(pd.read_csv(task["status"]))

    combined = pd.concat(frames, ignore_index=True)
    benchmark_order = {name: i for i, name in enumerate(BENCHMARKS)}
    method_order = {label: i for i, label in enumerate(METHOD_LABELS.values())}
    combined["_benchmark_order"] = combined["benchmark"].map(benchmark_order)
    combined["_method_order"] = combined["method"].map(method_order)
    combined = combined.sort_values(
        ["_benchmark_order", "dim", "_method_order", "seed", "iteration", "point_index"]
    ).drop(columns=["_benchmark_order", "_method_order"])
    combined.to_csv(SELECTED_BATCHES, index=False)

    if statuses:
        status = pd.concat(statuses, ignore_index=True)
        status["_benchmark_order"] = status["benchmark"].map(benchmark_order)
        status["_method_order"] = status["method"].map(method_order)
        status = status.sort_values(["_benchmark_order", "dim", "_method_order", "seed"])
        status = status.drop(columns=["_benchmark_order", "_method_order"])
        status.to_csv(STATUS_PATH, index=False)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)

    for path in [SELECTED_BATCHES, STATUS_PATH]:
        path.unlink(missing_ok=True)
    for path in PART_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    work_root = args.work_root.resolve()
    if work_root.exists():
        if "moo_batch_diversity" not in work_root.name:
            raise ValueError(f"Refusing to remove unexpected work root: {work_root}")
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    pending = deque(tasks)
    running: list[tuple[dict, subprocess.Popen, object, float]] = []
    completed: list[dict] = []
    failed = False
    start_all = time.time()

    print(f"Launching {len(tasks)} cells with {args.workers} worker(s).", flush=True)
    print(f"Temporary work root: {work_root}", flush=True)

    while pending or running:
        while pending and len(running) < args.workers and not failed:
            task = pending.popleft()
            proc, stdout_handle = launch_task(task, args)
            running.append((task, proc, stdout_handle, time.time()))
            print(f"START {task['name']} pid={proc.pid}", flush=True)

        time.sleep(5)
        still_running = []
        for task, proc, stdout_handle, start in running:
            code = proc.poll()
            if code is None:
                still_running.append((task, proc, stdout_handle, start))
                continue

            stdout_handle.close()
            elapsed = time.time() - start
            if code == 0:
                completed.append(task)
                print(f"DONE  {task['name']} elapsed={elapsed / 60:.1f} min", flush=True)
            else:
                failed = True
                print(f"FAIL  {task['name']} code={code} elapsed={elapsed / 60:.1f} min", flush=True)
                print(f"--- stdout tail for {task['name']} ---")
                print(tail(task["stdout"]))
                print(f"--- detail log tail for {task['name']} ---")
                print(tail(task["log"]))

        running = still_running
        if failed:
            for task, proc, stdout_handle, _ in running:
                proc.terminate()
                stdout_handle.close()
                print(f"TERM  {task['name']} pid={proc.pid}", flush=True)
            raise SystemExit(1)

    combine_outputs(tasks)
    elapsed_all = (time.time() - start_all) / 60
    print(f"Combined {len(tasks)} parts into {SELECTED_BATCHES}", flush=True)
    print(f"Total elapsed={elapsed_all:.1f} min", flush=True)


if __name__ == "__main__":
    main()
