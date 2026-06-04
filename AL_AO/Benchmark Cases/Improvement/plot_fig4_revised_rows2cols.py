from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "Benchmark"
OUT_DIR = BENCHMARK_ROOT / "Improvement"
PLOT_DIR = OUT_DIR / "plot"

BENCHMARKS = [
    {
        "name": "Ackley",
        "folder": "Package Module-III",
        "label": "Ackley",
        "bounds": (-32.768, 32.768),
        "bo_name": None,
    },
    {
        "name": "Rastrigin",
        "folder": "Package Module-III-rastrigin",
        "label": "Rastrigin",
        "bounds": (-5.12, 5.12),
        "bo_name": "Rastrigin",
    },
    {
        "name": "Zakharov",
        "folder": "Package Module-IIII",
        "label": "Zakharov",
        "bounds": (-10.0, 10.0),
        "bo_name": "Zakharov",
    },
    {
        "name": "Griewank",
        "folder": "Package Module-III-griewank",
        "label": "Griewank",
        "bounds": (-600.0, 600.0),
        "bo_name": "Griewank",
    },
    {
        "name": "Sphere",
        "folder": "Package Module-III-sphere",
        "label": "Sphere",
        "bounds": (-60.0, 60.0),
        "bo_name": "Sphere",
    },
]

BASELINE_METHODS = [
    ("Adaptive", "PROPOSED"),
    ("EI", "EI"),
    ("HV", "HV"),
    ("Random", "RANDOM"),
]
BO_METHODS = [("BO-EI", "ei"), ("BO-UCB", "ucb"), ("BO-POI", "poi")]
METHOD_COLORS = {
    "Adaptive": "#1f77b4",
    "EI": "#ff7f0e",
    "HV": "#2ca02c",
    "Random": "#d62728",
    "BO-EI": "#9467bd",
    "BO-UCB": "#8c564b",
    "BO-POI": "#e377c2",
}
SEEDS = list(range(30, 40))
N_ITER = 30


def segments_by_iteration(df: pd.DataFrame, iteration_col: str) -> list[list[int]]:
    iterations = pd.to_numeric(df[iteration_col], errors="coerce")
    segments: list[list[int]] = []
    current: list[int] = []
    previous = None
    for idx, value in enumerate(iterations):
        if pd.isna(value):
            continue
        if current and value <= previous:
            segments.append(current)
            current = []
        current.append(idx)
        previous = value
    if current:
        segments.append(current)
    return segments


def complete_segments(df: pd.DataFrame, iteration_col: str, final_iteration: int) -> list[list[int]]:
    iterations = pd.to_numeric(df[iteration_col], errors="coerce")
    return [
        seg
        for seg in segments_by_iteration(df, iteration_col)
        if len(seg) and int(iterations.iloc[seg[-1]]) == final_iteration
    ]


def normalize_best_values(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).reset_index(drop=True)
    best_col = next(col for col in df.columns if str(col).endswith("_Best"))
    iteration = pd.to_numeric(df["Iteration"], errors="coerce")
    best = pd.to_numeric(df[best_col], errors="coerce")
    seed = pd.to_numeric(df["random_seed"], errors="coerce") if "random_seed" in df else pd.Series(np.nan, index=df.index)

    shifted = (
        iteration.isna()
        & best.between(1, N_ITER, inclusive="both")
        & pd.to_numeric(df.get("Weight_EI"), errors="coerce").notna()
    )
    if shifted.any():
        iteration.loc[shifted] = best.loc[shifted]
        best.loc[shifted] = pd.to_numeric(df.loc[shifted, "Weight_EI"], errors="coerce")
        unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
        if unnamed_cols:
            seed.loc[shifted] = pd.to_numeric(df.loc[shifted, unnamed_cols[-1]], errors="coerce")

    out = pd.DataFrame({"Iteration": iteration, "Best": best, "seed": seed})
    return out[out["Iteration"].notna() & out["Best"].notna()].reset_index(drop=True)


def initial_best_from_lhs(bench: dict, method: str, dim: int) -> float:
    path = (
        BENCHMARK_ROOT
        / bench["folder"]
        / "Dataset"
        / f"lhs_samples_Ackley_{method}-{dim}D.xlsx"
    )
    sheet = f"RUN-0-{method}-{dim}D"
    df = pd.read_excel(path, sheet_name=sheet)
    value_cols = [col for col in df.columns if not str(col).lower().startswith("x")]
    if not value_cols:
        raise ValueError(f"No objective column found in {path}::{sheet}")
    return float(pd.to_numeric(df[value_cols[-1]], errors="coerce").max())


def baseline_cumulative_curves(bench: dict, method: str, dim: int) -> np.ndarray:
    path = (
        BENCHMARK_ROOT
        / bench["folder"]
        / "Dataset"
        / f"ackley_best_values_{method}-{dim}D.xlsx"
    )
    df = normalize_best_values(path)
    init_best = initial_best_from_lhs(bench, method, dim)
    curves = []

    seed_values = sorted({int(seed) for seed in df["seed"].dropna() if int(seed) in SEEDS})
    if len(seed_values) >= len(SEEDS):
        for seed in SEEDS:
            seed_df = df[df["seed"].eq(seed)].reset_index(drop=True)
            segments = complete_segments(seed_df, "Iteration", N_ITER)
            if not segments:
                segments = segments_by_iteration(seed_df, "Iteration")
            seg = segments[-1]
            best = pd.to_numeric(seed_df.iloc[seg]["Best"], errors="coerce").to_numpy(dtype=float)
            curves.append(cumulative_curve(best, init_best))
    else:
        segments = complete_segments(df, "Iteration", N_ITER)
        for seg in segments[-len(SEEDS) :]:
            best = pd.to_numeric(df.iloc[seg]["Best"], errors="coerce").to_numpy(dtype=float)
            curves.append(cumulative_curve(best, init_best))

    if not curves:
        raise ValueError(f"{path} yielded no usable cumulative curves")
    if len(curves) != len(SEEDS):
        print(f"WARNING: {path} yielded {len(curves)} usable curves, expected {len(SEEDS)}")
    return np.vstack(curves)


def cumulative_curve(best_values: np.ndarray, init_best: float) -> np.ndarray:
    curve = np.maximum(best_values - init_best, 0.0)
    if len(curve) == N_ITER:
        return curve
    if len(curve) > N_ITER:
        return curve[:N_ITER]
    if len(curve) == 0:
        raise ValueError("Cannot pad an empty cumulative curve")
    return np.pad(curve, (0, N_ITER - len(curve)), mode="edge")


def bo_file(bench: dict, af: str, dim: int) -> Path:
    if bench["bo_name"] is None:
        name = f"single_bo_{af}_{dim}D.xlsx"
    else:
        name = f"single_bo_{af}_{bench['bo_name']}_{dim}D.xlsx"
    return BENCHMARK_ROOT / bench["folder"] / "BO-RE" / name


def load_module(path: Path):
    module_name = f"_single_bo_{path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def bo_initial_bests(bench: dict, dim: int) -> list[float]:
    module = load_module(BENCHMARK_ROOT / bench["folder"] / "single_bo_custom.py")
    lb, ub = bench["bounds"]
    lower = np.array([lb] * dim)
    upper = np.array([ub] * dim)
    values = []
    for seed in SEEDS:
        np.random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            _, y_init = module.run_moo_initial_experiment(10, lower, upper, benchmark=bench["label"])
        values.append(float(np.max(y_init)))
    return values


def bo_cumulative_curves(bench: dict, af: str, dim: int) -> np.ndarray:
    df = pd.read_excel(bo_file(bench, af, dim)).reset_index(drop=True)
    best_col = next(col for col in df.columns if str(col).startswith("Current best"))
    segments = complete_segments(df, "iteration", 180)
    if len(segments) < len(SEEDS):
        raise ValueError(f"{bo_file(bench, af, dim)} yielded {len(segments)} complete BO curves")

    initial = bo_initial_bests(bench, dim)
    curves = []
    for init_best, seg in zip(initial, segments[-len(SEEDS) :]):
        best = pd.to_numeric(df.iloc[seg][best_col], errors="coerce").to_numpy(dtype=float)
        if len(best) != N_ITER:
            raise ValueError(f"{bo_file(bench, af, dim)} segment length {len(best)}, expected {N_ITER}")
        curves.append(cumulative_curve(best, init_best))
    return np.vstack(curves)


def mean_and_se(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return curves.mean(axis=0), curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])


def plot_fig4_revised_rows2cols() -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 14,
        }
    )

    fig, axes = plt.subplots(len(BENCHMARKS), 2, figsize=(12, 20), sharex=True, sharey=False)
    iterations = np.arange(1, N_ITER + 1)
    handles = labels = None

    for row_idx, bench in enumerate(BENCHMARKS):
        for col_idx, dim in enumerate([3, 5]):
            ax = axes[row_idx, col_idx]

            for label, method in BASELINE_METHODS:
                mean, se = mean_and_se(baseline_cumulative_curves(bench, method, dim))
                linewidth = 2.8 if label == "Adaptive" else 1.8
                zorder = 5 if label == "Adaptive" else 3
                line, = ax.plot(
                    iterations,
                    mean,
                    color=METHOD_COLORS[label],
                    linestyle="-",
                    linewidth=linewidth,
                    label=label,
                    zorder=zorder,
                )
                ax.fill_between(
                    iterations,
                    mean - se,
                    mean + se,
                    alpha=0.10,
                    color=line.get_color(),
                    zorder=1,
                )

            for label, af in BO_METHODS:
                mean, se = mean_and_se(bo_cumulative_curves(bench, af, dim))
                line, = ax.plot(
                    iterations,
                    mean,
                    color=METHOD_COLORS[label],
                    linestyle="--",
                    linewidth=1.8,
                    label=label,
                    zorder=2,
                )
                ax.fill_between(
                    iterations,
                    mean - se,
                    mean + se,
                    alpha=0.10,
                    color=line.get_color(),
                    zorder=1,
                )

            ax.set_title(f"{bench['name']}-{dim}D")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cumulative Improvement")
            ax.grid(True, linestyle="--", alpha=0.6)

            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(labels),
            frameon=False,
            fontsize=14,
            handlelength=2.5,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.034, 1, 1])

    main_out = OUT_DIR / "fig4_revised.png"
    styled_out = PLOT_DIR / "fig4_revised_rows2cols_SE.png"
    fig.savefig(main_out, dpi=400, bbox_inches="tight")
    fig.savefig(styled_out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return main_out, styled_out


if __name__ == "__main__":
    outputs = plot_fig4_revised_rows2cols()
    print("Done ->", " & ".join(str(path) for path in outputs))
