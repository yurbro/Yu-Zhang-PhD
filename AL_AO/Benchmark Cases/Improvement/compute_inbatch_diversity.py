from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Benchmark" / "Improvement"
SELECTED_BATCHES = OUT_DIR / "selected_batches.csv"

BOUNDS = {
    "ackley": (-32.768, 32.768),
    "rastrigin": (-5.12, 5.12),
    "zakharov": (-10.0, 10.0),
    "griewank": (-600.0, 600.0),
    "sphere": (-60.0, 60.0),
}
BENCHMARKS = ["ackley", "rastrigin", "zakharov", "griewank", "sphere"]
DIMS = [3, 5]
METHODS = ["EI-Pareto", "HV-Pareto", "Adaptive"]
SEEDS = list(range(30, 40))
ITERATIONS = list(range(1, 31))
TAU = 6


def mean_pairwise_distance(points: np.ndarray) -> float:
    distances = [
        np.linalg.norm(points[i] - points[j])
        for i, j in combinations(range(len(points)), 2)
    ]
    return float(np.mean(distances))


def diversity_for_batch(batch: pd.DataFrame, benchmark: str, dim: int) -> float:
    lb, ub = BOUNDS[benchmark]
    x_cols = [f"x{i}" for i in range(1, dim + 1)]
    points = batch.sort_values("point_index")[x_cols].to_numpy(dtype=float)
    scaled = (points - lb) / (ub - lb)
    return mean_pairwise_distance(scaled)


def build_batch_diversity(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings = []

    for benchmark in BENCHMARKS:
        for dim in DIMS:
            for method in METHODS:
                for seed in SEEDS:
                    for iteration in ITERATIONS:
                        mask = (
                            df["benchmark"].eq(benchmark)
                            & df["dim"].eq(dim)
                            & df["method"].eq(method)
                            & df["seed"].eq(seed)
                            & df["iteration"].eq(iteration)
                        )
                        batch = df[mask]
                        if len(batch) != TAU:
                            warnings.append(
                                f"{benchmark} {dim}D {method} seed={seed} iter={iteration}: "
                                f"{len(batch)} points, expected {TAU}"
                            )
                            continue
                        rows.append(
                            {
                                "benchmark": benchmark,
                                "dim": dim,
                                "method": method,
                                "seed": seed,
                                "iteration": iteration,
                                "diversity": diversity_for_batch(batch, benchmark, dim),
                            }
                        )

    return pd.DataFrame(rows), warnings


def summarize(batch_diversity: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        batch_diversity.groupby(["benchmark", "dim", "method", "seed"], as_index=False)["diversity"]
        .mean()
        .rename(columns={"diversity": "seed_mean_diversity"})
    )
    rows = []
    for benchmark in BENCHMARKS:
        for dim in DIMS:
            row = {"benchmark": benchmark, "dim": dim}
            for method in METHODS:
                values = per_seed[
                    per_seed["benchmark"].eq(benchmark)
                    & per_seed["dim"].eq(dim)
                    & per_seed["method"].eq(method)
                ]["seed_mean_diversity"].to_numpy(dtype=float)
                if len(values):
                    row[method] = f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}"
                else:
                    row[method] = "NA"
            rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary_values: pd.DataFrame) -> Path:
    plot_rows = []
    for benchmark in BENCHMARKS:
        for dim in DIMS:
            for method in METHODS:
                values = summary_values[
                    summary_values["benchmark"].eq(benchmark)
                    & summary_values["dim"].eq(dim)
                    & summary_values["method"].eq(method)
                ]["seed_mean_diversity"].to_numpy(dtype=float)
                plot_rows.append(
                    {
                        "label": f"{benchmark}-{dim}D",
                        "method": method,
                        "mean": values.mean(),
                        "std": values.std(ddof=1),
                    }
                )

    plot_df = pd.DataFrame(plot_rows)
    labels = [f"{benchmark}-{dim}D" for benchmark in BENCHMARKS for dim in DIMS]
    x = np.arange(len(labels))
    width = 0.25

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(14, 5.5))
    offsets = [-width, 0.0, width]
    for offset, method in zip(offsets, METHODS):
        method_df = plot_df[plot_df["method"].eq(method)].set_index("label").loc[labels]
        ax.bar(
            x + offset,
            method_df["mean"],
            width,
            yerr=method_df["std"],
            capsize=3,
            label=method,
        )

    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_title("In-Batch Diversity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    out_path = OUT_DIR / "fig_inbatch_diversity.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    if not SELECTED_BATCHES.exists():
        raise FileNotFoundError(f"{SELECTED_BATCHES} does not exist. Run the batch rerun first.")

    df = pd.read_csv(SELECTED_BATCHES)
    batch_diversity, warnings = build_batch_diversity(df)
    if batch_diversity.empty:
        raise ValueError("No complete selected batches found.")

    per_seed = (
        batch_diversity.groupby(["benchmark", "dim", "method", "seed"], as_index=False)["diversity"]
        .mean()
        .rename(columns={"diversity": "seed_mean_diversity"})
    )
    table = summarize(batch_diversity)
    fig_path = plot_summary(per_seed)

    batch_diversity.to_csv(OUT_DIR / "inbatch_diversity_by_batch.csv", index=False)
    per_seed.to_csv(OUT_DIR / "inbatch_diversity_by_seed.csv", index=False)
    table.to_csv(OUT_DIR / "inbatch_diversity_summary.csv", index=False)

    print(table.to_markdown(index=False))
    print(f"\nfigure={fig_path}")
    if warnings:
        print("\nWARNINGS")
        for warning in warnings[:50]:
            print("-", warning)
        if len(warnings) > 50:
            print(f"- ... {len(warnings) - 50} more warnings")


if __name__ == "__main__":
    main()
