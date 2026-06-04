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
ARCHIVE_ROOT = ROOT.parent / "(Archived)-AL-MOO" / "Benchmark"
OUT_DIR = BENCHMARK_ROOT / "Improvement"

BENCHMARKS = [
    {
        "name": "Ackley",
        "key": "ackley",
        "folder": "Package Module-III",
        "label": "Ackley",
        "bounds": (-32.768, 32.768),
        "bo_name": None,
    },
    {
        "name": "Rastrigin",
        "key": "rastrigin",
        "folder": "Package Module-III-rastrigin",
        "label": "Rastrigin",
        "bounds": (-5.12, 5.12),
        "bo_name": "Rastrigin",
    },
    {
        "name": "Zakharov",
        "key": "zakharov",
        "folder": "Package Module-IIII",
        "label": "Zakharov",
        "bounds": (-10.0, 10.0),
        "bo_name": "Zakharov",
    },
    {
        "name": "Griewank",
        "key": "griewank",
        "folder": "Package Module-III-griewank",
        "label": "Griewank",
        "bounds": (-600.0, 600.0),
        "bo_name": "Griewank",
    },
    {
        "name": "Sphere",
        "key": "sphere",
        "folder": "Package Module-III-sphere",
        "label": "Sphere",
        "bounds": (-60.0, 60.0),
        "bo_name": "Sphere",
    },
]

BASELINE_METHODS = [
    ("Adaptive", "PROPOSED"),
    ("EI-Pareto", "EI"),
    ("HV-Pareto", "HV"),
    ("Random", "RANDOM"),
]
BO_METHODS = [("BO-EI-fixed", "ei"), ("BO-UCB-fixed", "ucb"), ("BO-POI-fixed", "poi")]
PLOT_METHODS = ["Proposed", "EI", "HV", "RANDOM", "BO-EI", "BO-UCB", "BO-POI"]
SEEDS = list(range(30, 40))


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3g} +/- {std:.3g}"


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
    return [seg for seg in segments_by_iteration(df, iteration_col) if int(iterations.iloc[seg[-1]]) == final_iteration]


def bo_file(base: Path, bench: dict, af: str, dim: int) -> Path:
    if bench["bo_name"] is None:
        name = f"single_bo_{af}_{dim}D.xlsx"
    else:
        name = f"single_bo_{af}_{bench['bo_name']}_{dim}D.xlsx"
    return base / bench["folder"] / "BO-RE" / name


def read_bo_runs(path: Path, n_runs: int = 10) -> tuple[list[float], list[np.ndarray]]:
    df = pd.read_excel(path).reset_index(drop=True)
    best_col = next(col for col in df.columns if str(col).startswith("Current best"))
    segments = complete_segments(df, "iteration", 180)
    if len(segments) < n_runs:
        raise ValueError(f"{path} has only {len(segments)} complete BO run segments")
    chosen = segments[-n_runs:]
    finals = [float(df.iloc[seg[-1]][best_col]) for seg in chosen]
    curves = [pd.to_numeric(df.iloc[seg][best_col], errors="coerce").to_numpy(dtype=float) for seg in chosen]
    return finals, curves


def normalize_best_values(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).reset_index(drop=True)
    best_col = next(col for col in df.columns if str(col).endswith("_Best"))
    iteration = pd.to_numeric(df["Iteration"], errors="coerce")
    best = pd.to_numeric(df[best_col], errors="coerce")
    if "random_seed" in df.columns:
        seed = pd.to_numeric(df["random_seed"], errors="coerce")
    else:
        seed = pd.Series(np.nan, index=df.index)

    # Some appended sheets have later rows shifted one column right because the
    # old schema lacked random_seed. Repair those rows for analysis only.
    shifted = (
        iteration.isna()
        & best.between(1, 30, inclusive="both")
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


def read_baseline_values(path: Path) -> tuple[list[float], str]:
    df = normalize_best_values(path)
    seed_values = sorted({int(seed) for seed in df["seed"].dropna() if int(seed) in SEEDS})
    if len(seed_values) >= 10:
        values = []
        for seed in SEEDS:
            seed_df = df[df["seed"].eq(seed)].reset_index(drop=True)
            segments = complete_segments(seed_df, "Iteration", 30)
            if not segments:
                segments = segments_by_iteration(seed_df, "Iteration")
            values.append(float(seed_df.iloc[segments[-1][-1]]["Best"]))
        return values, "seeded"

    segments = complete_segments(df, "Iteration", 30)
    if len(segments) >= 10:
        chosen = segments[-10:]
    else:
        chosen = segments
    values = [float(df.iloc[seg[-1]]["Best"]) for seg in chosen]
    return values, f"complete_segments={len(chosen)}"


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


def initial_bests(bench: dict, dim: int) -> list[float]:
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


def fixed_bo_improvement_curves(bench: dict, dim: int, af: str) -> tuple[np.ndarray, np.ndarray]:
    _, curves = read_bo_runs(bo_file(BENCHMARK_ROOT, bench, af, dim))
    initial = initial_bests(bench, dim)
    increments = []
    for init_best, curve in zip(initial, curves):
        increments.append(np.maximum(np.diff(np.r_[init_best, curve]), 0.0))
    arr = np.vstack(increments)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)


def build_combined_table() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings = []
    for bench in BENCHMARKS:
        for dim in [3, 5]:
            row = {"benchmark": bench["name"], "dim": dim}
            for label, file_method in BASELINE_METHODS:
                path = BENCHMARK_ROOT / bench["folder"] / "Dataset" / f"ackley_best_values_{file_method}-{dim}D.xlsx"
                values, source = read_baseline_values(path)
                if len(values) != 10:
                    warnings.append(f"{bench['name']} {dim}D {label}: {len(values)} complete runs found ({source})")
                row[label] = fmt(float(np.mean(values)), sample_std(values))
            for label, af in BO_METHODS:
                values, _ = read_bo_runs(bo_file(BENCHMARK_ROOT, bench, af, dim))
                row[label] = fmt(float(np.mean(values)), sample_std(values))
            rows.append(row)
    return pd.DataFrame(rows), warnings


def build_material_changes() -> pd.DataFrame:
    rows = []
    for bench in BENCHMARKS:
        for dim in [3, 5]:
            for label, af in BO_METHODS:
                current_values, _ = read_bo_runs(bo_file(BENCHMARK_ROOT, bench, af, dim))
                old_values, _ = read_bo_runs(bo_file(ARCHIVE_ROOT, bench, af, dim))
                new_mean = float(np.mean(current_values))
                old_mean = float(np.mean(old_values))
                old_std = sample_std(old_values)
                rows.append(
                    {
                        "benchmark": bench["name"],
                        "dim": dim,
                        "method": label,
                        "old_mean": old_mean,
                        "old_std": old_std,
                        "new_mean": new_mean,
                        "new_std": sample_std(current_values),
                        "delta": new_mean - old_mean,
                        "changed_materially": abs(new_mean - old_mean) > old_std,
                    }
                )
    return pd.DataFrame(rows)


def build_revised_improvement_workbook() -> Path:
    source = OUT_DIR / "Full_Imp.xlsx"
    target = OUT_DIR / "Full_Imp_revised.xlsx"
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for bench in BENCHMARKS:
            for dim in [3, 5]:
                sheet = f"{bench['name']}-{dim}D"
                old = pd.read_excel(source, sheet_name=sheet)
                revised = old.copy()
                for plot_col, af in [("BO-EI", "ei"), ("BO-UCB", "ucb"), ("BO-POI", "poi")]:
                    mean_inc, _ = fixed_bo_improvement_curves(bench, dim, af)
                    revised.loc[: len(mean_inc) - 1, plot_col] = mean_inc
                revised.to_excel(writer, sheet_name=sheet, index=False)
    return target


def plot_fig4(workbook: Path) -> Path:
    out_path = OUT_DIR / "fig4_revised.png"
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(len(BENCHMARKS), 2, figsize=(12, 18), sharex=True, sharey=False)
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]
    label_map = {"Proposed": "Adaptive", "RANDOM": "Random"}
    handles = labels = None

    for row_idx, bench in enumerate(BENCHMARKS):
        for col_idx, dim in enumerate([3, 5]):
            ax = axes[row_idx, col_idx]
            df = pd.read_excel(workbook, sheet_name=f"{bench['name']}-{dim}D")
            # The legacy workbook has summary blocks after the first 30 rows,
            # and some of those blocks contain numeric entries in the first
            # column. Fig. 4 should use only the 30 iteration rows.
            df = df.iloc[:30].reset_index(drop=True)
            iterations = pd.to_numeric(df["Iteration"], errors="coerce").astype(int)
            for idx, method in enumerate(PLOT_METHODS):
                series = pd.to_numeric(df[method], errors="coerce").fillna(0.0).cumsum()
                ax.plot(
                    iterations,
                    series,
                    label=label_map.get(method, method),
                    linewidth=1.8,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                )
            ax.set_title(f"{bench['name']}-{dim}D")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cumulative Improvement")
            ax.grid(True, linestyle="--", alpha=0.45)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_path


def best_status(table: pd.DataFrame) -> pd.DataFrame:
    methods = [label for label, _ in BASELINE_METHODS] + [label for label, _ in BO_METHODS]
    rows = []
    for _, row in table.iterrows():
        means = {method: float(str(row[method]).split("+/-")[0]) for method in methods}
        best_mean = max(means.values())
        winners = [method for method, value in means.items() if np.isclose(value, best_mean)]
        rows.append(
            {
                "benchmark": row["benchmark"],
                "dim": int(row["dim"]),
                "winner": ", ".join(winners),
                "adaptive_status": "best" if "Adaptive" in winners else "worse",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    table, warnings = build_combined_table()
    material = build_material_changes()
    workbook = build_revised_improvement_workbook()
    fig_path = plot_fig4(workbook)
    status = best_status(table)

    table.to_csv(OUT_DIR / "combined_table_revised.csv", index=False)
    material.to_csv(OUT_DIR / "bo_material_changes_revised.csv", index=False)
    status.to_csv(OUT_DIR / "adaptive_status_revised.csv", index=False)

    print("COMBINED_TABLE")
    print(table.to_markdown(index=False))
    print("\nMATERIAL_CHANGES")
    changed = material[material["changed_materially"]].copy()
    print(changed.to_markdown(index=False))
    print("\nADAPTIVE_STATUS")
    print(status.to_markdown(index=False))
    print("\nWARNINGS")
    for warning in warnings:
        print("-", warning)
    print("\nOUTPUTS")
    print(f"figure={fig_path}")
    print(f"workbook={workbook}")
    print(f"combined_table={OUT_DIR / 'combined_table_revised.csv'}")
    print(f"material_changes={OUT_DIR / 'bo_material_changes_revised.csv'}")


if __name__ == "__main__":
    main()
