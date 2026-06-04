from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Benchmark" / "Improvement"


PLOTS = [
    {
        "pareto_path": ROOT / "Dataset" / "fulldata-s" / "pareto_front_fulldata.xlsx",
        "response_path": ROOT / "Dataset" / "IVRT-S.xlsx",
        "response_sheet": "R-S",
        "selected_indices": {4, 6, 8, 9, 10},
        "output": OUT_DIR / "fig5_revised.png",
    },
    {
        "pareto_path": ROOT / "Dataset" / "fulldata-s-epoch2" / "pareto_front_fulldata.xlsx",
        "response_path": ROOT / "Dataset" / "IVRT-S.xlsx",
        "response_sheet": "R-Opt-2",
        "selected_indices": {3, 4, 7, 8, 9, 10},
        "output": OUT_DIR / "fig7_revised.png",
    },
    {
        "pareto_path": ROOT / "Dataset" / "fulldata-s-epoch3" / "pareto_front_fulldata.xlsx",
        "response_path": ROOT / "Dataset" / "IVRT-S.xlsx",
        "response_sheet": "R-Opt-3",
        "selected_indices": set(),
        "output": OUT_DIR / "fig9_revised.png",
    },
]


def plot_selected_pareto(config):
    pareto = pd.read_excel(config["pareto_path"])
    responses = pd.read_excel(config["response_path"], sheet_name=config["response_sheet"])

    x = pareto["Mean"].to_numpy(dtype=float)
    y = pareto["Std"].to_numpy(dtype=float)
    incumbent_best = float(responses.iloc[:, -1].max())

    row_numbers = np.arange(1, len(pareto) + 1)
    selected_mask = np.isin(row_numbers, list(config["selected_indices"]))
    unselected_mask = ~selected_mask

    fig, ax = plt.subplots()

    not_selected, = ax.plot(
        x[unselected_mask],
        y[unselected_mask],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="red",
        linestyle="none",
        markersize=7.5,
        label="Pareto candidate (not selected)",
    )
    handles = [not_selected]
    if selected_mask.any():
        selected, = ax.plot(
            x[selected_mask],
            y[selected_mask],
            marker="^",
            markerfacecolor="red",
            markeredgecolor="red",
            linestyle="none",
            markersize=9.5,
            label="Selected for evaluation",
        )
        handles.append(selected)

    incumbent = ax.axvline(
        x=incumbent_best,
        color="black",
        linestyle="--",
        alpha=0.8,
        label="The incumbent best",
    )
    handles.append(incumbent)

    ax.set_xlabel("Prediction Mean")
    ax.set_ylabel("Prediction Standard Deviation (Uncertainty)")
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
    )

    config["output"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config["output"], dpi=400, bbox_inches="tight")
    plt.close(fig)

    return {
        "output": config["output"],
        "n_points": len(pareto),
        "n_selected": int(selected_mask.sum()),
        "incumbent_best": incumbent_best,
    }


def main():
    for config in PLOTS:
        result = plot_selected_pareto(config)
        print(
            f"{result['output']} | points={result['n_points']} | "
            f"selected={result['n_selected']} | incumbent={result['incumbent_best']:.6f}"
        )


if __name__ == "__main__":
    main()
