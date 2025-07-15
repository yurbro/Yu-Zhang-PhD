#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   Pareto_Fronts_Comparison.py
# Time    :   2025/07/15 19:34:54
# Author  :   Y.ZHANG (UniOfSurrey)
# Email   :   yu.zhang@surrey.ac.uk


import pandas as pd
import matplotlib.pyplot as plt

def plot_with_named_annotations(file_path, annotation_names=None):
    """
    Plot a scatter plot of Mean vs Std, supporting multiple sheets and annotation by sample names.
    :param file_path: Excel file path
    :param annotation_names: Optional, list, contains sample names you want to annotate (e.g., "Cf1")
    """
    xlsx = pd.ExcelFile(file_path)
    sheet_names = xlsx.sheet_names

    # plt.figure(figsize=(10, 8))
    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    markers = [ 'x', '*', 'o', 's', '^', 'D', 'P', '+']

    max_raw_x = None
    all_points = {}  # Used to store all point coordinates and their names

    for idx, sheet in enumerate(sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()

        # Find possible name column
        name_col = next((col for col in df.columns if 'formula' in col or 'num' in col), None)

        if name_col is None or not {'mean', 'std'}.issubset(df.columns):
            print(f"⚠️ Sheet '{sheet}' is missing required columns")
            continue

        x = df['mean']
        y = df['std']
        names = df[name_col]

        label = 'Raw' if sheet.lower() == 'raw' else sheet
        plt.scatter(x, y, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=label)

        for xi, yi, ni in zip(x, y, names):
            all_points[str(ni).strip()] = (xi, yi)

        if sheet.lower() == 'raw':
            # raw_combined = x + y
            raw_combined = x
            max_raw_x = x.loc[raw_combined.idxmax()]

    # Add specified annotations
    if annotation_names:
        # Assign different colors to each highlighted point
        highlight_colors = ['gold', 'limegreen', 'deepskyblue', 'crimson', 'violet', 'orange', 'brown', 'navy']
        for idx, name in enumerate(annotation_names):
            coord = all_points.get(name)
            if coord:
                color = highlight_colors[idx % len(highlight_colors)]
                plt.scatter(*coord, color=color, marker='o', s=40, edgecolor='black', zorder=5, label=f"Selected: {name}")
                # plt.annotate(name, coord, textcoords="offset points", xytext=(5, 5), ha='left')  # Uncomment to show text
            else:
                print(f"⚠️ Cannot find data point with name '{name}'.")

    # Add Raw maximum line
    if max_raw_x is not None:
        plt.axvline(x=max_raw_x, color='gray', linestyle='--', linewidth=1.5, label='The incumbent best')

    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Comparison of Raw and Optimisation Results')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    # plt.savefig('Multi-Objective Optimisation/Pareto_animation/Pareto_fronts_comparison_selected_NEI.png', dpi=300, bbox_inches='tight')
    plt.savefig('Multi-Objective Optimisation/Pareto_animation/Pareto_fronts_comparison_selected_NEI-FS-Corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call
file_path = 'Multi-Objective Optimisation/Dataset/Comparison of raw and optimisation in IVRT FS.xlsx'
# annotation_names = ['P_2', 'P_9', 'P_8', 'P_10', 'P_6']  # results of NEI
# annotation_names = ['P_4', 'P_5', 'P_6', 'P_7', 'P_8']  # results of HV
# annotation_names = ['P_3', 'P_5', 'P_7', 'P_6', 'P_8']  # results of HV-FS 
# annotation_names = ['P_1', 'P_10', 'P_9']  # results of NEI-FS FOR EPOCH-1
# annotation_names = ['P_2', 'P_5']  # results of NEI-FS-Corrected FOR EPOCH-1
annotation_names = ['P_2', 'P_3', 'P_5', 'P_4', 'P_9']  # results of NEI-LHS-Corrected FOR EPOCH-1


plot_with_named_annotations(file_path, annotation_names)

# Plot the Noisy EI value by using column

