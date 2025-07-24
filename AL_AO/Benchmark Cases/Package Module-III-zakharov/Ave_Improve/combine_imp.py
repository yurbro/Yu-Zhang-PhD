
import pandas as pd
import os
from openpyxl import load_workbook

def create_analysis_excel(method, alpha, dim):
    file_path_save = f"Multi-Objective Optimisation\\Benchmark\\Package Module-IIII\\Ave_Improve\\{method}_improvement_analysis_a-{alpha}-{dim}D.xlsx"
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path_save), exist_ok=True)

    # Prepare data for both sheets
    iterations = list(range(1, 31))
    df_incumbent = pd.DataFrame({
        'Iteration': iterations,
        'Inc_best': [''] * 30
    })
    df_ackley = pd.DataFrame({
        'Iteration': iterations,
        'Average Improvement': [''] * 30
    })

    if os.path.exists(file_path_save):
        # Append new sheets to existing file
        with pd.ExcelWriter(file_path_save, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_incumbent.to_excel(writer, sheet_name='IncumbentBest', index=False)
            df_ackley.to_excel(writer, sheet_name='Ackley_improvement', index=False)
    else:
        # Create new file with both sheets
        with pd.ExcelWriter(file_path_save, engine='openpyxl') as writer:
            df_incumbent.to_excel(writer, sheet_name='IncumbentBest', index=False)
            df_ackley.to_excel(writer, sheet_name='Ackley_improvement', index=False)

if __name__ == "__main__":
    method = "HV"  # or "PROPOSED", depending on the method
    alpha = 0.5
    dim = 5  # Dimension
    create_analysis_excel(method, alpha, dim)
    print(f"Analysis Excel file created at: Multi-Objective Optimisation\\Benchmark\\Package Module-III-zakharov\\Ave_Improve\\{method}_improvement_analysis_a-{alpha}-{dim}D.xlsx")
