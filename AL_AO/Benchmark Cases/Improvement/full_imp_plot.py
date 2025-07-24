import pandas as pd
import matplotlib.pyplot as plt
# File path: Make sure Full_Imp.xlsx and this script are in the same directory
# 文件路径：确保 Full_Imp.xlsx 和脚本在同一目录
file_path = "Multi-Objective Optimisation\\Benchmark\\Improvement\\Full_Imp.xlsx"

# 获取所有工作表名

# Read all sheet names
xls = pd.ExcelFile(file_path)
sheets = xls.sheet_names

# Automatically identify benchmark function names and dimensions
functions = {}
for sheet in sheets:
    # 假设 sheet 名如 'F1-3D', 'F1-5D', 'F2-3D', 'F2-5D' 等
    # Assume sheet names like 'F1-3D', 'F1-5D', 'F2-3D', 'F2-5D', etc.
    if '3D' in sheet:
        func_name = sheet.replace('3D', '').replace('-', '').strip()
        functions.setdefault(func_name, {})['3D'] = sheet
    elif '5D' in sheet:
        func_name = sheet.replace('5D', '').replace('-', '').strip()
        functions.setdefault(func_name, {})['5D'] = sheet

num_funcs = len(functions)

# Create a subplot with 2 rows and N columns
fig, axes = plt.subplots(2, num_funcs, figsize=(4*num_funcs, 8), sharex=True, sharey=False)

for col_idx, (func_name, dims) in enumerate(functions.items()):
    for row_idx, dim in enumerate(['3D', '5D']):
        if dim in dims:
            ax = axes[row_idx, col_idx]
            sheet = dims[dim]
            df = pd.read_excel(file_path, sheet_name=sheet)
            # Clean the Iteration column to ensure it is integer
            df = df[pd.to_numeric(df['Iteration'], errors='coerce').notnull()]
            df['Iteration'] = df['Iteration'].astype(int)
            # Calculate and plot cumulative improvement
            for col in df.columns[1:]:
                cum_series = df[col].cumsum()
                label = col.replace(' Imp', '').replace('.1', '')
                ax.plot(df['Iteration'], cum_series, label=label)
            ax.set_title(f'{func_name}-{dim}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cumulative Improvement')
        else:
            axes[row_idx, col_idx].axis('off')

# Add a unified legend and adjust layout
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
fig.savefig("Multi-Objective Optimisation\\Benchmark\\Improvement\\benchmark_improvement_cumulative.png", dpi=300, bbox_inches='tight')
plt.show()
