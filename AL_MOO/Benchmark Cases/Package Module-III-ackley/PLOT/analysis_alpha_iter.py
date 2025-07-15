

import pandas as pd
import matplotlib.pyplot as plt

# Set path and filename
path = 'Multi-Objective Optimisation/Benchmark/Package Module-III/GPT-PLOT'

filename = 'PROPOSED.xlsx'  # change to your file name
sheetname = 'Alpha-Iter'  # change to your sheet name
df = pd.read_excel(path + '/' + filename, sheet_name=sheetname)  # change to your file path

# Extract iteration and all alpha columns
iterations = df['Alpha']
alpha_columns = [col for col in df.columns if col.startswith('Iter')]

# Plot
plt.figure(figsize=(12, 6))
for col in alpha_columns:
    plt.plot(iterations, df[col], label=col)

# Plot settings
plt.xlabel('Alpha Value')
plt.ylabel('Ackley Best Value')
plt.title('Performance across different Alpha values')
plt.legend(loc='best', fontsize='small', ncol=2)
# plt.xticks(range(0, len(iterations), 1))
plt.savefig(path + '\\alpha_iter_performance.png', dpi=300, bbox_inches='tight')
plt.show()

