
import numpy as np
import pandas as pd
from icecream.icecream import ic

file_name = r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Dataset\Dataset_EachFormula-Real-2.xlsx'

Initial_data = pd.read_excel(file_name, sheet_name='Sheet1')

Formulas = ['E-1', 'E-2', 'E-3', 'E-4', 'E-5']

title_columns = Initial_data.columns.drop('Time')

record_f_data = {}

for f in Formulas:

    test_data = pd.read_excel(file_name, sheet_name=f)
    ic(test_data)

    y_data = test_data.iloc[:, 4:]
    ic(y_data)

    y_data_overall = np.mean(y_data, axis=0)
    ic(y_data_overall)

    y_data_overall = y_data_overall.to_numpy()

    record_f_data[f] = y_data_overall

ic(record_f_data)

# plot the all formulas data
import matplotlib.pyplot as plt
import seaborn as sns
markers = ['o', '^', 's', '*', 'D']

fig, ax = plt.subplots()
x_labels = np.array([1, 2, 3, 4, 6, 8, 22, 24, 26, 28])

for f in Formulas:
    ax.plot(x_labels, record_f_data[f], label=f, marker=markers[Formulas.index(f)])
    
ax.axhline(y=236.95, color='grey', linestyle='--', label=r'$y_{\mathrm{best}}$')
ax.set_xlabel('Sampling Time (h)', fontsize=12)
ax.set_ylabel('Measured cumulative amount of Ibu (μg/cm²)', fontsize=12)
# ax.set_title('Original Data')
ax.legend(fontsize=12)
plt.savefig(r'C:\Users\yz02380\OneDrive - University of Surrey\Science Research\Codes\Early_Decision\GPR_BO_Decision\Results\Original_Data.png', dpi=300, bbox_inches='tight')
plt.show()

    


