
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

df_abic_1 = pd.read_excel("Symbolic Regression/srloop/data/Q_pred_test_filtered_ABIC_1.xlsx")
df_abic_2 = pd.read_excel("Symbolic Regression/srloop/data/Q_pred_test_filtered_ABIC_2.xlsx")
df_abic_3 = pd.read_excel("Symbolic Regression/srloop/data/Q_pred_test_filtered_ABIC_3.xlsx")
df_true = pd.read_excel("Symbolic Regression/srloop/data/Q_true.xlsx")
# 进行对比分析
comparison_df = pd.DataFrame({
    "ABIC_1": df_abic_1.values.flatten(),
    "ABIC_2": df_abic_2.values.flatten(),
    "ABIC_3": df_abic_3.values.flatten(),
    "True": df_true.values.flatten()
})

# 进行统计分析
stats_df = comparison_df.describe()

# 进行可视化分析
import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.boxplot([comparison_df["ABIC_1"], comparison_df["ABIC_2"], comparison_df["ABIC_3"]],
#             labels=["ABIC_1", "ABIC_2", "ABIC_3"])
# plt.title("ABIC Scores Comparison")
# plt.ylabel("ABIC Score")
# plt.grid(True)
# # plt.savefig("Symbolic Regression/srloop/visualisation/ABIC_comparison_boxplot.png", dpi=300, bbox_inches='tight')
# plt.show()

# 预测值和真实值的散点图
plt.figure(figsize=(8, 6))
plt.plot([df_true.values.min(), df_true.values.max()], [df_true.values.min(), df_true.values.max()], 'k--', lw=2, label='Ideal Fit')
plt.scatter(df_true.values.flatten(), df_abic_1.values.flatten(), alpha=0.6, color='blue', label=f'No. 10 (R² = {r2_score(df_true.values.flatten(), df_abic_1.values.flatten()):.3f}, RMSE = {np.sqrt(mean_squared_error(df_true.values.flatten(), df_abic_1.values.flatten())):.3f}), MAE = {mean_absolute_error(df_true.values.flatten(), df_abic_1.values.flatten()):.3f})')
plt.scatter(df_true.values.flatten(), df_abic_2.values.flatten(), alpha=0.6, color='orange', label=f'No. 11 (R² = {r2_score(df_true.values.flatten(), df_abic_2.values.flatten()):.3f}, RMSE = {np.sqrt(mean_squared_error(df_true.values.flatten(), df_abic_2.values.flatten())):.3f}), MAE = {mean_absolute_error(df_true.values.flatten(), df_abic_2.values.flatten()):.3f})')
plt.scatter(df_true.values.flatten(), df_abic_3.values.flatten(), alpha=0.6, color='green', label=f'No. 9 (R² = {r2_score(df_true.values.flatten(), df_abic_3.values.flatten()):.3f}, RMSE = {np.sqrt(mean_squared_error(df_true.values.flatten(), df_abic_3.values.flatten())):.3f}), MAE = {mean_absolute_error(df_true.values.flatten(), df_abic_3.values.flatten()):.3f})')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
# plt.title("Scatter Plot of True vs Predicted Values")
plt.legend()
plt.grid(True)
plt.savefig("Symbolic Regression/srloop/visualisation/ABIC_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()
