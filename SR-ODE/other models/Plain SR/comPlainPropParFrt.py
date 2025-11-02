import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
plain_path = r"Symbolic Regression/srloop/other models/Plain SR/data/hall_of_fame_run-plainSR_maxsize38_restored.csv"
proposed_path = r"Symbolic Regression/srloop/data/hall_of_fame_run-8_restored_analyzed.csv"

# 读取CSV
df_plain = pd.read_csv(plain_path)
df_proposed = pd.read_csv(proposed_path)

# Plain SR 数据
plain = df_plain[["complexity", "loss"]].dropna()
plain = plain.sort_values("complexity")

# Proposed SR 数据
proposed = df_proposed[["restored_complexity", "loss", "is_pareto_front"]].dropna()
proposed_all = proposed
proposed_front = proposed[proposed["is_pareto_front"] == True].sort_values("restored_complexity")

# 画图
# plt.figure(figsize=(8,6))

# Plain SR（线 + 点）
plt.plot(plain["complexity"], plain["loss"], 
         marker="o", color="blue", label="PySR")

# # Proposed SR - 所有点（浅蓝色散点）
# plt.scatter(proposed_all["restored_complexity"], proposed_all["loss"], 
#             color="green", alpha=0.6, s=50, label="PI-RSR (all)")

# PIRSR - Pareto front
plt.plot(proposed_front["restored_complexity"], proposed_front["loss"], 
           marker="o", color="green", alpha=0.6, label="PI-RSR")

# # Proposed SR - Pareto front (红色方框标记)
# plt.scatter(proposed_front["restored_complexity"], proposed_front["loss"], 
#             facecolors='none', edgecolors='red', s=120, marker="s", 
#             label="PI-RSR (Pareto front)")

# # Pareto front 连线
# plt.plot(proposed_front["restored_complexity"], proposed_front["loss"], 
#          color="lightcoral", linestyle="-", linewidth=1.5)

plt.xlabel("Complexity", fontsize=12)
plt.ylabel("Loss", fontsize=12)
# plt.title("Pareto Front Comparison: Plain SR vs Proposed SR")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Symbolic Regression/srloop/other models/Plain SR/visualisation/pareto_front_comparison_plainprop_pure.png", dpi=300, bbox_inches='tight')
plt.show()

# -------- 图2: 只显示 complexity > 5 --------
plain_high = plain[plain["complexity"] > 5]
proposed_all_high = proposed_all[proposed_all["restored_complexity"] > 5]
proposed_front_high = proposed_front[proposed_front["restored_complexity"] > 5]

# plt.figure(figsize=(8,6))
plt.plot(plain_high["complexity"], plain_high["loss"], marker="o", color="blue", label="PySR (complexity>5)")
plt.plot(proposed_front_high["restored_complexity"], proposed_front_high["loss"], marker="o", color="green", label="PI-RSR (complexity>5)")
# plt.scatter(proposed_all_high["restored_complexity"], proposed_all_high["loss"], 
#             color="green", alpha=0.6, s=50, label="PI-RSR (all, complexity>10)")
# plt.scatter(proposed_front_high["restored_complexity"], proposed_front_high["loss"], 
#             facecolors='none', edgecolors='red', s=120, marker="s", label="PI-RSR (Pareto front)")
# plt.plot(proposed_front_high["restored_complexity"], proposed_front_high["loss"], 
#          color="lightcoral", linestyle="-", linewidth=1.5)

plt.xlabel("Complexity", fontsize=12)
plt.ylabel("Loss", fontsize=12)
# plt.title("Pareto Front Comparison (Complexity > 10)")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Symbolic Regression/srloop/other models/Plain SR/visualisation/pareto_front_comparison_plainprop_complexity10_pure.png", dpi=300, bbox_inches='tight')
plt.show()