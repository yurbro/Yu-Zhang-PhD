# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# base_path = r"Symbolic Regression\srloop\data"
# runs = range(1, 9)

# fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
# axes = axes.flatten()

# for idx, run in enumerate(runs):
#     file_path = os.path.join(base_path, f"feature_scores_run-{run}.csv")
#     if not os.path.exists(file_path):
#         print(f"Missing: {file_path}")
#         continue
    
#     df = pd.read_csv(file_path)
#     df["max_score"] = df[["MI", "Lasso"]].max(axis=1)
#     df_sorted = df.sort_values(by="max_score", ascending=True)

#     y = range(len(df_sorted))
#     bar_width = 0.4
#     ax = axes[idx]

#     ax.barh([i - bar_width/2 for i in y], df_sorted["MI"], height=bar_width, label="MI")
#     ax.barh([i + bar_width/2 for i in y], df_sorted["Lasso"], height=bar_width, label="LASSO")
#     ax.set_yticks(y)
#     ax.set_yticklabels(df_sorted.iloc[:,0], fontsize=8)
#     ax.set_title(f"Run {run}")

#     if idx == 0:
#         ax.legend()

# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = r"Symbolic Regression\srloop\data"
runs = range(1, 9)

fig, axes = plt.subplots(2, 4, figsize=(20, 15), sharex=True)
axes = axes.flatten()

for idx, run in enumerate(runs):
    file_path = os.path.join(base_path, f"feature_scores_run-{run}.csv")
    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        continue
    
    df = pd.read_csv(file_path)

    # 只保留 MI 或 Lasso > 0 的特征
    df = df[(df["MI"] > 0) | (df["Lasso"] > 0)]
    if df.empty:
        continue

    df["max_score"] = df[["MI", "Lasso"]].max(axis=1)
    df_sorted = df.sort_values(by="max_score", ascending=True)

    y = range(len(df_sorted))
    bar_width = 0.4
    ax = axes[idx]

    # 归一化 MI 和 LASSO 到 [0, 1]
    mi = df_sorted["MI"]
    lasso = df_sorted["Lasso"]
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min()) if mi.max() > mi.min() else mi
    lasso_norm = (lasso - lasso.min()) / (lasso.max() - lasso.min()) if lasso.max() > lasso.min() else lasso

    # 增强条形图显示效果（用归一化后的数据）
    ax.barh([i - bar_width/2 for i in y], mi_norm, height=bar_width, label="MI", color="#1f77b4", edgecolor="black", linewidth=1.5)
    ax.barh([i + bar_width/2 for i in y], lasso_norm, height=bar_width, label="LASSO", color="#ff7f0e", edgecolor="black", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted.iloc[:,0], fontsize=16, fontweight='bold')
    ax.set_title(f"Run {run}", fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel("Score", fontsize=16, fontweight='bold')
    # 加粗坐标轴
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    # 不在每个子图单独显示图例

plt.tight_layout()
# 在所有子图底部统一显示图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=18, frameon=False, bbox_to_anchor=(0.5, -0.03))
plt.savefig("Symbolic Regression/srloop/visualisation/feature_importance_trends_mi_lasso_nonzero_improved_nor.png", dpi=300, bbox_inches='tight')
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# base_path = r"Symbolic Regression\srloop\data"
# runs = range(1, 9)

# # 合并所有 run 数据
# all_data = []
# for run in runs:
#     file_path = os.path.join(base_path, f"feature_scores_run-{run}.csv")
#     if not os.path.exists(file_path):
#         continue
#     df = pd.read_csv(file_path)
#     df["run"] = run
#     # 只保留 MI 或 Lasso > 0
#     df = df[(df["MI"] > 0) | (df["Lasso"] > 0)]
#     all_data.append(df)

# df_all = pd.concat(all_data)

# # --- MI 趋势 ---
# plt.figure(figsize=(10,6))
# for feature in df_all.iloc[:,0].unique():
#     sub = df_all[df_all.iloc[:,0] == feature]
#     plt.plot(sub["run"], sub["MI"], marker="o", label=feature)

# plt.xlabel("Run")
# plt.ylabel("Mutual Information Score")
# plt.title("Feature Importance (MI) Across Iterations (Non-zero Only)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # --- LASSO 趋势 ---
# plt.figure(figsize=(10,6))
# for feature in df_all.iloc[:,0].unique():
#     sub = df_all[df_all.iloc[:,0] == feature]
#     plt.plot(sub["run"], sub["Lasso"], marker="s", label=feature)

# plt.xlabel("Run")
# plt.ylabel("LASSO Score")
# plt.title("Feature Importance (LASSO) Across Iterations (Non-zero Only)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# base_path = r"Symbolic Regression\srloop\data"
# runs = range(1, 9)

# # 合并所有 run 数据
# all_data = []
# for run in runs:
#     file_path = os.path.join(base_path, f"feature_scores_run-{run}.csv")
#     if not os.path.exists(file_path):
#         continue
#     df = pd.read_csv(file_path)
#     df["run"] = run
#     # 只保留 MI 或 Lasso > 0
#     df = df[(df["MI"] > 0) | (df["Lasso"] > 0)]
#     all_data.append(df)

# df_all = pd.concat(all_data)

# # --- 画叠加趋势图 ---
# plt.figure(figsize=(12,7))
# for feature in df_all.iloc[:,0].unique():
#     sub = df_all[df_all.iloc[:,0] == feature]

#     # MI 曲线
#     plt.plot(
#         sub["run"], sub["MI"],
#         marker="o", linestyle="-", label=f"{feature} (MI)"
#     )
#     # LASSO 曲线
#     plt.plot(
#         sub["run"], sub["Lasso"],
#         marker="s", linestyle="--", label=f"{feature} (LASSO)"
#     )

# plt.xlabel("Run")
# plt.ylabel("Score")
# plt.title("Feature Importance (MI vs LASSO) Across Iterations (Non-zero Only)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

