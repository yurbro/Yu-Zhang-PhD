import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== 高级期刊风格配色（ggplot2 / Nature 风常用这一套） =====
train_color   = "#0072B2"  # 深蓝，Train
test_color    = "#009E73"  # 祖母绿，Test
history_color = "#666666"  # 深灰，History data

# 多个优化配方的颜色（高对比、适合打印，也比较抗色弱）
opt_colors = [
    "#D55E00",  # 橙红
    "#CC79A7",  # 品红
    "#E69F00",  # 金黄
    "#56B4E9",  # 浅蓝
    "#F0E442",  # 柔黄
]

# 设置全局字体大小为12
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ========= 0. 输出路径设置 =========
out_dir = r"Symbolic Differential\SymODE\optimisation\opt_exp"
os.makedirs(out_dir, exist_ok=True)
    
# ===== 1. 读数据 =====
file_path = r"Symbolic Differential/SymODE/optimisation/opt_exp/Raw_train_test_data.xlsx"   

df_all = pd.read_excel(file_path, sheet_name="ALL")
df_opt = pd.read_excel(file_path, sheet_name="Opt")   # <-- 新的 sheet 名
df_form_train = pd.read_excel(file_path, sheet_name="Formulas-train")
df_form_test = pd.read_excel(file_path, sheet_name="Formulation-test")

# ========= 2. 根据 Run No 区分 Train / Test =========
train_runs = set(df_form_train["Run No"])
test_runs = set(df_form_test["Run No"])

# 有交集的默认划到 test
train_only_runs = sorted(train_runs - test_runs)
test_all_runs = sorted(test_runs)

df_train = df_all[df_all["Run No"].isin(train_only_runs)]
df_test  = df_all[df_all["Run No"].isin(test_all_runs)]

# ========= 3. 时间和列名 =========
mean_cols = sorted(
    [c for c in df_all.columns if c.startswith("M_t")],
    key=lambda x: int(x.split("t")[-1])
)
std_cols = sorted(
    [c for c in df_all.columns if c.startswith("Std_t")],
    key=lambda x: int(x.split("t")[-1])
)

time_hours = np.array([0.5, 1, 2, 3, 4, 5, 6])[:len(mean_cols)]

# 为多个 opt 配方准备颜色循环
# opt_colors = ["tab:red", "tab:purple", "tab:green", "tab:brown", "tab:pink"]

# ========= 4. 图 1：Train vs Test vs 多个 Opt =========
fig1, ax1 = plt.subplots()

# ---- 4.1 Train：蓝色线 + 蓝色 std 阴影 ----
for _, row in df_train.iterrows():
    mean_vals = row[mean_cols].to_numpy(dtype=float)
    std_vals  = row[std_cols].to_numpy(dtype=float)
    ax1.plot(time_hours, mean_vals, color=train_color, alpha=0.6, linewidth=1)
    ax1.fill_between(time_hours,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=train_color, alpha=0.10)
# ---- 4.2 Test：橙色线 + 橙色 std 阴影 ----
for _, row in df_test.iterrows():
    mean_vals = row[mean_cols].to_numpy(dtype=float)
    std_vals  = row[std_cols].to_numpy(dtype=float)
    ax1.plot(time_hours, mean_vals, color=test_color, alpha=0.6, linewidth=1)
    ax1.fill_between(time_hours,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=test_color, alpha=0.10)
# ---- 4.3 多个 Opt：不同颜色线 + 对应 std 阴影 ----
opt_handles = []
for i, (_, row) in enumerate(df_opt.iterrows()):
    mean_vals = row[mean_cols].to_numpy(dtype=float)
    std_vals  = row[std_cols].to_numpy(dtype=float)
    color = opt_colors[i % len(opt_colors)]
    # 如果你在表里建了 "Label" 列，就用那一列做名称，否则用 Opt-1/2/3 这种默认名
    if "Label" in df_opt.columns:
        label = str(row["Label"])
    else:
        label = f"Opt-{i+1}"

    ax1.plot(time_hours, mean_vals, color=color, linewidth=2.0)
    ax1.fill_between(time_hours,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=color, alpha=0.20)

    opt_handles.append(Line2D([0], [0], color=color, lw=2, label=label))

# 图 1：图例 & 美化
train_line = Line2D([0], [0], color="tab:blue",   lw=2, label="Train")
test_line  = Line2D([0], [0], color="tab:orange", lw=2, label="Test")

ax1.legend(handles=[train_line, test_line] + opt_handles, loc="best")

ax1.set_xlabel("Time (h)", fontsize=12)
ax1.set_ylabel("Cumulative amount of ibu release (µg/cm²)", fontsize=12)
# ax1.set_title("IVRT release profiles: Historical data vs Optimised formulations")
ax1.grid(True, linestyle="--", alpha=0.3)
fig1.tight_layout()

# —— 图 1 保存 ——
fig1.savefig(
    os.path.join(out_dir, "IVRT_release_profiles_train_test_multiopt-cns.png"),
    dpi=300,
    bbox_inches="tight"
)

# ========= 5. 图 2：History data vs 多个 Opt =========
fig2, ax2 = plt.subplots()

# ---- 5.1 History data（Train + Test 合并） ----
for _, row in df_all.iterrows():
    mean_vals = row[mean_cols].to_numpy(dtype=float)
    std_vals  = row[std_cols].to_numpy(dtype=float)
    ax2.plot(time_hours, mean_vals, color=history_color, alpha=0.5, linewidth=1)
    ax2.fill_between(time_hours,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=history_color, alpha=0.10)
# ---- 5.2 多个 Opt ----
opt_handles2 = []
for i, (_, row) in enumerate(df_opt.iterrows()):
    mean_vals = row[mean_cols].to_numpy(dtype=float)
    std_vals  = row[std_cols].to_numpy(dtype=float)
    color = opt_colors[i % len(opt_colors)]
    if "Label" in df_opt.columns:
        label = str(row["Label"])
    else:
        label = f"Opt-{i+1}"

    ax2.plot(time_hours, mean_vals, color=color, linewidth=2.0)
    ax2.fill_between(time_hours,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=color, alpha=0.20)

    opt_handles2.append(Line2D([0], [0], color=color, lw=2, label=label))

hist_line = Line2D([0], [0], color=history_color, lw=2, label="Historical data")
ax2.legend(handles=[hist_line] + opt_handles2, loc="best")

ax2.set_xlabel("Time (h)", fontsize=12)
ax2.set_ylabel("Cumulative amount of ibu release (µg/cm²)", fontsize=12)
# ax2.set_title("IVRT release profiles: Historical data vs Optimised formulations")
ax2.grid(True, linestyle="--", alpha=0.3)
fig2.tight_layout()

# —— 图 2 保存 ——
fig2.savefig(
    os.path.join(out_dir, "IVRT_release_profiles_historical_data_multiopt-cns.png"),
    dpi=300,
    bbox_inches="tight"
)

# see plots
plt.show()