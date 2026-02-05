import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# ================= 1. 全局绘图风格设置 (Nature/Science 风格) =================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 11,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "figure.dpi": 300
})

# ================= 2. 配色方案 =================
COLOR_TRAIN   = "#0072B2"  # 深蓝
COLOR_TEST    = "#009E73"  # 祖母绿
COLOR_HISTORY = "#999999"  # 浅灰

OPT_COLORS = [
    "#D55E00",  # 橙红
    "#56B4E9",  # 天蓝
    "#E69F00",  # 金黄
    "#CC79A7",  # 品红
    "#009E73",  # 青绿
    "#F0E442",  # 柔黄
]

# ================= 3. 数据读取与预处理 =================
out_dir = r"Symbolic Differential\SymODE\optimisation\opt_exp"
os.makedirs(out_dir, exist_ok=True)
file_path = r"Symbolic Differential/SymODE/optimisation/opt_exp/Raw_train_test_data.xlsx"

# 请确保文件路径正确
try:
    df_all = pd.read_excel(file_path, sheet_name="ALL")
    df_opt = pd.read_excel(file_path, sheet_name="Opt")
    df_form_train = pd.read_excel(file_path, sheet_name="Formulas-train")
    df_form_test = pd.read_excel(file_path, sheet_name="Formulation-test")
except FileNotFoundError:
    print(f"Error: 找不到文件 {file_path}，请检查路径。")
    exit()

# 区分 Train / Test
train_runs = set(df_form_train["Run No"])
test_runs = set(df_form_test["Run No"])
train_only_runs = sorted(train_runs - test_runs)
test_all_runs = sorted(test_runs)

df_train = df_all[df_all["Run No"].isin(train_only_runs)]
df_test  = df_all[df_all["Run No"].isin(test_all_runs)]

mean_cols = sorted([c for c in df_all.columns if c.startswith("M_t")], key=lambda x: int(x.split("t")[-1]))
std_cols = sorted([c for c in df_all.columns if c.startswith("Std_t")], key=lambda x: int(x.split("t")[-1]))
time_hours = np.array([0.5, 1, 2, 3, 4, 5, 6])[:len(mean_cols)]

# ================= 4. 核心绘图函数 (已修正参数) =================
def plot_release_profile(ax, background_dfs, opt_df, title_text=""):
    """
    ax: 绘图坐标轴
    background_dfs: 列表，结构为 [(DataFrame, color_code, label_string), ...]
    opt_df: 要高亮的优化配方 DataFrame
    title_text: 标题文本
    """
    
    # --- 1. 绘制背景数据 (Train/Test/History) ---
    lines_handles = []
    
    for df_bg, color, label in background_dfs:
        for _, row in df_bg.iterrows():
            mean_vals = row[mean_cols].to_numpy(dtype=float)
            std_vals  = row[std_cols].to_numpy(dtype=float)
            
            # 误差带
            ax.fill_between(time_hours, mean_vals - std_vals, mean_vals + std_vals,
                            color=color, alpha=0.05, linewidth=0, zorder=0)
            # 线条 (细且淡)
            ax.plot(time_hours, mean_vals, color=color, alpha=0.4, 
                    linewidth=1.0, linestyle='-', zorder=1)
        
        # 添加图例句柄 (仅添加一次)
        lines_handles.append(Line2D([0], [0], color=color, lw=1.5, label=label))

    # --- 2. 绘制优化数据 (Opt) ---
    opt_handles = []
    markers = ['o', 's', '^', 'D', 'v', '<'] 
    
    for i, (_, row) in enumerate(opt_df.iterrows()):
        mean_vals = row[mean_cols].to_numpy(dtype=float)
        std_vals  = row[std_cols].to_numpy(dtype=float)
        
        color = OPT_COLORS[i % len(OPT_COLORS)]
        marker = markers[i % len(markers)]
        
        # 标签处理
        if "Label" in opt_df.columns:
            label = str(row["Label"])
        else:
            label = str(row["Run No"]) if "Run No" in row else f"Opt-{i+1}"

        # 误差带 (Opt)
        ax.fill_between(time_hours, mean_vals - std_vals, mean_vals + std_vals,
                        color=color, alpha=0.2, linewidth=0, zorder=9)
        
        # 主线条 (粗且深) + Marker
        line, = ax.plot(time_hours, mean_vals, color=color, linewidth=2.5,
                        marker=marker, markersize=7, markeredgecolor='white', markeredgewidth=1.0,
                        label=label, zorder=10)
        
        opt_handles.append(line)

    # --- 3. 美化坐标轴 ---
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cumulative amount of ibu release (µg/cm²)")
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # 轻微网格
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='#d9d9d9', alpha=0.5, zorder=0)
    
    # 组合图例
    final_handles = lines_handles + opt_handles
    ncol = 1 if len(final_handles) < 6 else 2
    ax.legend(handles=final_handles, loc='best', ncol=ncol)
    
    if title_text:
        ax.set_title(title_text, pad=15)

# ================= 5. 生成 Figure 1 =================
fig1, ax1 = plt.subplots(figsize=(7, 5))

bg_data_1 = [
    (df_train, COLOR_TRAIN, "Train set"),
    (df_test,  COLOR_TEST,  "Test set")
]

# 修正后的调用：去掉了多余的空列表参数
plot_release_profile(ax1, bg_data_1, df_opt)

save_path1 = os.path.join(out_dir, "Fig1_Train_vs_Test_vs_Opt.png")
fig1.tight_layout()
fig1.savefig(save_path1, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path1}")

# ================= 6. 生成 Figure 2 =================
fig2, ax2 = plt.subplots(figsize=(7, 5))

bg_data_2 = [
    (df_all, COLOR_HISTORY, "Historical data")
]

# 修正后的调用
plot_release_profile(ax2, bg_data_2, df_opt)

save_path2 = os.path.join(out_dir, "Fig2_History_vs_Opt.png")
fig2.tight_layout()
fig2.savefig(save_path2, dpi=300, bbox_inches="tight")
print(f"Saved: {save_path2}")

plt.show()