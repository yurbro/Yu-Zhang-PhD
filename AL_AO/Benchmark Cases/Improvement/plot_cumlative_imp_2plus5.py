# -*- coding: utf-8 -*-
"""
绘制 5 个benchmark（3D/5D）的累计 improvement 图
布局：每个函数一行，两列（左 3D | 右 5D）
输出两版：SE 阴影 & 95% CI 阴影
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 配置区 ====
file_path = r"Multi-Objective Optimisation/Benchmark/Improvement/imp_summary.xlsx"   # 改成你的文件路径
out_dir = r"Multi-Objective Optimisation/Benchmark/Improvement/plot"                 # 输出目录
sheet_order = ["Ackley", "Rastrigin", "Zakharov", "Griewank", "Sphere"]              # 目标显示名称
pretty_names = {"Ackley":"Ackley", "Rastrigin":"Rastrigin",
                "Zakharov":"Zakharov", "Griewank":"Griewank", "Sphere":"Sphere"}
# 兼容：若 Excel 里拼成了 "Grirwank"
sheet_synonyms = {
    "Griewank": ["Griewank", "Grirwank"]
}
methods = ["Proposed", "EI", "HV", "Random", "BO-EI", "BO-UCB", "BO-POI"]
n_seeds = 10   # 你的实验用 10 个 seeds
label_map = {"Proposed": "Adaptive"}

# 如果没有 scipy，就用一个简单的 t_{0.975, df=n_seeds-1} 近似：
def get_t_crit_975(n: int) -> float:
    df = max(n - 1, 1)
    # 你是 n=10 -> df=9：2.262。否则粗略给个近似；n>=30 用 1.96
    if df == 9:
        return 2.262
    return 1.96 if n >= 30 else 2.2  # 粗略值，足够作为可视化带宽

def resolve_sheet_name(xls, target_name: str) -> str:
    """在 Excel 中找到目标 sheet；支持别名（如 Griewank/Grirwank）"""
    available = set(xls.sheet_names)
    if target_name in available:
        return target_name
    for alias in sheet_synonyms.get(target_name, []):
        if alias in available:
            return alias
    # 最后兜底：直接返回原名（让报错更直观）
    return target_name

def plot_cumulative_band(band_mode: str, out_path: str, match_row_ylim: bool=False):
    """
    band_mode: 'SE' / 'CI95' / 'SD'
    match_row_ylim: True 时，对同一行（同一函数的 3D/5D）匹配 y 轴范围，便于对比
    """
    # 统一风格（和你之前的图类似）
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 14,
    })

    # 5 行 × 2 列（左列 3D，右列 5D）
    fig, axes = plt.subplots(len(sheet_order), 2,
                             figsize=(12, 20),  # 两列+5行的尺寸
                             sharex=True, sharey=False)

    all_handles = None
    all_labels = None
    tcrit = get_t_crit_975(n_seeds)

    xls = pd.ExcelFile(file_path)

    for row_idx, sheet_key in enumerate(sheet_order):
        sheet = resolve_sheet_name(xls, sheet_key)
        df = pd.read_excel(file_path, sheet_name=sheet, header=[0, 1])

        # 取第一列为迭代号；过滤掉非数字的行（比如首行 "Iter"）
        iter_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        mask = iter_raw.notna()
        df = df[mask].reset_index(drop=True)
        iterations = iter_raw[mask].astype(int).to_numpy()

        # 每行的两个轴：左 3D，右 5D
        for col_idx, dim in enumerate(["3D", "5D"]):
            ax = axes[row_idx, col_idx]

            # 该维度是否存在
            has_dim = any((isinstance(col, tuple) and col[0] == dim) for col in df.columns)
            if not has_dim:
                ax.axis("off")
                continue

            for m in methods:
                mean_key = (dim, m)
                std_key  = (dim, m + ".1")
                if mean_key not in df.columns or std_key not in df.columns:
                    continue

                mean = pd.to_numeric(df[mean_key], errors='coerce').fillna(0.0).to_numpy()
                std  = pd.to_numeric(df[std_key],  errors='coerce').fillna(0.0).to_numpy()

                # 累计均值 & 累计SD（方差相加开方；假定各迭代增量独立）
                cum_mean = np.cumsum(mean)
                cum_sd   = np.sqrt(np.cumsum(std ** 2))

                # 阴影带宽
                if band_mode.upper() == "SE":
                    band = cum_sd / np.sqrt(n_seeds)
                elif band_mode.upper() == "CI95":
                    band = (cum_sd / np.sqrt(n_seeds)) * tcrit
                else:  # 'SD'
                    band = cum_sd

                # line, = ax.plot(iterations, cum_mean, linewidth=2, label=m)
                line, = ax.plot(iterations, cum_mean, linewidth=2,label=label_map.get(m, m))   # ← Proposed 显示为 Adaptive
                ax.fill_between(iterations, cum_mean - band, cum_mean + band, alpha=0.15)

            ax.set_title(f"{pretty_names.get(sheet_key, sheet_key)}-{dim}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cumulative Improvement")
            ax.grid(True, linestyle='--', alpha=0.6)

            if all_handles is None and all_labels is None:
                all_handles, all_labels = ax.get_legend_handles_labels()

        # 可选：对同一行的两个子图统一 y 轴范围（让 3D/5D 更直接比较）
        if match_row_ylim:
            ymins, ymaxs = [], []
            for col_idx in [0, 1]:
                y0, y1 = axes[row_idx, col_idx].get_ylim()
                ymins.append(y0); ymaxs.append(y1)
            ymin, ymax = min(ymins), max(ymaxs)
            for col_idx in [0, 1]:
                axes[row_idx, col_idx].set_ylim(ymin, ymax)

    # if all_handles and all_labels:
    #     fig.legend(all_handles, all_labels, loc="lower center",
    #                ncol=len(all_labels), frameon=False, fontsize=12,
    #                handlelength=2.5, handletextpad=0.6, columnspacing=1.2)

    # # 给底部图例留空间（字体变大时可适当调高 0.06 -> 0.10）
    # fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    # os.makedirs(out_dir, exist_ok=True)
    # fig.savefig(out_path, dpi=400, bbox_inches='tight')
    # plt.close(fig)

    # 放在你 if all_handles and all_labels: 的那一段
    legend = fig.legend(all_handles, all_labels,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.02),   # ← 把图例锚在离底部 2% 的位置
                        ncol=len(all_labels),
                        frameon=False, fontsize=14,
                        handlelength=2.5, handletextpad=0.6, columnspacing=1.2)

    # 预留更少的底部空间（给图例 + 一点缓冲）
    fig.tight_layout(rect=[0, 0.034, 1, 1])   # ← 原来是 0.10 和 1.00
    # 保存时可以稍微减小 pad
    fig.savefig(out_path, dpi=400, bbox_inches="tight")

if __name__ == "__main__":
    out_se = os.path.join(out_dir, "benchmark_cumulative_improvement_rows2cols_SE.png")
    out_ci = os.path.join(out_dir, "benchmark_cumulative_improvement_rows2cols_CI95.png")
    # match_row_ylim=True 时，每一行(3D/5D) y轴一致，更便于对比
    plot_cumulative_band("SE", out_se, match_row_ylim=False)
    plot_cumulative_band("CI95", out_ci, match_row_ylim=False)
    print("Done ->", out_se, " & ", out_ci)
