# import pandas as pd
# import matplotlib.pyplot as plt

# # 设置文件路径（你只需要替换成本地路径即可）
# file_path = 'Multi-Objective Optimisation\Dataset\Comparison of raw and optimisation.xlsx'

# # 读取所有 sheet
# xlsx = pd.ExcelFile(file_path)
# sheet_names = xlsx.sheet_names

# # 初始化绘图
# plt.figure(figsize=(10, 7))

# # 定义颜色和符号
# colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
# markers = [ 'x', '*', 'o', 's', '^', 'D', 'P', '+']

# # 遍历 sheet 并绘图
# for idx, sheet in enumerate(sheet_names):
#     df = pd.read_excel(file_path, sheet_name=sheet)
    
#     if sheet == 'Raw':
#         x = df['Mean']
#         y = df['Std']
#         label = 'Raw'
#     else:
#         x = df['Mean']
#         y = df['Std']
#         label = sheet

#     plt.scatter(x, y, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=label)

# # 添加图例和标签
# plt.xlabel('Mean')
# plt.ylabel('Standard Deviation')
# plt.title('Comparison of Raw and Optimisation Results')
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig('Multi-Objective Optimisation\Pareto_animation/Pareto_fronts_comparison.png', dpi=300)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# file_path = 'Multi-Objective Optimisation\Dataset\Comparison of raw and optimisation.xlsx'
# xlsx = pd.ExcelFile(file_path)
# sheet_names = xlsx.sheet_names

# plt.figure(figsize=(10, 7))
# colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
# markers = [ 'x', '*', 'o', 's', '^', 'D', 'P', '+']

# max_raw_x = None  # 初始化最大Mean+Std的位置

# for idx, sheet in enumerate(sheet_names):
#     df = pd.read_excel(file_path, sheet_name=sheet)
#     df.columns = df.columns.str.strip().str.lower()  # 清除列名前后空格

#     if not {'mean', 'std'}.issubset(df.columns):
#         print(f"⚠️ Sheet '{sheet}' is missing 'Mean' or 'Std' column.")
#         continue

#     x = df['mean']
#     y = df['std']

#     label = 'Raw' if sheet.lower() == 'raw' else sheet
#     plt.scatter(x, y, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=label)

#     # 找出 Raw 中 Mean + Std 最大的点
#     if sheet.lower() == 'raw':
#         raw_combined = df['mean'] + df['std']
#         max_raw_x = df.loc[raw_combined.idxmax(), 'mean']

# # 添加垂直虚线
# if max_raw_x is not None:
#     plt.axvline(x=max_raw_x, color='gray', linestyle='--', linewidth=1.5, label='Raw: Max <Mean+Std>')

# plt.xlabel('Mean')
# plt.ylabel('Standard Deviation')
# plt.title('Comparison of Raw and Optimisation Results')
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig('Multi-Objective Optimisation\Pareto_animation/Pareto_fronts_comparison.png', dpi=300)
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_with_named_annotations(file_path, annotation_names=None):
    """
    画出 Mean vs Std 的散点图，支持多个 sheet, 支持通过名称标注点。
    :param file_path: Excel 文件路径
    :param annotation_names: 可选, list, 包含你想标注的样本名称 (如 "Cf1")
    """
    xlsx = pd.ExcelFile(file_path)
    sheet_names = xlsx.sheet_names

    # plt.figure(figsize=(10, 8))
    colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    markers = [ 'x', '*', 'o', 's', '^', 'D', 'P', '+']

    max_raw_x = None
    all_points = {}  # 用于存储所有点坐标及其名称

    for idx, sheet in enumerate(sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()

        # 找可能的名字列
        name_col = next((col for col in df.columns if 'formula' in col or 'num' in col), None)

        if name_col is None or not {'mean', 'std'}.issubset(df.columns):
            print(f"⚠️ Sheet '{sheet}' 缺少所需列")
            continue

        x = df['mean']
        y = df['std']
        names = df[name_col]

        label = 'Raw' if sheet.lower() == 'raw' else sheet
        plt.scatter(x, y, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=label)

        for xi, yi, ni in zip(x, y, names):
            all_points[str(ni).strip()] = (xi, yi)

        if sheet.lower() == 'raw':
            # raw_combined = x + y
            raw_combined = x
            max_raw_x = x.loc[raw_combined.idxmax()]

    # 添加指定标注
    if annotation_names:
        # 为每个高亮点分配不同颜色
        highlight_colors = ['gold', 'limegreen', 'deepskyblue', 'crimson', 'violet', 'orange', 'brown', 'navy']
        for idx, name in enumerate(annotation_names):
            coord = all_points.get(name)
            if coord:
                color = highlight_colors[idx % len(highlight_colors)]
                plt.scatter(*coord, color=color, marker='o', s=40, edgecolor='black', zorder=5, label=f"Selected: {name}")
                # plt.annotate(name, coord, textcoords="offset points", xytext=(5, 5), ha='left')  # 如需显示文本，取消注释
            else:
                print(f"⚠️ 无法找到名称为 '{name}' 的数据点。")

    # 添加 Raw 最大线
    if max_raw_x is not None:
        plt.axvline(x=max_raw_x, color='gray', linestyle='--', linewidth=1.5, label='The incumbent best')

    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Comparison of Raw and Optimisation Results')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    # plt.savefig('Multi-Objective Optimisation/Pareto_animation/Pareto_fronts_comparison_selected_NEI.png', dpi=300, bbox_inches='tight')
    plt.savefig('Multi-Objective Optimisation/Pareto_animation/Pareto_fronts_comparison_selected_NEI-FS-Corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

# call
file_path = 'Multi-Objective Optimisation/Dataset/Comparison of raw and optimisation in IVRT FS - Corrected.xlsx'
# annotation_names = ['P_2', 'P_9', 'P_8', 'P_10', 'P_6']  # results of NEI
# annotation_names = ['P_4', 'P_5', 'P_6', 'P_7', 'P_8']  # results of HV
# annotation_names = ['P_3', 'P_5', 'P_7', 'P_6', 'P_8']  # results of HV-FS 
# annotation_names = ['P_1', 'P_10', 'P_9']  # results of NEI-FS FOR EPOCH-1
# annotation_names = ['P_2', 'P_5']  # results of NEI-FS-Corrected FOR EPOCH-1
annotation_names = ['P_2', 'P_3', 'P_5', 'P_4', 'P_9']  # results of NEI-LHS-Corrected FOR EPOCH-1


plot_with_named_annotations(file_path, annotation_names)

# plot the Noisy EI value by using colume 

