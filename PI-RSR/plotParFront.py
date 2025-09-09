import pandas as pd
import matplotlib.pyplot as plt
import os

# 路径模板
file_template = r"Symbolic Regression/srloop/data/hall_of_fame_run-{num}_restored_analyzed.csv"

# 设置颜色 (8次 run 不同颜色)
colors = plt.cm.tab10.colors  # tab10 调色板足够8个

# plt.figure(figsize=(10, 6))

for num in range(1, 9):
    file_path = file_template.format(num=num)
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        continue
    
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 提取 loss 和 complexity
    x = df['restored_complexity']
    y = df['loss']
    
    # # 绘制所有点 (圆点)
    # plt.scatter(x, y, color=colors[(num - 1) % len(colors)], 
    #             label=f'Run {num} original', alpha=0.6, s=40)
    
    # Pareto front 点
    pareto_df = df[df['is_pareto_front'] == True].copy()
    if not pareto_df.empty:
        # 按 restored_complexity 排序
        pareto_df = pareto_df.sort_values(by="restored_complexity")
        
        # 画 Pareto front 点 (点)
        plt.scatter(pareto_df['restored_complexity'], pareto_df['loss'],
                    marker='o',
                    facecolors=colors[(num - 1) % len(colors)], alpha=0.7,
                    # edgecolors='red',
                    linewidths=1.5,
                    label=f'Run {num}')
        
        # 画 Pareto front 连线
        plt.plot(pareto_df['restored_complexity'], pareto_df['loss'], 
                 color=colors[(num - 1) % len(colors)], 
                 linestyle='-', linewidth=2, alpha=0.8)

plt.xlabel("Restored Complexity", fontsize=12)
plt.ylabel("Loss", fontsize=12)
# plt.title("Pareto Front across 8 runs", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Symbolic Regression/srloop/visualisation/pareto_front_8runs_pirsr.png", dpi=300, bbox_inches='tight')
plt.show()

# ========== 图 2: 只画 Restored Complexity > 5 ==========
# plt.figure(figsize=(10, 6))

for num in range(1, 9):
    file_path = file_template.format(num=num)
    if not os.path.exists(file_path):
        continue
    
    df = pd.read_csv(file_path)
    
    # 过滤条件
    df_filtered = df[df['restored_complexity'] > 10]
    
    x = df_filtered['restored_complexity']
    y = df_filtered['loss']
    
    # # 所有点
    # plt.scatter(x, y, color=colors[(num - 1) % len(colors)], 
    #             label=f'Run {num}', alpha=0.6, s=40)
    
    # Pareto 点
    pareto_df = df_filtered[df_filtered['is_pareto_front'] == True]
    if not pareto_df.empty:
        plt.scatter(pareto_df['restored_complexity'], pareto_df['loss'], 
                    marker='o', 
                    facecolors=colors[(num - 1) % len(colors)], alpha=0.7,
                    # edgecolors='red', 
                    linewidths=1.5,
                    label=f'Run {num}')
        # 画 Pareto front 连线
        plt.plot(pareto_df['restored_complexity'], pareto_df['loss'], 
                 color=colors[(num - 1) % len(colors)], 
                 linestyle='-', linewidth=2, alpha=0.8)

plt.xlabel("Restored Complexity", fontsize=12)
plt.ylabel("Loss", fontsize=12)
# plt.title("Pareto Front across 8 runs (Restored Complexity > 10)", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Symbolic Regression/srloop/visualisation/pareto_front_8runs_complexity_gt10_pirsr.png", dpi=300, bbox_inches='tight')
plt.show()

# 