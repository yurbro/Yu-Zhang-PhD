import pandas as pd
import os

# 文件模板
file_template = r"Symbolic Regression/srloop/data/hall_of_fame_run-{num}_restored_analyzed.csv"

# 存储所有数据
all_data = []


# 用ExcelWriter分别写入不同sheet
output_excel = "Symbolic Regression/srloop/data/complexity_loss_all_runs.xlsx"
with pd.ExcelWriter(output_excel) as writer:
    has_data = False
    for num in range(1, 9):
        file_path = file_template.format(num=num)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df_selected = df[['restored_complexity', 'loss']].copy()
        df_selected.to_excel(writer, sheet_name=f"Run_{num}", index=False)
        has_data = True
    if has_data:
        print(f"已保存到: {output_excel}")
    else:
        print("没有可用的数据文件。")

# 合并所有run的数据到一个表
all_data = []
for num in range(1, 9):
    file_path = file_template.format(num=num)
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        continue
    df = pd.read_csv(file_path)
    df_selected = df[['restored_complexity', 'loss']].copy()
    df_selected['run'] = num
    all_data.append(df_selected)

if all_data:
    result_df = pd.concat(all_data, ignore_index=True)
    output_excel = "Symbolic Regression/srloop/data/complexity_loss_all_runs_merged.xlsx"
    result_df.to_excel(output_excel, index=False)
    print(f"已保存到: {output_excel}")
else:
    print("没有可用的数据文件。")