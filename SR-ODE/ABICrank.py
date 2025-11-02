
import pandas as pd
import numpy as np

def aggregate_aic_bic(
    df,
    aic_col="AIC",
    bic_col="BIC",
    loss_col="loss",
    complexity_col="complexity_recomputed",
    stability_col="score",
    weights_rank={"AIC": 0.5, "BIC": 0.5},
    weights_z={"AIC": 0.5, "BIC": 0.5},
    method="rank",   # "rank"（推荐）或 "zscore"
    tiebreak=("AIC","BIC","loss","complexity","stability")  # 从左到右依次比较，AIC/BIC越小越好，stability越大越好
):
    df = df.copy()

    # --- 1) 计算名次（越小越好）
    df["AIC_rank"] = df[aic_col].rank(method="min", ascending=True)
    df["BIC_rank"] = df[bic_col].rank(method="min", ascending=True)
    df["rank_combined"] = (weights_rank["AIC"] * df["AIC_rank"] +
                           weights_rank["BIC"] * df["BIC_rank"])

    # --- 2) 计算Z-score标准化后的综合分（越大越好）
    # 加上一丢丢数值稳健性
    def z(x): 
        s = np.std(x)
        return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)

    df["AIC_z"] = z(df[aic_col].values.astype(float))
    df["BIC_z"] = z(df[bic_col].values.astype(float))
    df["z_combined"] = (weights_z["AIC"] * -df["AIC_z"] +
                        weights_z["BIC"] * -df["BIC_z"])

    # --- 3) 主排序：method = "rank" 或 "zscore"
    if method == "rank":
        primary_key = "rank_combined"
        ascending = True
    elif method == "zscore":
        primary_key = "z_combined"
        ascending = False
    else:
        raise ValueError("method must be 'rank' or 'zscore'.")

    # --- 4) 设置并列裁决（tie-breakers）
    # 规则：AIC/BIC/loss/complexity 越小越好；stability 越大越好
    # 构造排序键列表
    sort_cols = [primary_key]
    sort_ascs = [ascending]
    for k in tiebreak:
        if k.lower() == "aic": 
            sort_cols.append(aic_col); sort_ascs.append(True)
        elif k.lower() == "bic":
            sort_cols.append(bic_col); sort_ascs.append(True)
        elif k.lower() == "loss":
            sort_cols.append(loss_col); sort_ascs.append(True)
        elif k.lower() == "complexity":
            sort_cols.append(complexity_col); sort_ascs.append(True)
        elif k.lower() == "stability":
            sort_cols.append(stability_col); sort_ascs.append(False)
        else:
            # 如果传了别的列名，就默认升序
            sort_cols.append(k); sort_ascs.append(True)

    df_sorted = df.sort_values(by=sort_cols, ascending=sort_ascs).reset_index(drop=True)
    return df_sorted

# ========= 示例调用 =========
# 三个模型的数据读入 df 后：
df = pd.read_csv("Symbolic Regression/srloop/data/final_physically_passed_models.csv")
# 推荐先用 rank 聚合（稳健），再看 zscore 聚合是否一致（稳健性检验）
df_rank = aggregate_aic_bic(df, method="rank",
                            weights_rank={"AIC":0.5,"BIC":0.5})
print("\n[Rank 加权聚合] 推荐顺序：")
print(df_rank[["restored_equation","AIC","BIC","AIC_rank","BIC_rank","rank_combined"]])

df_z = aggregate_aic_bic(df, method="zscore",
                         weights_z={"AIC":0.5,"BIC":0.5})
print("\n[Z-score 标准化加权] 推荐顺序：")
print(df_z[["restored_equation","AIC","BIC","AIC_z","BIC_z","z_combined"]])
