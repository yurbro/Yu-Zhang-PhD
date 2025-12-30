import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0) 路径设置
# =========================
file_path = r"Symbolic Differential/SymODE/pred_model/pred_result.xlsx"   # 改成你的文件路径
out_dir = Path("Symbolic Differential/SymODE/pred_model/pred_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 1) 读数据 & 基本清洗
# =========================
df = pd.read_excel(file_path)

required_cols = {"record_idx", "time_h", "Q_obs", "Q_pred"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in Excel: {missing}. "
                     f"Expected columns: {sorted(required_cols)}")

df = df.dropna(subset=["record_idx", "time_h", "Q_obs", "Q_pred"]).copy()
df["record_idx"] = df["record_idx"].astype(int)
df["time_h"] = df["time_h"].astype(float)
df["Q_obs"] = df["Q_obs"].astype(float)
df["Q_pred"] = df["Q_pred"].astype(float)

# =========================
# 2) 为每个配方补 0h（并把 0h 的 Q_obs/Q_pred 置 0）
# =========================
def add_or_force_zero_timepoint(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()

    # 若已有 0h 点：强制设为 0
    mask0 = np.isclose(out["time_h"].to_numpy(), 0.0)
    if mask0.any():
        out.loc[mask0, ["Q_obs", "Q_pred"]] = 0.0

    # 找出哪些配方缺少 0h
    has0 = out.groupby("record_idx")["time_h"].apply(lambda s: np.any(np.isclose(s.to_numpy(), 0.0)))
    missing_ridx = has0[~has0].index.tolist()

    # 缺少的配方补一行 0h=0
    if missing_ridx:
        add = pd.DataFrame({
            "record_idx": missing_ridx,
            "time_h": 0.0,
            "Q_obs": 0.0,
            "Q_pred": 0.0,
        })
        out = pd.concat([out, add], ignore_index=True)

    # 排序
    out = out.sort_values(["record_idx", "time_h"]).reset_index(drop=True)
    return out

df0 = add_or_force_zero_timepoint(df)

# =========================
# 3) 计算指标：R² & RMSE（整体 + 每配方）
# =========================
def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2, rmse

r2_all, rmse_all = r2_rmse(df0["Q_obs"].to_numpy(), df0["Q_pred"].to_numpy())
print(f"[Overall, incl. 0h] R^2 = {r2_all:.6f}")
print(f"[Overall, incl. 0h] RMSE = {rmse_all:.6f} (same unit as Q)")

# 每配方（可选打印）
by_form = []
for ridx, g in df0.groupby("record_idx"):
    r2_i, rmse_i = r2_rmse(g["Q_obs"].to_numpy(), g["Q_pred"].to_numpy())
    by_form.append((int(ridx), r2_i, rmse_i))
by_form_df = pd.DataFrame(by_form, columns=["record_idx", "R2", "RMSE"]).sort_values("record_idx")
print("\n[Per formulation, incl. 0h]")
print(by_form_df.to_string(index=False))

# =========================
# 4) 图1：Parity plot（预测 vs 测量）
# =========================
plt.figure()
plt.scatter(df0["Q_obs"], df0["Q_pred"], alpha=0.7, c="red", edgecolors="black", s=50)

mn = float(min(df0["Q_obs"].min(), df0["Q_pred"].min()))
mx = float(max(df0["Q_obs"].max(), df0["Q_pred"].max()))
plt.plot([mn, mx], [mn, mx], linewidth=1, color="black", linestyle="--")  # y=x 参考线

plt.xlabel("Measured value (μg/cm²)", fontsize=12)
plt.ylabel("Predicted value (μg/cm²)", fontsize=12)
# plt.title(f"Parity plot (incl. 0h)  R$^2$={r2_all:.3f}, RMSE={rmse_all:.1f}")
plt.tight_layout()
plt.savefig(out_dir / "parity_plot_with_0h.png", dpi=300, bbox_inches="tight")
plt.show()

# # =========================
# # 5) 图2：时间序列对比（按配方）
# # =========================
# plt.figure()

# for ridx, g in df0.groupby("record_idx"):
#     g = g.sort_values("time_h")
#     # 测量：实线 + 圆点；预测：虚线（同色）
#     plt.plot(g["time_h"], g["Q_obs"], marker="o", linestyle="-", label=f"Measured F{int(ridx)}")
#     plt.plot(g["time_h"], g["Q_pred"], linestyle="--", label=f"Predicted F{int(ridx)}")

# plt.xlabel("Time (h)")
# plt.ylabel("Q (μg/cm²)")
# plt.title("Measured vs Predicted over time (incl. 0h)")
# plt.legend(ncol=2, fontsize=9)
# plt.tight_layout()
# plt.savefig(out_dir / "timeseries_compare_with_0h.png", dpi=300, bbox_inches="tight")
# plt.show()

# print(f"\nSaved figures to: {out_dir.resolve()}")
