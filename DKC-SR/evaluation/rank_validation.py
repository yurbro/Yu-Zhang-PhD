#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranking validation (no CLI). Run directly.
- Loads TEST predictions (must contain: Q_obs, Q_pred, time_h; optional: Run No/id, C1,C2,C3)
- Focuses on Q(6h) ranking: Spearman ρ, Kendall τ-b, Pairwise (with binomial p), Hit@k
- Also outputs: bump chart, pairwise heatmap, ρ(t) curve, gains curve
- If enough samples / columns present, will add Monte-Carlo 30/6, Extreme test, Region-out.
"""

import os, math, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== USER CONFIG (直接改这里即可) ======
DEFAULT_TEST_PATH = r"Symbolic Differential/SymODE/evaluation/pred_test.csv"   # 你的测试CSV
DEFAULT_OUT_DIR   = r"Symbolic Differential/SymODE/evaluation/rank_report"                                           # 输出目录
MC_REPS = 20        # Monte-Carlo 30/6 重复次数（只有样本≥12时才会触发）
SEED    = 20251009  # 随机种子
# ==========================================

# ---- 可选：若默认路径不存在，弹出文件对话框（不需要命令行） ----
def _pick_file_dialog(title="Choose file"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=[("CSV files","*.csv"),("All files","*.*")])
        return path if path else None
    except Exception:
        return None

def _pick_dir_dialog(title="Select output folder"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askdirectory(title=title)
        return path if path else None
    except Exception:
        return None

# ------------------------- Metrics -------------------------
def kendall_tau_b(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x); conc = disc = 0; ties_x = ties_y = 0
    for i in range(n-1):
        for j in range(i+1, n):
            dx = np.sign(x[i] - x[j]); dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                ties_x += 1; ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            else:
                prod = dx * dy
                if prod > 0: conc += 1
                elif prod < 0: disc += 1
    denom = math.sqrt((conc + disc + ties_x) * (conc + disc + ties_y))
    if denom == 0: return 0.0
    return (conc - disc) / denom

def spearman_rho(x, y):
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    xz = (rx - rx.mean()) / (rx.std(ddof=0) if rx.std(ddof=0)!=0 else 1.0)
    yz = (ry - ry.mean()) / (ry.std(ddof=0) if ry.std(ddof=0)!=0 else 1.0)
    return float(np.mean(xz*yz))

def pairwise_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return float("nan"), 0, 0
    M = n*(n-1)//2
    correct = 0
    for i in range(n-1):
        for j in range(i+1, n):
            dtrue = y_true[i] - y_true[j]
            dpred = y_pred[i] - y_pred[j]
            if dtrue == 0 or dpred == 0:
                M -= 1  # ties ignored
            else:
                if dtrue * dpred > 0:
                    correct += 1
    if M <= 0: return float("nan"), 0, 0
    return correct / M, correct, M

def pairwise_p_value(correct, M):
    p = 0.0
    for k in range(correct, M+1):
        p += math.comb(M, k) * (0.5 ** M)
    return float(min(1.0, max(0.0, p)))

def precision_recall_hit_at_k(y_true, y_pred, k=2):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    idx_pred = np.argsort(-y_pred)
    idx_true = np.argsort(-y_true)
    set_pred_topk = set(idx_pred[:k].tolist())
    set_true_topk = set(idx_true[:k].tolist())
    inter = set_pred_topk & set_true_topk
    prec = len(inter) / max(1, k)
    rec  = len(inter) / max(1, k)
    hit  = 1.0 if idx_true[0] in set_pred_topk else 0.0
    return prec, rec, hit

def compute_rank_metrics(y_true, y_pred, k=2):
    rho = spearman_rho(y_true, y_pred)
    tau = kendall_tau_b(y_true, y_pred)
    pa, corr, M = pairwise_accuracy(y_true, y_pred)
    p_pa = pairwise_p_value(corr, M) if M>0 else float("nan")
    prec_k, rec_k, hit_k = precision_recall_hit_at_k(y_true, y_pred, k=k) if len(y_true)>=k else (float("nan"),)*3
    return dict(
        N=len(y_true), rho=rho, tau_b=tau, pairwise=pa, pairwise_correct=corr, pairwise_M=M,
        pairwise_p=p_pa, prec_at_k=prec_k, rec_at_k=rec_k, hit_at_k=hit_k, k=k
    )

# ------------------------- Plots -------------------------
def bump_chart(y_true, y_pred, path, title="Bump chart (ranking alignment)"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    r_true = pd.Series(y_true).rank(ascending=False, method="first").astype(int).to_numpy()
    r_pred = pd.Series(y_pred).rank(ascending=False, method="first").astype(int).to_numpy()
    order = np.argsort(r_true)
    x = np.arange(1, n+1)
    plt.figure()
    plt.plot(x, r_pred[order], marker="o")
    plt.plot(x, x)
    plt.xlabel("True rank (best→worst)", fontsize=12)
    plt.ylabel("Predicted rank", fontsize=12)
    plt.title(title, fontsize=14)
    plt.savefig(path, bbox_inches="tight"); plt.close()

def pairwise_heatmap(y_true, y_pred, path, title="Pairwise agreement heatmap (1=agree, 0=disagree)"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    idx = np.argsort(-y_true)
    yt = y_true[idx]; yp = y_pred[idx]
    n = len(yt)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0.5
            else:
                dt = yt[i] - yt[j]
                dp = yp[i] - yp[j]
                if dt == 0 or dp == 0:
                    M[i, j] = 0.5
                else:
                    M[i, j] = 1.0 if dt * dp > 0 else 0.0
    plt.figure()
    plt.imshow(M, origin="upper", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xlabel("Items (sorted by true descending)", fontsize=12)
    plt.ylabel("Items (sorted by true descending)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.savefig(path, bbox_inches="tight"); plt.close()

def dot_hist(values, path, title, xlabel):
    arr = np.array(values, dtype=float)
    plt.figure()
    plt.hist(arr, bins=min(10, max(3, len(arr)//3)))
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.savefig(path, bbox_inches="tight"); plt.close()

def line_xy(x, y, path, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title, fontsize=14); plt.xlabel(xlabel, fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.savefig(path, bbox_inches="tight"); plt.close()

def gains_curve(y_true, y_pred, path, title="Gains curve for Q(6h)"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < 4:  # too tiny for a meaningful curve
        return
    thresh_idx = max(1, int(np.ceil(0.25 * n)))
    idx_true = np.argsort(-y_true)
    good_set = set(idx_true[:thresh_idx].tolist())
    idx_pred = np.argsort(-y_pred)
    xs = []; ys = []; cum_good = 0
    for k in range(1, n+1):
        if idx_pred[k-1] in good_set:
            cum_good += 1
        xs.append(k / n)
        ys.append(cum_good / len(good_set))
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.plot([0,1], [0,1])
    plt.xlabel("Cumulative fraction (by predicted order)", fontsize=12)
    plt.ylabel("Recall of true top-25%", fontsize=12)
    plt.title(title, fontsize=14)
    plt.savefig(path, bbox_inches="tight"); plt.close()

# ------------------------- Data utils -------------------------
# ---- AUC helpers (add after region_out_metrics) ----
def _group_id_column(df: pd.DataFrame):
    # 优先用明确的 id；没有就尝试 (C1,C2,C3) 组合作为分组键
    id_col = detect_id_col(df)
    if id_col and id_col in df.columns:
        return id_col, None
    for cols in (["C1","C2","C3"],):
        if all(c in df.columns for c in cols):
            return None, cols
    raise ValueError("找不到样本分组列。请在 CSV 里提供 'Run No' 或 'id'，"
                     "或同时包含 C1,C2,C3 以便按配方分组。")

def compute_auc_table(df: pd.DataFrame, tmax: float = 6.0, anchor_zero: bool = True):
    """
    返回每个样本的 AUC_obs/AUC_pred（时间用小时，梯形法积分到 tmax）。
    anchor_zero=True 时，若该样本最早时间点 > 0，会在 (t=0,Q=0) 处补一个锚点（物理合理）。
    """
    id_col, key_cols = _group_id_column(df)
    groups = (df.groupby(id_col) if id_col else df.groupby(key_cols, dropna=False))
    rows = []
    for gid, sub in groups:
        sub = sub.sort_values("time_h")
        # 截断到 tmax
        sub = sub[sub["time_h"] <= tmax].copy()
        if sub.empty:
            continue
        t = sub["time_h"].to_numpy(dtype=float)
        q_obs = sub["Q_obs"].to_numpy(dtype=float)
        q_pred = sub["Q_pred"].to_numpy(dtype=float)
        # optional anchor at t=0
        if anchor_zero and (len(t) == 0 or t[0] > 0.0):
            t = np.insert(t, 0, 0.0)
            q_obs = np.insert(q_obs, 0, 0.0)
            q_pred = np.insert(q_pred, 0, 0.0)
        auc_obs = float(np.trapz(q_obs, t))
        auc_pred = float(np.trapz(q_pred, t))
        row = {"AUC_obs": auc_obs, "AUC_pred": auc_pred}
        if id_col:
            row[id_col] = gid
        else:
            for c, v in zip(key_cols, gid if isinstance(gid, tuple) else (gid,)):
                row[c] = v
        rows.append(row)
    return pd.DataFrame(rows)

def detect_id_col(df):
    for c in ["Run No", "record_idx", "id", "run_id", "sample_id"]:
        if c in df.columns:
            return c
    return None

def load_pred_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ["q_obs","q_pred","time_h"]
    if not all(k in cols for k in need):
        raise ValueError(f"{path} must contain at least columns: Q_obs, Q_pred, time_h")
    df = df.rename(columns={cols["q_obs"]:"Q_obs", cols["q_pred"]:"Q_pred", cols["time_h"]:"time_h"})
    return df

def pick_q6(df):
    d6 = df[df["time_h"] == 6.0].copy()
    d6 = d6.dropna(subset=["Q_obs","Q_pred"])
    return d6

def metrics_by_time(df):
    rows=[]
    for t, sub in df.groupby("time_h"):
        if len(sub) < 3:  # min size
            continue
        m = compute_rank_metrics(sub["Q_obs"].to_numpy(), sub["Q_pred"].to_numpy(), k=2)
        m["time_h"] = float(t)
        rows.append(m)
    if rows:
        return pd.DataFrame(rows).sort_values("time_h")
    return pd.DataFrame()

def region_out_metrics(d6, outpath):
    if not all(c in d6.columns for c in ["C1","C2","C3"]):
        return
    D = d6.copy()
    for c in ["C1","C2","C3"]:
        mi, ma = float(D[c].min()), float(D[c].max())
        D[f"_{c}n"] = (D[c] - mi) / (ma - mi + 1e-12)
    D["_b1"] = (D["_C1n"] >= 0.5).astype(int)
    D["_b2"] = (D["_C2n"] >= 0.5).astype(int)
    D["_b3"] = (D["_C3n"] >= 0.5).astype(int)
    D["_cube"] = D["_b1"].astype(str) + D["_b2"].astype(str) + D["_b3"].astype(str)
    rows=[]
    for cube, sub in D.groupby("_cube"):
        if len(sub) < 4: 
            continue
        m = compute_rank_metrics(sub["Q_obs"].to_numpy(), sub["Q_pred"].to_numpy(), k=2)
        m["cube"] = cube; m["n_test"] = int(len(sub))
        rows.append(m)
    if rows:
        pd.DataFrame(rows).sort_values("cube").to_csv(outpath, index=False)

# ------------------------- Main (no CLI) -------------------------
def main():
    # 1) 读取 TEST 路径（默认路径若不存在，则弹对话框）
    test_path = Path(DEFAULT_TEST_PATH)
    if not test_path.exists():
        picked = _pick_file_dialog("请选择测试集预测文件（pred_test.csv）")
        if not picked:
            print("未选择测试文件，退出。"); return
        test_path = Path(picked)

    # 2) 输出目录（默认 or 对话框）
    outdir = Path(DEFAULT_OUT_DIR)
    if not outdir.exists():
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except Exception:
            picked = _pick_dir_dialog("请选择输出文件夹")
            if not picked:
                print("未选择输出目录，退出。"); return
            outdir = Path(picked)

    print(f"[INFO] Using TEST file: {test_path}")
    print(f"[INFO] Output dir     : {outdir.resolve()}")

    # 3) 读取数据
    data = load_pred_csv(str(test_path))
    data["_dataset"] = "test"

    # 4) 取 Q6 表
    d6 = pick_q6(data)
    d6.to_csv(outdir / "Q6_table_test.csv", index=False)

    # 5) 测试集 Q6 排序指标 + 图
    y_true = d6["Q_obs"].to_numpy()
    y_pred = d6["Q_pred"].to_numpy()
    metrics = compute_rank_metrics(y_true, y_pred, k=4)
    with open(outdir / "summary_test_Q6.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    bump_chart(y_true, y_pred, outdir / "bump_test_Q6.png", title="Test Q(6h): Bump chart")
    pairwise_heatmap(y_true, y_pred, outdir / "pairwise_heatmap_test_Q6.png", title="Test Q(6h): Pairwise heatmap")
    gains_curve(y_true, y_pred, outdir / "gains_curve_Q6.png", title="Test Q(6h): Gains curve (top-25%)")

    # 6) 时间维度的秩相关（0.5–6h），证明趋势不是只在6h“侥幸对”
    mbt = metrics_by_time(data)
    if not mbt.empty:
        mbt.to_csv(outdir / "metrics_by_time.csv", index=False)
        plt.figure()
        plt.plot(mbt["time_h"].values, mbt["rho"].values, marker="o")
        plt.title("Spearman ρ by time"); plt.xlabel("Time (h)"); plt.ylabel("Spearman ρ")
        plt.savefig(outdir / "rho_by_time.png", bbox_inches="tight"); plt.close()

    # 7) 如果带 C1–C3，可做 Region-out（2×2×2分箱）
    region_out_metrics(d6, outdir / "metrics_region_out.csv")

        # --- 7.5) AUC(0-6h) 表 + 排序指标 + 图 ---
    try:
        auc_df = compute_auc_table(data, tmax=6.0, anchor_zero=True)
        auc_path = outdir / "AUC_table_test.csv"
        auc_df.to_csv(auc_path, index=False)

        y_auc_true = auc_df["AUC_obs"].to_numpy()
        y_auc_pred = auc_df["AUC_pred"].to_numpy()
        auc_metrics = compute_rank_metrics(y_auc_true, y_auc_pred, k=2)
        with open(outdir / "summary_test_AUC.json", "w", encoding="utf-8") as f:
            json.dump(auc_metrics, f, indent=2)

        bump_chart(y_auc_true, y_auc_pred, outdir / "bump_test_AUC.png",
                   title="Test AUC(0–6h): Bump chart")
        pairwise_heatmap(y_auc_true, y_auc_pred, outdir / "pairwise_heatmap_test_AUC.png",
                         title="Test AUC(0–6h): Pairwise heatmap")
        gains_curve(y_auc_true, y_auc_pred, outdir / "gains_curve_AUC.png",
                    title="Test AUC(0–6h): Gains curve (top-25%)")
    except Exception as e:
        print("[WARN] AUC 计算失败：", e)

    # 8) 清单
    manifest = {
        "Q6_table_test": str(outdir / "Q6_table_test.csv"),
        "summary_test_Q6": str(outdir / "summary_test_Q6.json"),
        "bump_test_Q6": str(outdir / "bump_test_Q6.png"),
        "heatmap_test_Q6": str(outdir / "pairwise_heatmap_test_Q6.png"),
        "gains_curve_Q6": str(outdir / "gains_curve_Q6.png"),
        "metrics_by_time": str(outdir / "metrics_by_time.csv"),
        "rho_by_time": str(outdir / "rho_by_time.png"),
        "region_out_metrics": str(outdir / "metrics_region_out.csv"),
        "AUC_table_test": str(outdir / "AUC_table_test.csv"),
        "summary_test_AUC": str(outdir / "summary_test_AUC.json"),
        "bump_test_AUC": str(outdir / "bump_test_AUC.png"),
        "heatmap_test_AUC": str(outdir / "pairwise_heatmap_test_AUC.png"),
        "gains_curve_AUC": str(outdir / "gains_curve_AUC.png"),

    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[OK] 完成。关键文件：")
    for k,v in manifest.items():
        if v and Path(v).exists():
            print(" -", k, "=>", v)

    # 9) 尝试自动打开输出目录
    try:
        if sys.platform.startswith("win"):
            os.startfile(outdir.resolve())
        elif sys.platform == "darwin":
            os.system(f'open "{outdir.resolve()}"')
        else:
            os.system(f'xdg-open "{outdir.resolve()}"')
    except Exception:
        pass

if __name__ == "__main__":
    main()
