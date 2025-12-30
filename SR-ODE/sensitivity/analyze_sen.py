# -*- coding: utf-8 -*-
"""
Algorithm-level sensitivity analysis for SR-ODE runs.

Input:  algo_sensitivity.csv  (with columns renamed to standard names)
Expected columns (any superset is fine):
  Hyperparameters: pop_size, ngen, cxpb, tournsize, tree_len_max, init_depth_max, alpha_complexity
  Metrics: seed, size, sec, train_MSE, train_MSE_norm, test_MSE, test_MSE_norm, best_infix, error

Outputs to: algo_sens_report/
  - spearman_summary.csv
  - group_summary.csv
  - decisions.csv
  - time_vs_perf.png
  - box_{hp}.png
  - scatter_{hp}.png
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

CSV = Path(r"Symbolic Differential/SymODE/sensitivity/algo_sensitivity.csv")
OUT = Path(r"Symbolic Differential/SymODE/sensitivity/algo_sens_report")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Load & clean ----------
df = pd.read_csv(CSV)

# keep successful runs
if "error" in df.columns:
    df = df[df["error"].isna()].copy()
if df.empty:
    raise SystemExit("No successful runs in CSV (all rows contain errors).")

# ensure numeric dtypes where expected
for col in ["pop_size","ngen","cxpb","tournsize","tree_len_max","init_depth_max","alpha_complexity",
            "sec","train_MSE","train_MSE_norm","test_MSE","test_MSE_norm","size"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

# metric to analyze (prefer normalized)
metric_col = "test_MSE_norm" if "test_MSE_norm" in df.columns else ("test_MSE" if "test_MSE" in df.columns else None)
if metric_col is None:
    raise SystemExit("Neither test_MSE_norm nor test_MSE found. Please ensure columns are named correctly.")
time_col = "sec" if "sec" in df.columns else None

# Mapping of column names to display labels
label_map = {
    "test_MSE_norm": "Normalised test MSE",
    "alpha_complexity": "Complexity penalty",
    "pop_size": "Population size",
    "ngen": "Number of generations",
    "tournsize": "Tournament size",
    "test_MSE": "test MSE",
    "sec": "Train time (sec)"
}

def get_label(col_name):
    return label_map.get(col_name, col_name)

# hyperparameter columns present
hp_cols = [c for c in ["pop_size","ngen","cxpb","tournsize","tree_len_max","init_depth_max","alpha_complexity"] if c in df.columns]
if not hp_cols:
    raise SystemExit("No hyperparameter columns found. Expected at least one of: " + ", ".join(
        ["pop_size","ngen","cxpb","tournsize","tree_len_max","init_depth_max","alpha_complexity"]))

# ---------- Baseline detection ----------
preferred = {"pop_size":4000,"ngen":250,"cxpb":0.6,"tournsize":5,"tree_len_max":25,"init_depth_max":4,"alpha_complexity":5e-4}
baseline = {}
for h in hp_cols:
    if h in preferred and (df[h] == preferred[h]).any():
        baseline[h] = preferred[h]
    else:
        baseline[h] = df[h].mode().iloc[0]

# ---------- Spearman correlations ----------
def spearman(x, y):
    rx = pd.Series(x).rank(method="average")
    ry = pd.Series(y).rank(method="average")
    if np.std(rx)==0 or np.std(ry)==0: 
        return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

spearman_rows = []
for hp in hp_cols:
    r = spearman(df[hp].values, df[metric_col].values)
    spearman_rows.append({"hyperparam":hp, "spearman_r_vs_"+metric_col: r})
spearman_df = pd.DataFrame(spearman_rows).sort_values("spearman_r_vs_"+metric_col, key=lambda s: s.abs(), ascending=False)
spearman_df.to_csv(OUT/"spearman_summary.csv", index=False)

# ---------- Group summaries & baseline deltas ----------
group_frames = []
for hp in hp_cols:
    cols = [metric_col] + ([time_col] if time_col else [])
    g = df.groupby(hp)[cols].agg(["median","mean","std","count"]).reset_index()
    g.columns = [f"{a}_{b}" if b else a for a,b in g.columns]
    g.insert(0, "hyperparam", hp)
    # deltas to baseline level for this hp
    base_level = baseline.get(hp, None)
    if base_level is not None and (g[hp]==base_level).any():
        base_row = g[g[hp]==base_level].iloc[0]
        g["delta_median_"+metric_col] = (g[metric_col+"_median"] - float(base_row[metric_col+"_median"])) / max(1e-12, float(base_row[metric_col+"_median"]))
        if time_col:
            g["delta_median_"+time_col] = (g[time_col+"_median"] - float(base_row[time_col+"_median"])) / max(1e-12, float(base_row[time_col+"_median"]))
        g["baseline_level"] = base_level
    else:
        g["delta_median_"+metric_col] = np.nan
        if time_col: g["delta_median_"+time_col] = np.nan
        g["baseline_level"] = base_level
    group_frames.append(g)
group_df = pd.concat(group_frames, ignore_index=True)
group_df.to_csv(OUT/"group_summary.csv", index=False)

# ---------- Decisions: best level per HP ----------
decisions = []
for hp in hp_cols:
    g = group_df[group_df["hyperparam"]==hp].copy()
    if g.empty: 
        continue
    sort_cols = [(metric_col+"_median", True)]
    if time_col: sort_cols.append((time_col+"_median", True))
    sort_cols.append((metric_col+"_std", True))
    g_sorted = g.sort_values([c for c,_ in sort_cols], ascending=[asc for _,asc in sort_cols])
    best = g_sorted.iloc[0]
    decisions.append({
        "hyperparam": hp,
        "best_level": best[hp],
        f"best_{metric_col}_median": float(best[metric_col+"_median"]),
        f"best_{metric_col}_std": float(best[metric_col+"_std"]),
        "baseline_level": baseline.get(hp, None),
        "delta_vs_baseline_%": float(best.get("delta_median_"+metric_col, np.nan))*100.0,
        **({f"best_{time_col}_median": float(best[time_col+"_median"])} if time_col else {}),
        **({f"delta_time_vs_baseline_%": float(best.get("delta_median_"+time_col, np.nan))*100.0} if time_col else {}),
        "n_at_best": int(best[metric_col+"_count"]),
    })
dec_df = pd.DataFrame(decisions)
dec_df.to_csv(OUT/"decisions.csv", index=False)

# ---------- Plots ----------
if time_col:
    plt.figure(figsize=(6,4))
    plt.scatter(df[time_col], df[metric_col], alpha=0.6)
    plt.xlabel(get_label(time_col)); plt.ylabel(get_label(metric_col))
    plt.title("Time vs "+get_label(metric_col))
    plt.tight_layout(); plt.savefig(OUT/"time_vs_perf.png", dpi=160); plt.close()

for hp in hp_cols:
    levels = sorted(df[hp].dropna().unique().tolist())
    data = [df[df[hp]==v][metric_col].values for v in levels]
    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=[str(v) for v in levels], showmeans=True)
    plt.xlabel(get_label(hp), fontsize=12); plt.ylabel(get_label(metric_col), fontsize=12)
    # plt.title(f"{get_label(hp)} vs {get_label(metric_col)}")
    plt.tight_layout(); plt.savefig(OUT/f"box_{hp}.png", dpi=160); plt.close()

# ---------- Done ----------
print("[Saved]")
print("  ", OUT/"spearman_summary.csv")
print("  ", OUT/"group_summary.csv")
print("  ", OUT/"decisions.csv")
if time_col: print("  ", OUT/"time_vs_perf.png")
for hp in hp_cols:
    print("  ", OUT/f"box_{hp}.png")
