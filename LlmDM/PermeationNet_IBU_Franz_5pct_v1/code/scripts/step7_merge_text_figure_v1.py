import argparse
from pathlib import Path
import pandas as pd

OUT_DIR = Path("GenAI/outputs")
DEFAULT_OUT = OUT_DIR / "PermeationNet_IBU_Franz_5pct_v1_merged.csv"

def find_text_verified(outputs_dir: Path) -> Path:
    cands = sorted(outputs_dir.glob("*PermeationNet*verified*.csv"))
    if cands:
        return cands[-1]
    # fallback: any verified records
    cands = sorted(outputs_dir.glob("*verified_records*.csv"))
    if cands:
        return cands[-1]
    raise FileNotFoundError("Cannot find a text verified CSV in outputs/. Please pass --text explicitly.")

def find_figure_pass(outputs_dir: Path) -> Path:
    # prefer the non-refined pass file (you already have 4 there)
    p = outputs_dir / "PermeationNet_IBU_Franz_5pct_v1_figure_pass5pct.csv"
    if p.exists():
        return p
    # fallback: qc file and filter
    p2 = outputs_dir / "PermeationNet_IBU_Franz_5pct_v1_figure_qc.csv"
    if p2.exists():
        return p2
    raise FileNotFoundError("Cannot find figure pass/qc CSV in outputs/. Please pass --figure explicitly.")

def normalize_cols(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source

    # common renames if needed
    rename_map = {
        "endpoint_time": "endpoint_time_h",
        "endpoint_time_unit": "endpoint_time_unit",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # ensure key fields exist
    for col in ["doi", "endpoint_time_h", "endpoint_kind"]:
        if col not in df.columns:
            df[col] = None

    # keep a unified endpoint value column
    if "endpoint_value_ug_per_cm2" in df.columns:
        df["endpoint_value_std"] = df["endpoint_value_ug_per_cm2"]
        df["endpoint_unit_std"] = df.get("endpoint_unit_std", "ug/cm^2")
    elif "endpoint_value" in df.columns:
        df["endpoint_value_std"] = df["endpoint_value"]
        df["endpoint_unit_std"] = df.get("endpoint_unit", "")
    else:
        df["endpoint_value_std"] = None
        df["endpoint_unit_std"] = ""

    # unify formulation label if present
    if "formulation_label" not in df.columns:
        for alt in ["vehicle_label", "formulation_id", "formula_id"]:
            if alt in df.columns:
                df["formulation_label"] = df[alt]
                break
        if "formulation_label" not in df.columns:
            df["formulation_label"] = ""

    return df

def main(text_path: str = None, figure_path: str = None, out_path: str = None):
    outputs_dir = OUT_DIR
    text_p = Path(text_path) if text_path else find_text_verified(outputs_dir)
    fig_p = Path(figure_path) if figure_path else find_figure_pass(outputs_dir)
    out_p = Path(out_path) if out_path else DEFAULT_OUT

    df_text = pd.read_csv(text_p)
    df_fig = pd.read_csv(fig_p)

    # If figure input is QC file, filter to 5%
    if "api_5pct_confirmed_v2" in df_fig.columns:
        df_fig = df_fig[df_fig["api_5pct_confirmed_v2"] == True].copy()
    elif "api_5pct_confirmed" in df_fig.columns:
        df_fig = df_fig[df_fig["api_5pct_confirmed"] == True].copy()

    df_text_n = normalize_cols(df_text, "text")
    df_fig_n = normalize_cols(df_fig, "figure")

    # concatenate
    df_all = pd.concat([df_text_n, df_fig_n], ignore_index=True)

    # de-dup heuristic key
    key_cols = ["doi", "formulation_label", "endpoint_kind", "endpoint_time_h", "endpoint_value_std"]
    for c in key_cols:
        if c not in df_all.columns:
            df_all[c] = None

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=key_cols, keep="first")
    after = len(df_all)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_p, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Text verified:", text_p, "rows:", len(df_text))
    print("Figure pass5pct:", fig_p, "rows_used:", len(df_fig))
    print("Merged saved:", out_p, "rows:", len(df_all), f"(dedup {before-after})")
    print("\nsource counts:\n", df_all["source"].value_counts(dropna=False))
    if "endpoint_kind" in df_all.columns:
        print("\nendpoint_kind counts:\n", df_all["endpoint_kind"].value_counts(dropna=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None, help="Path to text verified CSV")
    ap.add_argument("--figure", type=str, default=None, help="Path to figure pass/qc CSV")
    ap.add_argument("--out", type=str, default=None, help="Output merged CSV path")
    args = ap.parse_args()
    main(args.text, args.figure, args.out)
