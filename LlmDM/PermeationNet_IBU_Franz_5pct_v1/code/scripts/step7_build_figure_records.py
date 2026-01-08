import json, ast, re
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

ENDPTS = Path("GenAI/outputs/figure_digitized_endpoints.csv")
FORM_FLAT = Path("GenAI/outputs/figure_formulations_v1_flat.csv")
MAP_CSV = Path("GenAI/outputs/figure_curve_formulation_map.csv")

OUT_DATA = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure.csv")
OUT_UNMAPPED = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_unmapped.csv")

def safe_load(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def unit_normalize_amount_per_area(value: Optional[float], unit: str) -> Optional[float]:
    """
    Standardize to ug/cm^2 when possible.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    u = (unit or "").lower().replace("µ", "u").replace(" ", "")
    v = float(value)
    # common patterns
    if "mg/cm2" in u or "mg/cm^2" in u:
        return v * 1000.0
    if "ug/cm2" in u or "ug/cm^2" in u:
        return v
    if "ng/cm2" in u or "ng/cm^2" in u:
        return v / 1000.0
    return None

def is_api_5pct(api_val, api_unit, api_basis) -> bool:
    try:
        v = float(api_val)
    except Exception:
        return False
    u = (str(api_unit) or "").lower()
    b = (str(api_basis) or "").lower()
    if "%" in u and ("w/w" in u or "w/w" in b or "wt" in u or "wt" in b or b == ""):
        return abs(v - 5.0) < 1e-6
    return False

def main(strict_5pct: bool = True):
    df_end = pd.read_csv(ENDPTS)
    df_end = df_end[df_end["status"] == "ok"].copy()
    # only amount
    if "y_kind" in df_end.columns:
        df_end = df_end[df_end["y_kind"] != "percent"].copy()

    df_map = pd.read_csv(MAP_CSV)
    df_fm = pd.read_csv(FORM_FLAT)

    # join endpoints with mapping
    df = df_end.merge(
        df_map,
        how="left",
        left_on=["doi", "curve_id"],
        right_on=["doi", "curve_id"]
    )

    # split mapped/unmapped
    unmapped = df[df["mapped_formulation_label"].isna()].copy()
    mapped = df[df["mapped_formulation_label"].notna()].copy()

    # join mapped with formulations
    fm_key = df_fm.copy()
    fm_key["doi"] = fm_key["doi"].astype(str).str.strip()
    fm_key["formulation_label"] = fm_key["formulation_label"].astype(str).str.strip()

    mapped["mapped_formulation_label"] = mapped["mapped_formulation_label"].astype(str).str.strip()
    mapped = mapped.merge(
        fm_key,
        how="left",
        left_on=["doi", "mapped_formulation_label"],
        right_on=["doi", "formulation_label"],
        suffixes=("", "_fm")
    )

    # build v1 figure records
    records = []
    for _, r in mapped.iterrows():
        endpoint_value = r.get("endpoint_value", None)
        endpoint_unit = r.get("endpoint_unit", "")

        y_kind = str(r.get("y_kind", "unknown") or "unknown")
        endpoint_std = None
        endpoint_std_unit = None
        if y_kind == "amount_per_area":
            endpoint_std = unit_normalize_amount_per_area(endpoint_value, endpoint_unit)
            endpoint_std_unit = "ug/cm^2" if endpoint_std is not None else None

        api_ok = is_api_5pct(r.get("api_concentration_value", None),
                             r.get("api_concentration_unit", ""),
                             r.get("api_basis", ""))

        if strict_5pct and not api_ok:
            continue

        records.append({
            "doi": r["doi"],
            "source": "figure",
            "figure_id": r.get("figure_id", ""),
            "page_number": r.get("page_number", None),
            "subplot": r.get("subplot", ""),
            "curve_id": r.get("curve_id", ""),
            "curve_color": r.get("curve_color", ""),
            "curve_label": r.get("curve_label", ""),
            "mapping_status": r.get("mapping_status", ""),
            "formulation_label": r.get("mapped_formulation_label", ""),

            "api_name": r.get("api_name", "ibuprofen"),
            "api_concentration_value": r.get("api_concentration_value", None),
            "api_concentration_unit": r.get("api_concentration_unit", ""),
            "api_basis": r.get("api_basis", ""),
            "api_5pct_confirmed": bool(api_ok),

            "dosage_form": r.get("dosage_form", ""),
            "table_id": r.get("table_id", ""),
            "components_json": r.get("components_json", ""),

            "endpoint_time": r.get("endpoint_time", None),
            "endpoint_time_unit": r.get("endpoint_time_unit", ""),
            "endpoint_kind": y_kind,
            "endpoint_value_raw": endpoint_value,
            "endpoint_unit_raw": endpoint_unit,
            "endpoint_value_ug_per_cm2": endpoint_std,
            "endpoint_unit_std": endpoint_std_unit,
            "image_path": r.get("image_path", ""),
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")

    unmapped.to_csv(OUT_UNMAPPED, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Saved:", OUT_DATA, "rows:", len(df_out))
    print("Saved:", OUT_UNMAPPED, "rows:", len(unmapped))
    if len(df_out) > 0:
        print("\nendpoint_kind counts:\n", df_out["endpoint_kind"].value_counts(dropna=False))
        print("\napi_5pct_confirmed counts:\n", df_out["api_5pct_confirmed"].value_counts(dropna=False))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-strict-5pct", action="store_true", help="Keep records even if API 5% not confirmed in table")
    args = ap.parse_args()
    main(strict_5pct=not args.no_strict_5pct)
