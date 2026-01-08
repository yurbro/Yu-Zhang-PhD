import json, ast, re
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import pandas as pd
import numpy as np

IN_FIG = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure.csv")
OUT_QC = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_qc.csv")
OUT_PASS = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_pass5pct.csv")
OUT_MANUAL = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_need_manual_5pct.csv")

def safe_load(x: Any):
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

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def parse_percent(text: str) -> Optional[float]:
    """
    Extract numeric percent value like '5%', '5 %', '5 wt%', '5% w/w'.
    """
    t = text.replace("µ", "u")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|wt%|w/w%|%w/w)", t, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def parse_mg_per_g(text: str) -> Optional[float]:
    """
    Convert mg/g to percent (w/w): 50 mg/g = 5% (since 1 g = 1000 mg, mg/g * 0.1 = %).
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*g", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)) * 0.1
    return None

def parse_g_per_100g(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\s*/\s*100\s*g", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*g\s*per\s*100\s*g", text, flags=re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None

def is_ibuprofen(name_raw: str) -> bool:
    n = norm(name_raw)
    # allow common variants
    return ("ibuprofen" in n) or ("ibu" == n) or ("ibuprofenum" in n)

def infer_api_from_components(components_json: Any) -> Tuple[Optional[float], str, str, str]:
    """
    Return (percent_value, unit, basis, evidence_snippet)
    """
    comps = safe_load(components_json)
    if not isinstance(comps, list):
        return None, "", "", ""

    for c in comps:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name_raw", "") or "")
        if not is_ibuprofen(name):
            continue

        # Try direct numeric field first
        val = c.get("concentration_value", None)
        unit = str(c.get("concentration_unit", "") or "")
        basis = str(c.get("basis", "") or "")
        remark = str(c.get("remark", "") or "")

        # If numeric value exists and unit looks like percent, use it
        try:
            if val is not None and unit:
                v = float(val)
                if "%" in unit or "wt" in unit.lower():
                    ev = f"{name}: {v} {unit} {basis} {remark}".strip()
                    return v, unit, (basis or "w/w"), ev
        except Exception:
            pass

        # Otherwise parse from unit/remark/name strings
        blob = " ".join([name, unit, basis, remark]).strip()
        p = parse_percent(blob)
        if p is None:
            p = parse_mg_per_g(blob)
        if p is None:
            p = parse_g_per_100g(blob)

        if p is not None:
            ev = f"{name}: {blob}".strip()
            return float(p), "%", "w/w", ev

    return None, "", "", ""

def api_is_5pct(v: Optional[float]) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return abs(float(v) - 5.0) < 1e-6

def main():
    df = pd.read_csv(IN_FIG)

    inferred_vals = []
    inferred_units = []
    inferred_basis = []
    inferred_ev = []
    qc_flag = []

    for _, r in df.iterrows():
        # prefer existing api fields if they already look valid
        api_val = r.get("api_concentration_value", None)
        api_unit = str(r.get("api_concentration_unit", "") or "")
        api_basis = str(r.get("api_basis", "") or "")

        existing_ok = False
        try:
            if api_val is not None and api_unit:
                v = float(api_val)
                if "%" in api_unit or "wt" in api_unit.lower():
                    existing_ok = True
                    inferred_vals.append(v)
                    inferred_units.append(api_unit)
                    inferred_basis.append(api_basis or "w/w")
                    inferred_ev.append("from_api_fields")
                    qc_flag.append(api_is_5pct(v))
                    continue
        except Exception:
            pass

        v, u, b, ev = infer_api_from_components(r.get("components_json", ""))
        inferred_vals.append(v)
        inferred_units.append(u)
        inferred_basis.append(b)
        inferred_ev.append(ev)
        qc_flag.append(api_is_5pct(v))

    df["api_conc_inferred_pct"] = inferred_vals
    df["api_conc_inferred_unit"] = inferred_units
    df["api_conc_inferred_basis"] = inferred_basis
    df["api_conc_inferred_evidence"] = inferred_ev
    df["api_5pct_confirmed_v2"] = qc_flag

    df.to_csv(OUT_QC, index=False, encoding="utf-8-sig")

    df_pass = df[df["api_5pct_confirmed_v2"] == True].copy()
    df_manual = df[df["api_5pct_confirmed_v2"] == False].copy()

    df_pass.to_csv(OUT_PASS, index=False, encoding="utf-8-sig")
    df_manual.to_csv(OUT_MANUAL, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Saved:", OUT_QC)
    print("Saved pass5pct:", OUT_PASS, "rows:", len(df_pass))
    print("Saved need_manual_5pct:", OUT_MANUAL, "rows:", len(df_manual))
    print("\napi_5pct_confirmed_v2 counts:\n", df["api_5pct_confirmed_v2"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
