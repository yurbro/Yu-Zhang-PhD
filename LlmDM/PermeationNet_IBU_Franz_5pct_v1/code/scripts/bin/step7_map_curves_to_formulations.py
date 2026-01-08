import json, re, ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

TRIAGE = Path("GenAI/outputs/figure_triage_v1.csv")
ENDPTS = Path("GenAI/outputs/figure_digitized_endpoints.csv")
FORM_FLAT = Path("GenAI/outputs/figure_formulations_v1_flat.csv")

OUT_MAP = Path("GenAI/outputs/figure_curve_formulation_map.csv")

def safe_load(x: str):
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

def norm_label(s: str) -> str:
    s = (s or "").lower()
    # remove common words that add noise
    s = re.sub(r"\b(formulation|formulations|vehicle|vehicles|gel|cream|solution|ointment|batch)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def extract_id_tokens(s: str) -> List[str]:
    s2 = (s or "").upper()
    tokens = []
    # F1, F2, F10
    tokens += re.findall(r"\bF\d+\b", s2)
    # A, B, C when used as formulation label (keep single letters cautiously)
    tokens += re.findall(r"\b[A-Z]\b", s2)
    # Gel-1, Gel 2
    tokens += [t.replace(" ", "") for t in re.findall(r"\bGEL[-\s]?\d+\b", s2)]
    # Formulation A / B
    tokens += [t.replace(" ", "") for t in re.findall(r"\bFORMULATION[-\s]?[A-Z0-9]+\b", s2)]
    # remove duplicates while keeping order
    out = []
    for t in tokens:
        if t not in out:
            out.append(t)
    return out

def color_match(curve_color: str, color_hint: str) -> bool:
    c = (curve_color or "").lower()
    h = (color_hint or "").lower()
    if not c or not h:
        return False
    # allow partial like "dark red"
    return c in h or h in c

def map_curve_label(legend: list, curve_color: str, curve_id: str) -> Tuple[str, str]:
    """
    returns (curve_label, label_source)
    """
    if isinstance(legend, list) and curve_color:
        matches = []
        for item in legend:
            if isinstance(item, dict):
                if color_match(curve_color, item.get("color_hint", "")):
                    lab = str(item.get("label", "")).strip()
                    if lab:
                        matches.append(lab)
        if len(matches) == 1:
            return matches[0], "legend_color_unique"
        if len(matches) > 1:
            # ambiguous, but keep first
            return matches[0], "legend_color_ambiguous"
    return curve_id, "curve_id_fallback"

def map_to_formulation(curve_label: str, formulation_labels: List[str]) -> Tuple[Optional[str], str]:
    """
    returns (mapped_formulation_label, status)
    """
    if not formulation_labels:
        return None, "no_formulations"

    # 1) exact normalized match
    nl = norm_label(curve_label)
    norm_map = {lab: norm_label(lab) for lab in formulation_labels}
    for lab, n in norm_map.items():
        if nl and nl == n and n:
            return lab, "exact_norm_match"

    # 2) token match (F1 etc.)
    c_tokens = extract_id_tokens(curve_label)
    if c_tokens:
        for lab in formulation_labels:
            l_tokens = extract_id_tokens(lab)
            # any shared token is a good sign
            if any(t in l_tokens for t in c_tokens):
                return lab, "token_match"

    # 3) substring match (normalized)
    for lab, n in norm_map.items():
        if nl and n and (nl in n or n in nl):
            return lab, "substring_norm_match"

    return None, "unmapped"

def main():
    df_tri = pd.read_csv(TRIAGE)
    df_end = pd.read_csv(ENDPTS)
    df_fm = pd.read_csv(FORM_FLAT)

    df_end = df_end[df_end["status"] == "ok"].copy()
    # only amount (you已经决定只保留 amount)
    if "y_kind" in df_end.columns:
        df_end = df_end[df_end["y_kind"] != "percent"].copy()

    tri_map = {str(r["doi"]).strip().lower(): r for _, r in df_tri.iterrows()}

    out_rows = []
    for doi, sub_end in df_end.groupby(df_end["doi"].astype(str).str.strip().str.lower()):
        tri = tri_map.get(doi, None)
        legend = safe_load(str(tri.get("legend", ""))) if tri is not None else None

        fm_sub = df_fm[df_fm["doi"].astype(str).str.strip().str.lower() == doi].copy()
        form_labels = fm_sub["formulation_label"].dropna().astype(str).tolist()

        for _, er in sub_end.iterrows():
            curve_id = str(er.get("curve_id", "") or "").strip()
            curve_color = str(er.get("curve_color", "") or "").strip()

            curve_label, label_src = map_curve_label(legend, curve_color, curve_id)
            mapped, status = map_to_formulation(curve_label, form_labels)

            out_rows.append({
                "doi": doi,
                "curve_id": curve_id,
                "curve_color": curve_color,
                "curve_label": curve_label,
                "curve_label_source": label_src,
                "mapped_formulation_label": mapped,
                "mapping_status": status,
                "candidate_formulations": "|".join(form_labels[:30])  # for debugging
            })

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_MAP, index=False, encoding="utf-8-sig")
    print("Done. Saved:", OUT_MAP)
    print("\nmapping_status counts:\n", df_out["mapping_status"].value_counts(dropna=False))

if __name__ == "__main__":
    
    main()
