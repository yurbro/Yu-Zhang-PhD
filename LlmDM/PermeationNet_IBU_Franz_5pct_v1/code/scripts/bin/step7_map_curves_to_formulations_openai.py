import json
import ast
import re
import time
import random
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

TRIAGE = Path("GenAI/outputs/figure_triage_v1.csv")
ENDPTS = Path("GenAI/outputs/figure_digitized_endpoints.csv")
FORM_FLAT = Path("GenAI/outputs/figure_formulations_v1_flat.csv")

OUT_MAP = Path("GenAI/outputs/figure_curve_formulation_map.csv")

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 90
SLEEP_BETWEEN = 0.15
MAX_RETRIES = 6

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You align digitized figure curves to formulation labels from a formulation table. "
    "Use only the provided legend entries, curve colors, and formulation compositions. "
    "Do not guess: if mapping is ambiguous or unsupported, output null with a short reason."
)

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

def backoff(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def summarize_components(components_json: str, max_items: int = 8) -> str:
    comps = safe_load(components_json)
    if not isinstance(comps, list):
        return ""
    names = []
    for c in comps:
        if isinstance(c, dict):
            n = str(c.get("name_raw","")).strip()
            if n:
                names.append(n)
    # keep unique preserving order
    seen = set()
    uniq = []
    for n in names:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    if not uniq:
        return ""
    return "; ".join(uniq[:max_items])

class MapItem(BaseModel):
    curve_id: str
    curve_color: str = ""
    assigned_formulation_label: Optional[str] = None
    rationale: str = ""  # short, evidence-based
    confidence: float = Field(..., ge=0.0, le=1.0)

class MapResult(BaseModel):
    mappings: List[MapItem] = Field(default_factory=list)
    notes: str = ""

def call_llm_for_mapping(doi: str,
                         title: str,
                         figure_id: str,
                         subplot: str,
                         legend_entries: List[Dict[str,str]],
                         curves: List[Dict[str,Any]],
                         formulations: List[Dict[str,Any]]) -> Dict[str,Any]:

    # concise text payload
    leg_txt = "\n".join([
        f"- label: {str(e.get('label','')).strip()} | color_hint: {str(e.get('color_hint','')).strip()}"
        for e in legend_entries[:20]
        if str(e.get('label','')).strip() or str(e.get('color_hint','')).strip()
    ]) or "(none)"

    curve_txt = "\n".join([
        f"- curve_id={c['curve_id']}, curve_color={c.get('curve_color','')}, endpoint={c.get('endpoint_value',None)} {c.get('endpoint_unit','')}"
        for c in curves
    ])

    form_txt_lines = []
    for f in formulations[:40]:
        lab = f.get("formulation_label","")
        api = f.get("api","")
        comps = f.get("components_summary","")
        form_txt_lines.append(f"- {lab} | {api} | comps: {comps}")
    form_txt = "\n".join(form_txt_lines) or "(none)"

    prompt = f"""
Task: Map each digitized curve to a formulation_label from the formulation composition table.

Paper:
DOI: {doi}
TITLE: {title}
Figure: {figure_id} {subplot}

Legend entries (if any):
{leg_txt}

Digitized curves (each must be mapped or set null):
{curve_txt}

Formulation labels + composition summary:
{form_txt}

Rules:
- Use legend labels and color hints when possible.
- If legend labels mention formulation IDs (e.g., F1, ME2, A/B), map accordingly.
- If legend uses descriptive names (e.g., "Microemulsion", "Hydrogel"), use component keywords to map.
- If multiple formulations could match, set assigned_formulation_label=null and explain ambiguity briefly.
Return structured output only.
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": prompt}],
        text_format=MapResult
    )
    return resp.output_parsed.model_dump()

def main():
    df_tri = pd.read_csv(TRIAGE)
    df_end = pd.read_csv(ENDPTS)
    df_fm = pd.read_csv(FORM_FLAT)

    # only OK endpoints, only amount
    df_end = df_end[df_end["status"] == "ok"].copy()
    if "y_kind" in df_end.columns:
        df_end = df_end[df_end["y_kind"] != "percent"].copy()

    tri_map = {str(r["doi"]).strip().lower(): r for _, r in df_tri.iterrows()}

    out_rows = []
    dois = sorted(df_end["doi"].astype(str).str.strip().str.lower().unique().tolist())
    print(f"[LLM-MAP] DOIs to map: {len(dois)}")

    for doi in dois:
        tri = tri_map.get(doi, None)
        if tri is None:
            continue

        title = str(tri.get("title","")).strip()
        figure_id = str(tri.get("figure_id","") or "")
        subplot = str(tri.get("subplot","") or "")

        legend = safe_load(str(tri.get("legend","")))
        legend_entries = legend if isinstance(legend, list) else []

        sub_end = df_end[df_end["doi"].astype(str).str.strip().str.lower() == doi].copy()
        curves = []
        for _, er in sub_end.iterrows():
            curves.append({
                "curve_id": str(er.get("curve_id","")).strip(),
                "curve_color": str(er.get("curve_color","") or "").strip(),
                "endpoint_value": er.get("endpoint_value", None),
                "endpoint_unit": str(er.get("endpoint_unit","") or "").strip(),
            })

        sub_fm = df_fm[df_fm["doi"].astype(str).str.strip().str.lower() == doi].copy()
        formulations = []
        for _, fr in sub_fm.iterrows():
            formulations.append({
                "formulation_label": str(fr.get("formulation_label","")).strip(),
                "api": f"{fr.get('api_name','ibuprofen')} {fr.get('api_concentration_value',None)} {fr.get('api_concentration_unit','')}".strip(),
                "components_summary": summarize_components(str(fr.get("components_json","")))
            })

        attempt = 0
        while True:
            try:
                out = call_llm_for_mapping(
                    doi=doi,
                    title=title,
                    figure_id=figure_id,
                    subplot=subplot,
                    legend_entries=legend_entries,
                    curves=curves,
                    formulations=formulations
                )
                # convert to rows
                mapped = {m["curve_id"]: m for m in out.get("mappings", [])}
                for c in curves:
                    m = mapped.get(c["curve_id"], None)
                    out_rows.append({
                        "doi": doi,
                        "curve_id": c["curve_id"],
                        "curve_color": c.get("curve_color",""),
                        "curve_label": "",  # leave empty; we trust assigned_formulation_label
                        "curve_label_source": "llm_alignment",
                        "mapped_formulation_label": m.get("assigned_formulation_label") if m else None,
                        "mapping_status": "llm_mapped" if (m and m.get("assigned_formulation_label")) else "unmapped",
                        "mapping_rationale": m.get("rationale","") if m else "",
                        "mapping_confidence": m.get("confidence", 0.0) if m else 0.0,
                    })
                break
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    # all curves unmapped for this doi
                    for c in curves:
                        out_rows.append({
                            "doi": doi,
                            "curve_id": c["curve_id"],
                            "curve_color": c.get("curve_color",""),
                            "curve_label": "",
                            "curve_label_source": "llm_alignment",
                            "mapped_formulation_label": None,
                            "mapping_status": "unmapped",
                            "mapping_rationale": f"error:{type(e).__name__}",
                            "mapping_confidence": 0.0,
                        })
                    break
                backoff(attempt)
            time.sleep(SLEEP_BETWEEN)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_MAP, index=False, encoding="utf-8-sig")
    print("Done. Saved:", OUT_MAP)
    if "mapping_status" in df_out.columns:
        print("\nmapping_status counts:\n", df_out["mapping_status"].value_counts(dropna=False))
    if "mapped_formulation_label" in df_out.columns:
        print("\nmapped_formulation_label non-null:", int(df_out["mapped_formulation_label"].notna().sum()), "/", len(df_out))

if __name__ == "__main__":
    main()
