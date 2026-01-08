import json, ast, time, random
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Tuple

import pandas as pd
import numpy as np
import cv2
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
    "You align digitized curves to formulation labels using the provided zoomed plot image (legend/labels) "
    "and the formulation table labels. Only choose from the provided formulation_label list. "
    "If unclear or ambiguous, return null with a short reason. Do not guess."
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

def parse_bbox(v) -> Optional[List[float]]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None

def crop_zoom(image_path: str, bbox: List[float], margin: float = 0.10) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [float(x) for x in bbox]
    # expand bbox
    dx = (x1 - x0) * margin
    dy = (y1 - y0) * margin
    x0 = max(0.0, x0 - dx); x1 = min(1.0, x1 + dx)
    y0 = max(0.0, y0 - dy); y1 = min(1.0, y1 + dy)
    X0 = int(round(x0 * w)); X1 = int(round(x1 * w))
    Y0 = int(round(y0 * h)); Y1 = int(round(y1 * h))
    X0 = max(0, min(w-2, X0)); X1 = max(X0+1, min(w-1, X1))
    Y0 = max(0, min(h-2, Y0)); Y1 = max(Y0+1, min(h-1, Y1))
    crop = img[Y0:Y1, X0:X1].copy()
    # upsample for readability
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return crop

def img_to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("imencode failed")
    b64 = base64_encode(buf.tobytes())
    return f"data:image/jpeg;base64,{b64}"

def base64_encode(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode("utf-8")

def summarize_components(components_json: str, max_items: int = 10) -> str:
    comps = safe_load(components_json)
    if not isinstance(comps, list):
        return ""
    names = []
    for c in comps:
        if isinstance(c, dict):
            n = str(c.get("name_raw","")).strip()
            if n:
                names.append(n)
    # unique
    seen=set(); uniq=[]
    for n in names:
        k=n.lower()
        if k in seen: 
            continue
        seen.add(k); uniq.append(n)
    return "; ".join(uniq[:max_items])

class MapItem(BaseModel):
    curve_id: str
    curve_color: str = ""
    assigned_formulation_label: Optional[str] = None
    evidence_from_legend: str = ""   # short phrase you read from legend/plot
    rationale: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)

class MapResult(BaseModel):
    mappings: List[MapItem] = Field(default_factory=list)
    notes: str = ""

def call_llm(doi: str, title: str, allowed_labels: List[str], curves: List[Dict[str,Any]], plot_data_url: str, formulations: List[Dict[str,Any]]) -> Dict[str,Any]:
    allowed = "\n".join([f"- {x}" for x in allowed_labels]) or "(none)"

    curve_txt = "\n".join([
        f"- curve_id={c['curve_id']}, curve_color={c.get('curve_color','')}, endpoint≈{c.get('endpoint_value',None)} {c.get('endpoint_unit','')}"
        for c in curves
    ])

    form_txt = "\n".join([
        f"- {f['label']} | api={f['api']} | comps: {f['comps']}"
        for f in formulations[:40]
    ]) or "(none)"

    prompt = f"""
You are given a ZOOMED plot image (axes + curves + legend). Your job is to map each curve_id to one formulation_label.

Allowed formulation labels (choose ONLY from this list):
{allowed}

Curves to map:
{curve_txt}

Formulation composition summary:
{form_txt}

Instructions:
- Read legend labels directly from the image. Use curve_color as a hint, but rely on legend text.
- If legend label matches a formulation_label (e.g., F1, ME2, A, B), map directly.
- If legend uses descriptive names, use composition summary to decide.
- If still ambiguous, set assigned_formulation_label=null and explain briefly.
Return structured output only.
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": plot_data_url},
            ]}
        ],
        text_format=MapResult
    )
    return resp.output_parsed.model_dump()

def main():
    df_tri = pd.read_csv(TRIAGE)
    df_end = pd.read_csv(ENDPTS)
    df_fm = pd.read_csv(FORM_FLAT)

    df_end = df_end[df_end["status"] == "ok"].copy()
    if "y_kind" in df_end.columns:
        df_end = df_end[df_end["y_kind"] != "percent"].copy()

    tri_map = {str(r["doi"]).strip().lower(): r for _, r in df_tri.iterrows()}

    out_rows = []
    dois = sorted(df_end["doi"].astype(str).str.strip().str.lower().unique().tolist())
    print(f"[VISION-MAP] DOIs to map: {len(dois)}")

    for doi in dois:
        tri = tri_map.get(doi, None)
        if tri is None:
            continue

        image_path = str(tri.get("selected_image_path","") or "").strip()
        bbox = parse_bbox(tri.get("plot_bbox", None))
        if not image_path or bbox is None:
            # cannot do vision mapping
            sub_end = df_end[df_end["doi"].astype(str).str.strip().str.lower() == doi]
            for _, er in sub_end.iterrows():
                out_rows.append({
                    "doi": doi,
                    "curve_id": str(er.get("curve_id","")).strip(),
                    "curve_color": str(er.get("curve_color","") or "").strip(),
                    "curve_label": "",
                    "curve_label_source": "vision_alignment",
                    "mapped_formulation_label": None,
                    "mapping_status": "unmapped",
                    "mapping_rationale": "missing_image_or_bbox",
                    "mapping_confidence": 0.0,
                })
            continue

        crop = crop_zoom(image_path, bbox, margin=0.12)
        if crop is None:
            continue
        plot_data_url = img_to_data_url(crop)

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
        allowed_labels = sub_fm["formulation_label"].dropna().astype(str).str.strip().tolist()
        formulations = []
        for _, fr in sub_fm.iterrows():
            formulations.append({
                "label": str(fr.get("formulation_label","")).strip(),
                "api": f"{fr.get('api_name','ibuprofen')} {fr.get('api_concentration_value',None)} {fr.get('api_concentration_unit','')}".strip(),
                "comps": summarize_components(str(fr.get("components_json","")))
            })

        attempt = 0
        while True:
            try:
                out = call_llm(
                    doi=doi,
                    title=str(tri.get("title","") or ""),
                    allowed_labels=allowed_labels,
                    curves=curves,
                    plot_data_url=plot_data_url,
                    formulations=formulations
                )
                mapped = {m["curve_id"]: m for m in out.get("mappings", [])}
                for c in curves:
                    m = mapped.get(c["curve_id"], None)
                    lab = m.get("assigned_formulation_label") if m else None
                    out_rows.append({
                        "doi": doi,
                        "curve_id": c["curve_id"],
                        "curve_color": c.get("curve_color",""),
                        "curve_label": m.get("evidence_from_legend","") if m else "",
                        "curve_label_source": "vision_alignment",
                        "mapped_formulation_label": lab,
                        "mapping_status": "vision_mapped" if lab else "unmapped",
                        "mapping_rationale": m.get("rationale","") if m else "",
                        "mapping_confidence": m.get("confidence", 0.0) if m else 0.0,
                    })
                break
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    for c in curves:
                        out_rows.append({
                            "doi": doi,
                            "curve_id": c["curve_id"],
                            "curve_color": c.get("curve_color",""),
                            "curve_label": "",
                            "curve_label_source": "vision_alignment",
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
    print("\nmapping_status counts:\n", df_out["mapping_status"].value_counts(dropna=False))
    print("\nmapped non-null:", int(df_out["mapped_formulation_label"].notna().sum()), "/", len(df_out))

if __name__ == "__main__":
    main()
