import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field, conlist

# ==============================
# Paths / Config
# ==============================
QUEUE_PATH = "GenAI/outputs/extraction_queue_step6.csv"
OUT_DIR = Path("GenAI/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "step6_raw_records.jsonl"
OUT_FLAT_CSV = OUT_DIR / "step6_records_flat.csv"
OUT_5PCT_CSV = OUT_DIR / "PermeationNet_IBU_Franz_5pct_v1.csv"

MODEL = "gpt-4o-mini"   # 抽取任务一般够用；如果你追求更稳，可换 gpt-4o
TIMEOUT_SECONDS = 90

# 选页策略：优先抓 formulation 表、endpoint 表所在页
MAX_PAGES_TOTAL = 14
ALWAYS_INCLUDE_FIRST_PAGES = 2
MAX_CHARS_TOTAL = 28000

CHECKPOINT_EVERY = 5
SLEEP_BETWEEN = 0.25

# ============ Keywords ============
FORM_KW = [
    "formulation", "composition", "ingredients", "w/w", "% w/w", "mg/g", "mg g",
    "ibuprofen", "ibu", "ethanol", "propylene glycol", "poloxamer", "p407", "mct",
    "table", "materials", "methods", "vehicle", "gel", "cream", "emulsion"
]
END_KW = [
    "cumulative", "amount permeated", "amount released", "permeation", "release",
    "flux", "jss", "steady-state", "table", "results", "µg/cm", "ug/cm", "mg/cm",
    "hour", " h ", "time", "sampling"
]

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You are a careful scientific data extraction assistant. "
    "You MUST use ONLY the provided PDF text pages (with PAGE markers). "
    "If a value is not explicitly present in the provided pages, set it to null/uncertain and explain in notes. "
    "Prefer tabulated numeric results over narrative text; do not guess from figures unless numeric values are stated."
)

# ==============================
# Structured Output Schema
# ==============================
class EvidenceItem(BaseModel):
    field: str = Field(..., description="What this evidence supports, e.g., 'ibuprofen_concentration', 'Q_final', 't_last', 'barrier'.")
    locator: str = Field(..., description="Where in provided text (use PAGE markers like 'PAGE 8').")
    snippet: str = Field(..., description="Short exact snippet copied from provided text.")

class Ingredient(BaseModel):
    name: str
    concentration_raw: str = ""
    concentration_value: Optional[float] = None
    concentration_unit: str = ""
    concentration_basis: Literal["w/w", "v/v", "w/v", "mg_per_g", "mg_per_mL", "other", "uncertain"] = "uncertain"

class Endpoint(BaseModel):
    endpoint_type: Literal["Q_final", "flux", "Jss"]
    value: Optional[float] = None
    unit: str = ""
    time_value: Optional[float] = None
    time_unit: str = ""  # "h", "min", etc.

class ExtractedRecord(BaseModel):
    study_type: Literal["IVPT", "IVRT", "both", "uncertain"]
    cell_type: Literal["Franz", "diffusion_cell_unspecified", "other", "uncertain"]
    dosage_form: str = ""
    barrier_category: Literal["skin", "synthetic_membrane", "both", "uncertain"]
    barrier_name: str = ""

    api_name: Literal["ibuprofen"] = "ibuprofen"
    api_conc_raw: str = ""
    api_conc_value: Optional[float] = None
    api_conc_unit: str = ""     # e.g. "%", "mg/g"
    api_conc_basis: Literal["w/w", "mg_per_g", "other", "uncertain"] = "uncertain"

    endpoint_main: Endpoint  # must be Q_final ideally
    endpoint_optional: List[Endpoint] = []

    ingredients: List[Ingredient] = []
    evidence: List[EvidenceItem] = []

    needs_supp: Literal["yes", "no", "uncertain"] = "uncertain"
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""

class PaperExtraction(BaseModel):
    doi: str = ""
    title: str = ""
    records: conlist(ExtractedRecord, min_length=0, max_length=5) = []
    extraction_notes: str = ""


# ==============================s
# PDF page selection
# ==============================
def _score(text: str, kws: List[str]) -> int:
    low = f" {text.lower()} "
    s = 0
    for kw in kws:
        if kw in low:
            # 给更关键的词更高权重
            if kw in ["ibuprofen", "% w/w", "mg/g", "composition", "formulation", "table", "flux", "jss", "µg/cm", "ug/cm", "amount permeated"]:
                s += 3
            else:
                s += 1
    return s

def extract_relevant_pages(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    n = doc.page_count

    texts = []
    s_form = []
    s_end = []
    for pno in range(n):
        txt = doc.load_page(pno).get_text("text") or ""
        texts.append(txt)
        s_form.append(_score(txt, FORM_KW))
        s_end.append(_score(txt, END_KW))

    selected = []
    used = set()

    # Always include first pages
    for pno in range(min(ALWAYS_INCLUDE_FIRST_PAGES, n)):
        used.add(pno)
        selected.append((999, pno, texts[pno], "first"))

    # top formulation pages
    form_rank = sorted([(s_form[pno], pno) for pno in range(n) if s_form[pno] > 0], reverse=True)[:6]
    for s, pno in form_rank:
        if pno in used:
            continue
        used.add(pno)
        selected.append((s, pno, texts[pno], "form"))

    # top endpoint pages
    end_rank = sorted([(s_end[pno], pno) for pno in range(n) if s_end[pno] > 0], reverse=True)[:6]
    for s, pno in end_rank:
        if pno in used:
            continue
        used.add(pno)
        selected.append((s, pno, texts[pno], "end"))

    # keep within max pages
    selected = sorted(selected, reverse=True, key=lambda x: x[0])[:MAX_PAGES_TOTAL]
    selected = sorted(selected, key=lambda x: x[1])  # restore page order for readability

    if not selected:
        # fallback
        for pno in range(min(3, n)):
            selected.append((0, pno, texts[pno], "fallback"))

    parts = []
    for s, pno, txt, tag in selected:
        parts.append(f"\n--- PAGE {pno+1} (kw_score={s}, tag={tag}) ---\n{txt}")

    doc.close()
    merged = " ".join("\n".join(parts).split())
    return merged[:MAX_CHARS_TOTAL]


# ==============================
# Normalization helpers (local rules)
# ==============================
def time_to_hours(val: Optional[float], unit: str) -> Optional[float]:
    if val is None:
        return None
    u = (unit or "").strip().lower()
    if u in ["h", "hr", "hrs", "hour", "hours"]:
        return float(val)
    if u in ["min", "mins", "minute", "minutes"]:
        return float(val) / 60.0
    if u in ["s", "sec", "secs", "second", "seconds"]:
        return float(val) / 3600.0
    return None

def endpoint_to_ug_cm2(val: Optional[float], unit: str) -> Optional[float]:
    if val is None:
        return None
    u = (unit or "").strip().lower().replace("μ", "u")
    u = u.replace("²", "2").replace("^2", "2")
    # accept variants: ug/cm2, ug/cm^2, µg/cm²
    if "ug" in u and "cm" in u:
        return float(val)
    if "mg" in u and "cm" in u:
        return float(val) * 1000.0
    if "ng" in u and "cm" in u:
        return float(val) / 1000.0
    return None

def parse_ibuprofen_5pct(api_conc_raw: str, api_conc_value: Optional[float], api_conc_unit: str, api_conc_basis: str) -> str:
    """
    Return: 'yes'/'no'/'uncertain' for whether ibuprofen is 5% w/w (or 50 mg/g equivalent).
    """
    raw = (api_conc_raw or "").lower()
    unit = (api_conc_unit or "").lower()
    basis = (api_conc_basis or "").lower()

    # direct patterns
    if re.search(r"\b5(\.0)?\s*%\s*(w/w)?\b", raw):
        return "yes"
    if re.search(r"\b50(\.0)?\s*mg\s*/\s*g\b", raw) or re.search(r"\b50(\.0)?\s*mg\s*g-1\b", raw):
        return "yes"
    if re.search(r"\b0\.05\s*\(?\s*w/w\s*\)?", raw):
        return "yes"

    # structured value check
    if api_conc_value is not None:
        v = float(api_conc_value)
        if basis == "w/w" and unit in ["%", "percent", "percentage"]:
            return "yes" if abs(v - 5.0) <= 0.2 else "no"
        if basis == "mg_per_g" and ("mg/g" in unit or unit == "mg/g"):
            return "yes" if abs(v - 50.0) <= 2.0 else "no"

    return "uncertain"


# ==============================
# LLM call
# ==============================
def backoff(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def extract_one(doi: str, title: str, pdf_path: str) -> Dict[str, Any]:
    pages_text = extract_relevant_pages(pdf_path)

    user = f"""
TASK: Extract structured data for PermeationNet (ibuprofen, diffusion-cell based IVPT/IVRT).

You will be given partial PDF text with PAGE markers.
Extract up to 5 records ONLY if multiple distinct formulations/conditions with ibuprofen are explicitly listed.
If only one formulation is clearly defined, output a single record.

We care about:
- Cell type: Franz (explicit) or diffusion cell unspecified
- Study type: IVPT / IVRT
- Barrier category and name
- Formulation ingredients with concentrations (as provided)
- Ibuprofen concentration (raw + parsed if possible)
- Main endpoint: cumulative amount permeated/released at the LAST reported time (Q_final@t_last).
  Provide the numeric value, unit, and the corresponding time value+unit.
- Optional endpoints: flux and/or Jss if explicitly reported.
- Provide EVIDENCE items: each key field should include locator (PAGE #) and a short exact snippet.

Rules:
- Prefer numeric tables over narrative.
- If the endpoint is only in figures with no numeric values stated, set endpoint_main.value=null and mark notes.
- If you suspect required info is in Supplement (e.g., 'see Table S1'), set needs_supp=yes.

Return ONLY structured output.

METADATA:
DOI: {doi}
TITLE: {title}

PDF TEXT (keyword-selected pages):
{pages_text}
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        text_format=PaperExtraction
    )
    return resp.output_parsed.model_dump()


# ==============================
# Flatten & Save
# ==============================
def flatten_records(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    doi = paper.get("doi", "")
    title = paper.get("title", "")
    for j, rec in enumerate(paper.get("records", []), start=1):
        main = rec.get("endpoint_main", {}) or {}
        t_h = time_to_hours(main.get("time_value"), main.get("time_unit", ""))
        v_ug = endpoint_to_ug_cm2(main.get("value"), main.get("unit", ""))

        pass5 = parse_ibuprofen_5pct(
            rec.get("api_conc_raw", ""),
            rec.get("api_conc_value"),
            rec.get("api_conc_unit", ""),
            rec.get("api_conc_basis", "")
        )

        out.append({
            "doi": doi,
            "title": title,
            "record_idx": j,
            "study_type": rec.get("study_type"),
            "cell_type": rec.get("cell_type"),
            "dosage_form": rec.get("dosage_form"),
            "barrier_category": rec.get("barrier_category"),
            "barrier_name": rec.get("barrier_name"),

            "api_name": rec.get("api_name"),
            "api_conc_raw": rec.get("api_conc_raw"),
            "api_conc_value": rec.get("api_conc_value"),
            "api_conc_unit": rec.get("api_conc_unit"),
            "api_conc_basis": rec.get("api_conc_basis"),
            "is_ibuprofen_5pct_w_w": pass5,

            "endpoint_type": main.get("endpoint_type"),
            "endpoint_value": main.get("value"),
            "endpoint_unit": main.get("unit"),
            "endpoint_time_value": main.get("time_value"),
            "endpoint_time_unit": main.get("time_unit"),
            "endpoint_time_h": t_h,
            "endpoint_value_ug_cm2": v_ug,

            "needs_supp": rec.get("needs_supp"),
            "confidence": rec.get("confidence"),
            "notes": rec.get("notes"),
        })
    return out

def main(limit: Optional[int] = None, resume: bool = True):
    df = pd.read_csv(QUEUE_PATH)
    if limit:
        df = df.head(limit)

    done = set()
    if resume and OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add((str(obj.get("doi","")).strip().lower(), str(obj.get("pdf_path","")).strip()))
                except Exception:
                    pass

    flat_rows = []
    if resume and OUT_FLAT_CSV.exists():
        try:
            flat_rows = pd.read_csv(OUT_FLAT_CSV).to_dict("records")
        except Exception:
            flat_rows = []

    processed = 0
    for _, row in df.iterrows():
        doi = str(row.get("doi","") or "").strip()
        title = str(row.get("title","") or "").strip()
        pdf_path = str(row.get("pdf_path","") or "").strip()

        key = (doi.lower(), pdf_path)
        if key in done:
            continue

        attempt = 0
        while True:
            try:
                paper = extract_one(doi, title, pdf_path)
                paper["pdf_path"] = pdf_path  # keep trace
                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(paper, ensure_ascii=False) + "\n")

                flat_rows.extend(flatten_records(paper))
                done.add(key)
                processed += 1
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    # write a stub so we don't loop forever
                    stub = {
                        "doi": doi,
                        "title": title,
                        "pdf_path": pdf_path,
                        "records": [],
                        "extraction_notes": f"error:{type(e).__name__}"
                    }
                    with open(OUT_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(stub, ensure_ascii=False) + "\n")
                    done.add(key)
                    processed += 1
                    break
                backoff(attempt)

        if processed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(flat_rows).to_csv(OUT_FLAT_CSV, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: processed {processed}", flush=True)

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(flat_rows).to_csv(OUT_FLAT_CSV, index=False, encoding="utf-8-sig")

    # Apply hard filters for v1: Franz + ibuprofen 5% w/w + Q_final@t_last with time
    df_flat = pd.DataFrame(flat_rows)
    if len(df_flat) > 0:
        v1 = df_flat[
            (df_flat["cell_type"] == "Franz") &
            (df_flat["is_ibuprofen_5pct_w_w"] == "yes") &
            (df_flat["endpoint_type"] == "Q_final") &
            (df_flat["endpoint_value"].notna()) &
            (df_flat["endpoint_time_h"].notna())
        ].copy()
        v1.to_csv(OUT_5PCT_CSV, index=False, encoding="utf-8-sig")

        print("Done.")
        print("All extracted records:", len(df_flat))
        print("V1 (Franz + 5% w/w + Q_final@t_last):", len(v1))
        print("Saved:", OUT_JSONL)
        print("Saved:", OUT_FLAT_CSV)
        print("Saved:", OUT_5PCT_CSV)
    else:
        print("Done, but no records extracted.")
        print("Saved:", OUT_JSONL)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    main(limit=args.limit, resume=(not args.no_resume))
