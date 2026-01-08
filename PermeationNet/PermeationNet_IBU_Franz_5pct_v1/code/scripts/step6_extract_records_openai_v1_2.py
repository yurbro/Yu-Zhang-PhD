import re
import json
import time
import random
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
from typing import Annotated
from pydantic import BaseModel, Field

# ==============================
# Config
# ==============================
QUEUE_PATH = "GenAI/outputs/extraction_queue_step6_text.csv"  # from Step5 v1.3
OUT_DIR = Path("GenAI/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "step6_raw_records_v1_2.jsonl"
OUT_FLAT = OUT_DIR / "step6_records_flat_v1_2.csv"

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 90

MAX_PAGES_TOTAL = 18
ALWAYS_INCLUDE_FIRST_PAGES = 2
ANCHOR_NEIGHBOR = 2
MAX_CHARS_TOTAL = 32000

CHECKPOINT_EVERY = 5
SLEEP_BETWEEN = 0.2

# General scoring keywords (no specific excipient list)
FORM_KW = [
    "formulation", "composition", "ingredients", "vehicle", "base",
    "gel", "cream", "ointment", "emulsion", "microemulsion", "hydrogel",
    "% w/w", "w/w", "mg/g", "table"
]
END_KW = [
    "cumulative", "amount permeated", "amount released", "permeation", "release",
    "flux", "jss", "steady-state",
    "µg/cm", "ug/cm", "mg/cm", "ng/cm",
    "table", "results", "after", "hour", "hours", "time", "sampling"
]
CELL_KW = ["franz", "diffusion cell", "permeation cell", "vertical diffusion", "donor", "receptor"]
AREA_KW = ["diffusion area", "effective area", "exposed area", "cm2", "cm^2", "cm²"]

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You are a careful scientific data extraction assistant. "
    "Use ONLY provided PDF text pages with PAGE markers. Do not guess. "
    "Prefer numeric tables. If endpoint appears only in figures without numeric values, set it null and explain."
)

# ==============================
# Structured Output
# ==============================
class EvidenceItem(BaseModel):
    field: str
    locator: str
    snippet: str

class Ingredient(BaseModel):
    name: str
    concentration_raw: str = ""

class Endpoint(BaseModel):
    endpoint_type: Literal["Q_final", "flux", "Jss"]
    value: Optional[float] = None
    unit: str = ""
    time_value: Optional[float] = None
    time_unit: str = ""  # h/min etc.

class ExtractedRecord(BaseModel):
    study_type: Literal["IVPT", "IVRT", "both", "uncertain"]
    cell_type: Literal["Franz", "diffusion_cell_unspecified", "other", "uncertain"]
    dosage_form: str = ""
    barrier_category: Literal["skin", "synthetic_membrane", "both", "uncertain"]
    barrier_name: str = ""

    api_name: Literal["ibuprofen"] = "ibuprofen"
    api_conc_raw: str = ""

    diffusion_area_cm2: Optional[float] = None

    endpoint_main: Endpoint
    endpoint_optional: List[Endpoint] = []

    ingredients: List[Ingredient] = []
    evidence: List[EvidenceItem] = []

    needs_supp: Literal["yes", "no", "uncertain"] = "uncertain"
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""

class PaperExtraction(BaseModel):
    doi: str = ""
    title: str = ""
    records: Annotated[List[ExtractedRecord], Field(default_factory=list, max_length=5)] = Field(default_factory=list)
    extraction_notes: str = ""
    
# ==============================
# Helpers
# ==============================
def _score(text: str, kws: List[str]) -> int:
    low = f" {text.lower()} "
    s = 0
    for kw in kws:
        if kw in low:
            s += 2 if kw in ["ibuprofen", "table", "% w/w", "mg/g", "cumulative", "diffusion area", "franz"] else 1
    return s

def parse_anchor_pages(where_text: str) -> List[int]:
    if not where_text or str(where_text).strip() == "":
        return []
    t = str(where_text)
    nums = set()
    for m in re.finditer(r"\bpage\s*(\d+)\b", t, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"\bp\.?\s*(\d+)\b", t, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))
    pages = []
    for n in sorted(nums):
        if n > 0:
            pages.append(n - 1)
    return pages

def table_hint(where_endpoint: str) -> Optional[str]:
    # try capture "Table 2" etc.
    if not where_endpoint:
        return None
    m = re.search(r"(table\s*\d+)", str(where_endpoint), flags=re.IGNORECASE)
    return m.group(1) if m else None

def extract_pages(pdf_path: str, anchors: List[int], table_tag: Optional[str]) -> str:
    doc = fitz.open(pdf_path)
    n = doc.page_count

    texts = []
    scores = []
    has_table_tag = []
    for i in range(n):
        txt = doc.load_page(i).get_text("text") or ""
        texts.append(txt)
        scores.append(_score(txt, FORM_KW) + _score(txt, END_KW) + _score(txt, CELL_KW) + _score(txt, AREA_KW))
        has_table_tag.append((table_tag.lower() in txt.lower()) if table_tag else False)

    selected = set()

    # always include first pages
    for i in range(min(ALWAYS_INCLUDE_FIRST_PAGES, n)):
        selected.add(i)

    # anchors +/- neighbor
    for a in anchors:
        for j in range(a - ANCHOR_NEIGHBOR, a + ANCHOR_NEIGHBOR + 1):
            if 0 <= j < n:
                selected.add(j)

    # if table tag exists, force include its pages +/- 2
    if table_tag:
        for i in range(n):
            if has_table_tag[i]:
                for j in range(i - 2, i + 3):
                    if 0 <= j < n:
                        selected.add(j)

    # fill with top scored pages
    ranked = sorted([(scores[i], i) for i in range(n) if scores[i] > 0], reverse=True)
    for _, i in ranked:
        selected.add(i)
        if len(selected) >= MAX_PAGES_TOTAL:
            break

    if not selected:
        selected = set(range(min(3, n)))

    selected = sorted(selected)
    parts = []
    for i in selected:
        parts.append(f"\n--- PAGE {i+1} (score={scores[i]}) ---\n{texts[i]}")

    doc.close()
    merged = " ".join("\n".join(parts).split())
    return merged[:MAX_CHARS_TOTAL]

def backoff(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def extract_one(doi: str, title: str, pdf_path: str, anchors: List[int], table_tag: Optional[str]) -> Dict[str, Any]:
    pages_text = extract_pages(pdf_path, anchors, table_tag)

    user = f"""
TASK: Extract structured data for ibuprofen diffusion-cell IVPT/IVRT (text-route).

You are given selected PDF pages with PAGE markers. The selection includes anchor pages around the endpoint location and table context.
Extract up to 5 records only if multiple distinct ibuprofen formulations/conditions are explicitly listed.

Extract:
- cell_type: Franz if explicitly stated; else diffusion_cell_unspecified
- study_type IVPT/IVRT, barrier category+name
- formulation: list ingredients (as stated) + ibuprofen concentration (raw)
- diffusion_area_cm2 if stated (effective/exposed/diffusion area)
- endpoint_main = Q_final@t_last: numeric value + unit + time value+unit (time may appear in table header)
- optional flux/Jss if stated
- evidence: for ibuprofen conc, diffusion area, endpoint value, endpoint time, Franz/diffusion cell.

Rules:
- Prefer numeric tables.
- If endpoint appears only in figures without numeric values, set endpoint_main.value=null and explain.
- If you suspect endpoint/formulation is in Supplement, set needs_supp=yes.

Return structured output ONLY.

META:
DOI: {doi}
TITLE: {title}
ANCHORS(0-index): {anchors}
TABLE_HINT: {table_tag}

PAGES:
{pages_text}
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": user}],
        text_format=PaperExtraction
    )
    out = resp.output_parsed.model_dump()
    out["pdf_path"] = pdf_path
    out["anchors"] = anchors
    out["table_hint"] = table_tag or ""
    return out

def flatten(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    doi = paper.get("doi", "")
    title = paper.get("title", "")
    pdf_path = paper.get("pdf_path", "")
    for i, rec in enumerate(paper.get("records", []), start=1):
        main = rec.get("endpoint_main", {}) or {}
        out.append({
            "doi": doi,
            "title": title,
            "pdf_path": pdf_path,
            "record_idx": i,
            "study_type": rec.get("study_type"),
            "cell_type": rec.get("cell_type"),
            "barrier_category": rec.get("barrier_category"),
            "barrier_name": rec.get("barrier_name"),
            "dosage_form": rec.get("dosage_form"),
            "api_conc_raw": rec.get("api_conc_raw"),
            "diffusion_area_cm2": rec.get("diffusion_area_cm2"),
            "endpoint_type": main.get("endpoint_type"),
            "endpoint_value": main.get("value"),
            "endpoint_unit": main.get("unit"),
            "endpoint_time_value": main.get("time_value"),
            "endpoint_time_unit": main.get("time_unit"),
            "needs_supp": rec.get("needs_supp"),
            "confidence": rec.get("confidence"),
            "notes": rec.get("notes"),
        })
    return out

# ==============================
# Main
# ==============================
def main(limit: Optional[int] = None, no_resume: bool = False):
    df = pd.read_csv(QUEUE_PATH)
    if limit:
        df = df.head(limit)

    done = set()
    if (not no_resume) and OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add((str(obj.get("doi","")).strip().lower(), str(obj.get("pdf_path","")).strip()))
                except Exception:
                    pass

    flat_rows = []
    if (not no_resume) and OUT_FLAT.exists():
        try:
            flat_rows = pd.read_csv(OUT_FLAT).to_dict("records")
        except Exception:
            flat_rows = []

    processed = 0
    for _, row in df.iterrows():
        doi = str(row.get("doi","") or "").strip()
        title = str(row.get("title","") or "").strip()
        pdf_path = str(row.get("pdf_path","") or "").strip()

        anchors = []
        anchors += parse_anchor_pages(str(row.get("where_endpoint","") or ""))
        anchors += parse_anchor_pages(str(row.get("where_franz","") or ""))
        anchors += parse_anchor_pages(str(row.get("where_diffusion_cell","") or ""))
        anchors = sorted(set(anchors))

        ttag = table_hint(str(row.get("where_endpoint","") or ""))

        key = (doi.lower(), pdf_path)
        if key in done:
            continue

        attempt = 0
        while True:
            try:
                paper = extract_one(doi, title, pdf_path, anchors, ttag)
                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(paper, ensure_ascii=False) + "\n")
                flat_rows.extend(flatten(paper))
                done.add(key)
                processed += 1
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    stub = {
                        "doi": doi, "title": title, "pdf_path": pdf_path,
                        "records": [], "extraction_notes": f"error:{type(e).__name__}",
                        "anchors": anchors, "table_hint": ttag or ""
                    }
                    with open(OUT_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(stub, ensure_ascii=False) + "\n")
                    done.add(key)
                    processed += 1
                    break
                backoff(attempt)

        if processed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(flat_rows).to_csv(OUT_FLAT, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: processed {processed}", flush=True)
        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(flat_rows).to_csv(OUT_FLAT, index=False, encoding="utf-8-sig")
    print("Done.")
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_FLAT)
    print("Extracted flat records:", len(flat_rows))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    main(limit=args.limit, no_resume=args.no_resume)
