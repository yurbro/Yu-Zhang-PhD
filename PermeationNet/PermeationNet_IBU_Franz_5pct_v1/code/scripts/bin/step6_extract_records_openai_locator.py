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

OUT_JSONL = OUT_DIR / "step6_raw_records_locator.jsonl"
OUT_FLAT_CSV = OUT_DIR / "step6_records_flat_locator.csv"

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 90

# Locator-aware selection
MAX_PAGES_TOTAL = 16
ALWAYS_INCLUDE_FIRST_PAGES = 2
ANCHOR_NEIGHBOR = 1      # include anchor page +/- this
MAX_CHARS_TOTAL = 30000

CHECKPOINT_EVERY = 5
SLEEP_BETWEEN = 0.25

FORM_KW = [
    "formulation", "composition", "ingredients", "w/w", "% w/w", "mg/g", "mg g",
    "ibuprofen", "ibu", "ethanol", "propylene glycol", "poloxamer", "p407", "mct",
    "table", "materials", "methods", "vehicle", "gel", "cream", "emulsion"
]
END_KW = [
    "cumulative", "amount permeated", "amount released", "permeation", "release",
    "flux", "jss", "steady-state", "table", "results",
    "µg/cm", "ug/cm", "mg/cm", "ng/cm", "after", "h", "hour", "hours", "time", "sampling"
]
CELL_KW = ["franz", "diffusion cell", "vertical diffusion", "permeation cell", "receptor", "donor"]
AREA_KW = ["diffusion area", "effective area", "cm2", "cm^2", "cm²"]

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You are a careful scientific data extraction assistant. "
    "You MUST use ONLY the provided PDF text pages (with PAGE markers). "
    "If a value is not explicitly present, set it to null/uncertain and explain in notes. "
    "Prefer tabulated numeric results; do not guess from figures unless numeric values are stated."
)

# ==============================
# Structured Output Schema
# ==============================
class EvidenceItem(BaseModel):
    field: str
    locator: str
    snippet: str

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
    time_unit: str = ""

class ExtractedRecord(BaseModel):
    study_type: Literal["IVPT", "IVRT", "both", "uncertain"]
    cell_type: Literal["Franz", "diffusion_cell_unspecified", "other", "uncertain"]
    dosage_form: str = ""
    barrier_category: Literal["skin", "synthetic_membrane", "both", "uncertain"]
    barrier_name: str = ""

    api_name: Literal["ibuprofen"] = "ibuprofen"
    api_conc_raw: str = ""
    api_conc_value: Optional[float] = None
    api_conc_unit: str = ""
    api_conc_basis: Literal["w/w", "mg_per_g", "other", "uncertain"] = "uncertain"

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
    records: conlist(ExtractedRecord, min_length=0, max_length=5) = []
    extraction_notes: str = ""


# ==============================
# Page selection helpers
# ==============================
def _score(text: str, kws: List[str]) -> int:
    low = f" {text.lower()} "
    s = 0
    for kw in kws:
        if kw in low:
            s += 2 if kw in ["ibuprofen", "% w/w", "mg/g", "composition", "formulation", "table",
                             "flux", "jss", "µg/cm", "ug/cm", "amount permeated", "diffusion area"] else 1
    return s

def parse_anchor_pages(where_text: str) -> List[int]:
    """
    Parse page numbers from strings like:
    - 'Table 2 p8'
    - 'Methods p5'
    - 'PAGE 8'
    - 'page 12'
    Return 0-indexed pdf page numbers.
    """
    if not where_text:
        return []
    t = where_text.strip()

    nums = set()

    # PAGE 8 / page 8
    for m in re.finditer(r"\bpage\s*(\d+)\b", t, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))

    # p8 / p. 8
    for m in re.finditer(r"\bp\.?\s*(\d+)\b", t, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))

    # If nothing found, return empty
    pages = []
    for n in sorted(nums):
        if n > 0:
            pages.append(n - 1)
    return pages

def extract_pages(pdf_path: str, anchors: List[int]) -> str:
    doc = fitz.open(pdf_path)
    n = doc.page_count

    texts = []
    scores = []
    for i in range(n):
        txt = doc.load_page(i).get_text("text") or ""
        texts.append(txt)
        scores.append(
            _score(txt, FORM_KW) +
            _score(txt, END_KW) +
            _score(txt, CELL_KW) +
            _score(txt, AREA_KW)
        )

    selected = set()

    # first pages
    for i in range(min(ALWAYS_INCLUDE_FIRST_PAGES, n)):
        selected.add(i)

    # anchors +/- neighbor
    for a in anchors:
        for j in range(a - ANCHOR_NEIGHBOR, a + ANCHOR_NEIGHBOR + 1):
            if 0 <= j < n:
                selected.add(j)

    # top scored pages fill
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

# ==============================
# LLM call
# ==============================
def extract_one(doi: str, title: str, pdf_path: str, anchors: List[int]) -> Dict[str, Any]:
    pages_text = extract_pages(pdf_path, anchors)

    user = f"""
TASK: Extract structured data for PermeationNet (ibuprofen, diffusion-cell based IVPT/IVRT).

You will be given partial PDF text with PAGE markers. The selection includes anchor pages around where endpoints are reported.
Extract up to 5 records ONLY if multiple distinct formulations/conditions with ibuprofen are explicitly listed.
If only one formulation is clearly defined, output a single record.

We care about:
- Cell type: Franz (explicit) or diffusion cell unspecified
- Study type: IVPT / IVRT
- Barrier category and name
- Formulation ingredients with concentrations (as provided)
- Ibuprofen concentration (raw + parsed if possible)
- Main endpoint: cumulative amount permeated/released at the LAST reported time (Q_final@t_last).
  Provide numeric value, unit, and the corresponding time value+unit.
- Optional endpoints: flux and/or Jss if explicitly reported.
- Provide EVIDENCE items (PAGE + snippet) for key fields.

Rules:
- Prefer numeric tables over narrative.
- If endpoint only in figures with no numeric values, set endpoint_main.value=null and explain in notes.
- If you suspect required info is in Supplement, set needs_supp=yes.

Return ONLY structured output.

META:
DOI: {doi}
TITLE: {title}
ANCHOR_PAGES(0-index): {anchors}

PDF TEXT:
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
# Flatten
# ==============================
def flatten_records(paper: Dict[str, Any], pdf_path: str) -> List[Dict[str, Any]]:
    out = []
    doi = paper.get("doi", "")
    title = paper.get("title", "")
    for j, rec in enumerate(paper.get("records", []), start=1):
        main = rec.get("endpoint_main", {}) or {}
        out.append({
            "doi": doi,
            "title": title,
            "pdf_path": pdf_path,
            "record_idx": j,
            "study_type": rec.get("study_type"),
            "cell_type": rec.get("cell_type"),
            "dosage_form": rec.get("dosage_form"),
            "barrier_category": rec.get("barrier_category"),
            "barrier_name": rec.get("barrier_name"),
            "api_conc_raw": rec.get("api_conc_raw"),
            "api_conc_value": rec.get("api_conc_value"),
            "api_conc_unit": rec.get("api_conc_unit"),
            "api_conc_basis": rec.get("api_conc_basis"),
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
    if (not no_resume) and OUT_FLAT_CSV.exists():
        try:
            flat_rows = pd.read_csv(OUT_FLAT_CSV).to_dict("records")
        except Exception:
            flat_rows = []

    processed = 0
    for _, row in df.iterrows():
        doi = str(row.get("doi","") or "").strip()
        title = str(row.get("title","") or "").strip()
        pdf_path = str(row.get("pdf_path","") or "").strip()

        # locator-aware anchors
        where_endpoint = str(row.get("where_endpoint","") or "")
        where_franz = str(row.get("where_franz","") or "")
        where_cell = str(row.get("where_diffusion_cell","") or "")
        anchors = []
        anchors += parse_anchor_pages(where_endpoint)
        anchors += parse_anchor_pages(where_franz)
        anchors += parse_anchor_pages(where_cell)
        anchors = sorted(set([a for a in anchors if isinstance(a, int)]))

        key = (doi.lower(), pdf_path)
        if key in done:
            continue

        attempt = 0
        while True:
            try:
                paper = extract_one(doi, title, pdf_path, anchors)
                paper["pdf_path"] = pdf_path
                paper["anchors"] = anchors

                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(paper, ensure_ascii=False) + "\n")

                flat_rows.extend(flatten_records(paper, pdf_path))
                done.add(key)
                processed += 1
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    stub = {
                        "doi": doi, "title": title, "pdf_path": pdf_path,
                        "records": [], "extraction_notes": f"error:{type(e).__name__}",
                        "anchors": anchors
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
    print("Done.")
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_FLAT_CSV)
    print("Extracted flat records:", len(flat_rows))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    main(limit=args.limit, no_resume=args.no_resume)
