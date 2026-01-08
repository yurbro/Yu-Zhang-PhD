import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

import fitz  # PyMuPDF
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

# ======================
# Config
# ======================
IN_JSONL = "GenAI/outputs/step6_raw_records_v1_2.jsonl"  # using the output from step6_extract_records_openai_locators.py
OUT_VERIFIED = "GenAI/outputs/step6_verified_records_v1_2.csv"  # full verified output
OUT_V1 = "GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_verified_v1_2.csv"  # strict v1 pass

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 90

MAX_PAGES_TOTAL = 12
NEIGHBOR_PAGES = 1
MAX_CHARS_TOTAL = 26000

CHECKPOINT_EVERY = 10
SLEEP_BETWEEN = 0.2

FORM_KW = [
    "formulation", "composition", "ingredients", "ibuprofen", "ibu",
    "% w/w", "w/w", "mg/g", "adhesive", "gel", "cream", "microemulsion",
    "table", "materials", "methods"
]
END_KW = [
    "cumulative", "amount permeated", "amount released", "permeation", "release",
    "flux", "jss", "steady-state",
    "µg/cm", "ug/cm", "mg/cm", "ng/cm",
    "table", "results", "after", "h", "hour", "hours"
]
CELL_KW = ["franz", "diffusion cell", "vertical diffusion", "permeation cell", "receptor", "donor"]
AREA_KW = ["diffusion area", "effective area", "cm2", "cm^2", "cm²"]

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You are a strict verifier. Use ONLY the provided PDF text pages with PAGE markers. "
    "Do not guess. If a claim is not directly supported, mark it 'no' or 'uncertain'. "
    "Your job is to verify/correct: Franz, ibuprofen 5% w/w, and Q_final@t_last with correct units and meaning."
)

class EvidenceItem(BaseModel):
    field: str
    locator: str
    snippet: str

class VerifyResult(BaseModel):
    franz_confirmed: Literal["yes", "no", "uncertain"]
    ibuprofen_5pct_w_w: Literal["yes", "no", "uncertain"]
    api_conc_raw: str = ""
    # Endpoint verification
    endpoint_value: Optional[float] = None
    endpoint_unit: str = ""
    endpoint_time_value: Optional[float] = None
    endpoint_time_unit: str = ""
    endpoint_kind: Literal["amount_per_area", "amount_total", "percent", "unknown"] = "unknown"
    diffusion_area_cm2: Optional[float] = None  # if stated, helps convert total->per_area
    evidence: List[EvidenceItem] = []
    needs_supp: Literal["yes", "no", "uncertain"] = "uncertain"
    notes: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)

# ======================
# Helpers
# ======================
def _score(text: str, kws: List[str]) -> int:
    low = f" {text.lower()} "
    s = 0
    for kw in kws:
        if kw in low:
            s += 2 if kw in ["ibuprofen", "franz", "table", "% w/w", "mg/g", "µg/cm", "ug/cm", "diffusion area"] else 1
    return s

def parse_page_numbers_from_evidence(record: Dict[str, Any]) -> List[int]:
    pages = set()
    for ev in record.get("evidence", []) or []:
        loc = (ev.get("locator") or "")
        m = re.search(r"PAGE\s+(\d+)", loc, flags=re.IGNORECASE)
        if m:
            p = int(m.group(1)) - 1  # to 0-index
            pages.add(p)
    return sorted(pages)

def extract_pages_for_verify(pdf_path: str, anchor_pages: List[int]) -> str:
    # Path robustness (Windows backslashes are fine, but normalize anyway)
    p = Path(pdf_path)
    if not p.exists():
        p = Path(str(pdf_path).replace("\\", "/"))
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(p)
    n = doc.page_count
    texts = []
    scores = []

    for i in range(n):
        txt = doc.load_page(i).get_text("text") or ""
        texts.append(txt)
        s = _score(txt, FORM_KW) + _score(txt, END_KW) + _score(txt, CELL_KW) + _score(txt, AREA_KW)
        scores.append(s)

    selected = set()

    # include anchors +/- neighbors
    for a in anchor_pages:
        for j in range(a - NEIGHBOR_PAGES, a + NEIGHBOR_PAGES + 1):
            if 0 <= j < n:
                selected.add(j)

    # add top scored pages
    ranked = sorted([(scores[i], i) for i in range(n) if scores[i] > 0], reverse=True)[:MAX_PAGES_TOTAL]
    for _, i in ranked:
        selected.add(i)
        if len(selected) >= MAX_PAGES_TOTAL:
            break

    if not selected:
        selected = set(range(min(3, n)))

    selected = sorted(selected)
    parts = []
    for i in selected:
        parts.append(f"\n--- PAGE {i+1} ---\n{texts[i]}")
    doc.close()

    merged = " ".join("\n".join(parts).split())
    return merged[:MAX_CHARS_TOTAL]

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

def amount_to_ug_cm2(val: Optional[float], unit: str, area_cm2: Optional[float], kind: str) -> Optional[float]:
    if val is None:
        return None
    u = (unit or "").strip().lower()
    u = u.replace("μ", "u").replace("µ", "u").replace("²", "2").replace("^2", "2")

    if kind == "percent":
        return None

    # amount_per_area
    if kind == "amount_per_area":
        if "ug" in u and "cm" in u:
            return float(val)
        if "mg" in u and "cm" in u:
            return float(val) * 1000.0
        if "ng" in u and "cm" in u:
            return float(val) / 1000.0
        return None

    # amount_total -> need area
    if kind == "amount_total":
        if area_cm2 is None or area_cm2 <= 0:
            return None
        if "ug" in u:
            return float(val) / float(area_cm2)
        if "mg" in u:
            return float(val) * 1000.0 / float(area_cm2)
        if "ng" in u:
            return float(val) / 1000.0 / float(area_cm2)
        return None

    return None

def backoff(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def verify_one(paper_meta: Dict[str, Any], record: Dict[str, Any], record_idx: int) -> Dict[str, Any]:
    doi = paper_meta.get("doi", "")
    title = paper_meta.get("title", "")
    pdf_path = paper_meta.get("pdf_path", "")

    anchors = parse_page_numbers_from_evidence(record)
    pages_text = extract_pages_for_verify(pdf_path, anchors)

    # what Step6 extracted (for context only)
    extracted_summary = {
        "cell_type": record.get("cell_type"),
        "api_conc_raw": record.get("api_conc_raw"),
        "endpoint_main": record.get("endpoint_main"),
        "barrier_name": record.get("barrier_name"),
        "dosage_form": record.get("dosage_form"),
    }

    user = f"""
Verify the extracted record STRICTLY using the PDF pages.

Goals (must be evidence-supported):
1) Franz_confirmed?
2) Ibuprofen concentration is 5% w/w (or 50 mg/g equivalent)?
3) Q_final@t_last: numeric value + unit + time.
   Also classify endpoint_kind:
   - amount_per_area: e.g., µg/cm2
   - amount_total: e.g., mg (total amount), then try to find diffusion_area_cm2
   - percent: e.g., 63% (NOT acceptable for Q in µg/cm2)
   - unknown

If you cannot find a claim directly, set it to 'uncertain' or null.

Return structured output only.

META:
DOI: {doi}
TITLE: {title}
record_idx: {record_idx}

Step6 extracted (for context, may be wrong):
{json.dumps(extracted_summary, ensure_ascii=False)}

PDF TEXT (selected pages):
{pages_text}
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": user}],
        text_format=VerifyResult
    )
    out = resp.output_parsed.model_dump()
    out["doi"] = doi
    out["title"] = title
    out["record_idx"] = record_idx
    out["pdf_path"] = pdf_path
    return out

# ======================
# Main
# ======================
def main():
    in_path = Path(IN_JSONL)
    if not in_path.exists():
        raise FileNotFoundError(IN_JSONL)

    # load papers
    papers = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))

    # dedupe by pdf_path (avoid accidental duplicates)
    by_pdf = {}
    for p in papers:
        by_pdf.setdefault(p.get("pdf_path",""), p)
    papers = list(by_pdf.values())

    verified_rows = []
    processed = 0

    for p in papers:
        recs = p.get("records", []) or []
        for idx, rec in enumerate(recs, start=1):
            attempt = 0
            while True:
                try:
                    v = verify_one(p, rec, idx)

                    # compute QC flags & normalized endpoint
                    t_h = time_to_hours(v.get("endpoint_time_value"), v.get("endpoint_time_unit",""))
                    q_ugcm2 = amount_to_ug_cm2(
                        v.get("endpoint_value"),
                        v.get("endpoint_unit",""),
                        v.get("diffusion_area_cm2"),
                        v.get("endpoint_kind","unknown")
                    )

                    v["endpoint_time_h"] = t_h
                    v["endpoint_value_ug_cm2"] = q_ugcm2

                    # strict v1 pass
                    v["pass_v1"] = (
                        v.get("franz_confirmed") == "yes"
                        and v.get("ibuprofen_5pct_w_w") == "yes"
                        and v.get("endpoint_value") is not None
                        and t_h is not None
                        and v.get("endpoint_kind") in ["amount_per_area", "amount_total"]
                        and q_ugcm2 is not None
                    )
                    verified_rows.append(v)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= 6:
                        verified_rows.append({
                            "doi": p.get("doi",""),
                            "title": p.get("title",""),
                            "record_idx": idx,
                            "pdf_path": p.get("pdf_path",""),
                            "franz_confirmed": "uncertain",
                            "ibuprofen_5pct_w_w": "uncertain",
                            "api_conc_raw": "",
                            "endpoint_value": None,
                            "endpoint_unit": "",
                            "endpoint_time_value": None,
                            "endpoint_time_unit": "",
                            "endpoint_kind": "unknown",
                            "diffusion_area_cm2": None,
                            "endpoint_time_h": None,
                            "endpoint_value_ug_cm2": None,
                            "needs_supp": "uncertain",
                            "notes": f"error:{type(e).__name__}",
                            "confidence": 0.0,
                            "pass_v1": False,
                            "evidence": []
                        })
                        break
                    backoff(attempt)

            processed += 1
            if processed % CHECKPOINT_EVERY == 0:
                pd.DataFrame(verified_rows).to_csv(OUT_VERIFIED, index=False, encoding="utf-8-sig")
                print(f"Checkpoint: verified {processed}", flush=True)
            time.sleep(SLEEP_BETWEEN)

    dfv = pd.DataFrame(verified_rows)
    dfv.to_csv(OUT_VERIFIED, index=False, encoding="utf-8-sig")

    v1 = dfv[dfv["pass_v1"] == True].copy()
    v1.to_csv(OUT_V1, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Verified records:", len(dfv))
    print("pass_v1:", int(v1.shape[0]))
    print("Saved:", OUT_VERIFIED)
    print("Saved:", OUT_V1)

if __name__ == "__main__":
    main()
