import os
import time
import random
from pathlib import Path
from typing import Literal

import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field

# ==============================
# Config
# ==============================
INV_PATH = "GenAI/outputs/fulltext_inventory.csv"
OUT_PATH = "GenAI/outputs/evidence_index_v1_2.csv"
QUEUE_PATH = "GenAI/outputs/extraction_queue_step6.csv"

MODEL = "gpt-4o-mini"  # evidence indexing 足够用

TIMEOUT_SECONDS = 60
MAX_PAGES_FOR_LLM = 10
ALWAYS_INCLUDE_FIRST_PAGES = 2
MAX_CHARS_TOTAL = 24000

CHECKPOINT_EVERY = 10
SLEEP_BETWEEN = 0.2

KEYWORDS = [
    "ibuprofen", " ibu ", "franz", "diffusion cell", "diffusion cells",
    "permeation", "percutaneous", "in vitro permeation", "in vitro release",
    "ivpt", "ivrt", "vertical diffusion", "receptor", "donor",
    "cumulative", "amount permeated", "amount released", "flux", "jss",
    "membrane", "skin", "human skin", "porcine", "rat skin",
    "strat-m", "nylon", "synthetic membrane",
    "% w/w", "w/w", "mg/g", "mg g", "µg/cm", "ug/cm", "cm2", "cm^2",
    "table", "supplement", "supporting information", "composition", "formulation"
]

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You are an evidence-indexing assistant for building a dataset of ibuprofen in vitro permeation/release "
    "experiments using diffusion cells (Franz included). "
    "You MUST rely ONLY on the provided paper text pages. "
    "Do NOT assume or guess. If not explicitly found, output 'uncertain' or empty strings. "
    "Your job is to LOCATE evidence (where it appears), not to extract full datasets."
)

# ==============================
# Structured Output Schema (v1.2)
# ==============================
class EvidenceIndexV12(BaseModel):
    paper_type: Literal["primary_experiment", "review", "clinical", "other", "uncertain"] = Field(
        ..., description="Classify paper type based on text evidence."
    )
    mentions_ibuprofen: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether ibuprofen is explicitly studied (API)."
    )

    mentions_diffusion_cell: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether diffusion/permeation cell apparatus is mentioned (Franz or other)."
    )
    franz_confirmed: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether the term 'Franz' is explicitly mentioned."
    )
    where_diffusion_cell: str = Field(
        ..., description="Short locator for diffusion cell evidence (e.g., 'Methods p5'). Empty if not found."
    )
    where_franz: str = Field(
        ..., description="Short locator for 'Franz' mention. Empty if not found."
    )

    study_type: Literal["IVPT", "IVRT", "both", "uncertain"] = Field(
        ..., description="IVPT vs IVRT based on barrier and phrasing."
    )
    barrier_category: Literal["skin", "synthetic_membrane", "both", "uncertain"] = Field(
        ..., description="Barrier category."
    )
    barrier_name_raw: str = Field(
        ..., description="Raw barrier name if stated (e.g., human skin, Strat-M). Empty if not found."
    )

    # Concentration: NOT used for Step5 gating now; only evidence capturing for Step6.
    ibuprofen_conc_found: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether ibuprofen concentration statement is found in provided pages."
    )
    ibuprofen_conc_value_raw: str = Field(
        ..., description="Raw concentration expression (e.g., '5% w/w', '50 mg/g', '1.0% w/w'). Empty if not found."
    )
    ibuprofen_conc_basis: Literal["w/w", "mg_per_g", "other", "uncertain"] = Field(
        ..., description="Basis of the concentration if found."
    )
    ibuprofen_conc_snippet: str = Field(
        ..., description="SHORT exact snippet containing ibuprofen/IBU and the concentration. Empty if not found."
    )

    endpoint_qfinal_found: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether numeric cumulative amount permeated/released at a timepoint/final time exists (table/text)."
    )
    endpoint_time_found: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether the time associated with endpoint is present."
    )
    where_endpoint: str = Field(
        ..., description="Locator for endpoint (e.g., 'Table 2 p8', 'Results p9', 'Supp Table S1'). Empty if not found."
    )
    figure_only: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether endpoints appear only in figures without numeric tables."
    )

    endpoint_likely_in_supp: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether key endpoints likely in supplementary info."
    )
    formulation_table_missing: Literal["yes", "no", "uncertain"] = Field(
        ..., description="Whether formulation composition table seems missing from provided pages."
    )

    notes: str = Field(
        ..., description="Critical caveats (e.g., 'systematic review', 'ibuprofen not studied', 'no numeric table')."
    )

# ==============================
# Helpers
# ==============================
def norm_ws(s: str) -> str:
    return " ".join((s or "").split())

def score_page(text: str) -> int:
    low = f" {text.lower()} "
    score = 0
    for kw in KEYWORDS:
        if kw in low:
            if kw in ["ibuprofen", "franz", "diffusion cell", "ivpt", "ivrt", "table", "% w/w", "mg/g", "composition", "formulation"]:
                score += 3
            else:
                score += 1
    return score

def extract_relevant_pages(pdf_path: str,
                           max_pages: int = MAX_PAGES_FOR_LLM,
                           always_first: int = ALWAYS_INCLUDE_FIRST_PAGES) -> str:
    """
    Keyword-driven page selection: scan all pages, score, select:
    - first N pages
    - top scored pages
    """
    doc = fitz.open(pdf_path)
    n = doc.page_count

    page_texts = []
    scores = []
    for pno in range(n):
        txt = doc.load_page(pno).get_text("text") or ""
        page_texts.append(txt)
        scores.append(score_page(txt))

    # Select pages
    selected = []
    used = set()

    for pno in range(min(always_first, n)):
        used.add(pno)
        selected.append((999, pno, page_texts[pno]))

    scored_pages = [(scores[pno], pno, page_texts[pno]) for pno in range(n) if scores[pno] > 0]
    scored_pages.sort(reverse=True, key=lambda x: x[0])

    for s, pno, txt in scored_pages:
        if pno in used:
            continue
        selected.append((s, pno, txt))
        used.add(pno)
        if len(selected) >= max_pages:
            break

    if not selected:
        for pno in range(min(3, n)):
            selected.append((0, pno, page_texts[pno]))

    parts = []
    for s, pno, txt in selected:
        parts.append(f"\n--- PAGE {pno+1} (kw_score={s}) ---\n{txt}")

    doc.close()
    merged = norm_ws("\n".join(parts))
    return merged[:MAX_CHARS_TOTAL]

def backoff(attempt: int) -> None:
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def post_validate(res: dict) -> dict:
    """
    Keep concentration evidence honest:
    - snippet must mention ibuprofen/IBU, otherwise clear it.
    """
    snip = (res.get("ibuprofen_conc_snippet") or "").strip()
    low = snip.lower()
    if snip and ("ibuprofen" not in low) and ("ibu" not in low):
        res["ibuprofen_conc_found"] = "uncertain"
        res["ibuprofen_conc_value_raw"] = ""
        res["ibuprofen_conc_basis"] = "uncertain"
        res["ibuprofen_conc_snippet"] = ""
        res["notes"] = (res.get("notes") or "") + " | conc_snippet_missing_ibuprofen"
    return res

def compute_ready_for_step6(res: dict) -> str:
    """
    Step5 gating for Step6 queue:
    - DO NOT require 5% here
    - Focus on: primary experiment + ibuprofen + diffusion cell + endpoint+time + not figure-only
    """
    if res.get("paper_type") != "primary_experiment":
        return "no"
    if res.get("mentions_ibuprofen") != "yes":
        return "no"
    if not (res.get("franz_confirmed") == "yes" or res.get("mentions_diffusion_cell") == "yes"):
        return "no"
    if res.get("endpoint_qfinal_found") != "yes":
        return "no"
    if res.get("endpoint_time_found") != "yes":
        return "no"
    if res.get("figure_only") == "yes":
        return "no"
    return "yes"

def index_one(doi: str, title: str, pdf_path: str) -> dict:
    pages_text = extract_relevant_pages(pdf_path)

    user = f"""
TASK: Evidence indexing (NOT full extraction).

Decide and LOCATE evidence for:
1) paper_type: primary experiment vs review/clinical
2) mentions_ibuprofen: ibuprofen explicitly studied?
3) mentions_diffusion_cell and franz_confirmed, plus where evidence appears
4) IVPT/IVRT and barrier info
5) ibuprofen concentration statement if present (any value). Provide raw expression and snippet.
   IMPORTANT: snippet must explicitly contain 'ibuprofen' or 'IBU' together with the concentration.
6) numeric endpoint existence: cumulative amount permeated/released at final time (or a clear timepoint), AND whether time is provided
7) whether endpoint appears only in figures (figure_only)
8) whether key info likely in Supplementary Info / formulation table missing

Return structured output only.

METADATA:
DOI: {doi}
TITLE: {title}

PAGES (keyword-selected, partial):
{pages_text}
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        text_format=EvidenceIndexV12
    )
    res = resp.output_parsed.model_dump()
    res = post_validate(res)
    return res

# ==============================
# Main
# ==============================
def main(limit: int | None = None, resume: bool = False):
    df = pd.read_csv(INV_PATH)
    df["pdf_status"] = df["pdf_status"].fillna("").astype(str).str.lower()
    df_ok = df[df["pdf_status"] == "ok"].copy()
    if limit:
        df_ok = df_ok.head(limit)

    out_file = Path(OUT_PATH)
    queue_file = Path(QUEUE_PATH)

    rows = []
    done = set()

    if resume and out_file.exists():
        old = pd.read_csv(out_file)
        rows = old.to_dict("records")
        for _, r in old.iterrows():
            done.add((str(r.get("doi","")).lower().strip(), str(r.get("title","")).strip()))
    else:
        if out_file.exists():
            out_file.unlink()

    total = len(df_ok)
    for idx, row in enumerate(df_ok.itertuples(index=False), start=1):
        doi = str(getattr(row, "doi", "") or "").strip()
        title = str(getattr(row, "title", "") or "").strip()
        pdf_path = str(getattr(row, "pdf_path", "") or "").strip()

        key = (doi.lower().strip(), title.strip())
        if key in done:
            continue

        attempt = 0
        while True:
            try:
                res = index_one(doi, title, pdf_path)
                ready = compute_ready_for_step6(res)
                rec = {"doi": doi, "title": title, "pdf_path": pdf_path, "ready_for_step6": ready, **res}
                rows.append(rec)
                done.add(key)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    rows.append({
                        "doi": doi,
                        "title": title,
                        "pdf_path": pdf_path,
                        "ready_for_step6": "no",
                        "paper_type": "uncertain",
                        "mentions_ibuprofen": "uncertain",
                        "mentions_diffusion_cell": "uncertain",
                        "franz_confirmed": "uncertain",
                        "where_diffusion_cell": "",
                        "where_franz": "",
                        "study_type": "uncertain",
                        "barrier_category": "uncertain",
                        "barrier_name_raw": "",
                        "ibuprofen_conc_found": "uncertain",
                        "ibuprofen_conc_value_raw": "",
                        "ibuprofen_conc_basis": "uncertain",
                        "ibuprofen_conc_snippet": "",
                        "endpoint_qfinal_found": "uncertain",
                        "endpoint_time_found": "uncertain",
                        "where_endpoint": "",
                        "figure_only": "uncertain",
                        "endpoint_likely_in_supp": "uncertain",
                        "formulation_table_missing": "uncertain",
                        "notes": f"error:{type(e).__name__}"
                    })
                    done.add(key)
                    break
                backoff(attempt)

        if len(rows) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: {len(rows)}/{total}", flush=True)

        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_file, index=False, encoding="utf-8-sig")

    # Build Step6 queue
    q = df_out[df_out["ready_for_step6"] == "yes"].copy()
    # simple priority scoring
    q["priority"] = 0
    q.loc[q["where_endpoint"].fillna("").str.contains("table", case=False), "priority"] += 2
    q.loc[q["franz_confirmed"] == "yes", "priority"] += 1
    q.loc[q["ibuprofen_conc_found"] == "yes", "priority"] += 1
    q = q.sort_values(["priority"], ascending=False)

    q.to_csv(queue_file, index=False, encoding="utf-8-sig")

    print("Done. Saved:", out_file)
    print("ready_for_step6 counts:")
    print(df_out["ready_for_step6"].value_counts(dropna=False))
    print("paper_type counts:")
    print(df_out["paper_type"].value_counts(dropna=False))
    print("Saved Step6 queue:", queue_file, "rows:", len(q))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    main(limit=args.limit, resume=args.resume)
