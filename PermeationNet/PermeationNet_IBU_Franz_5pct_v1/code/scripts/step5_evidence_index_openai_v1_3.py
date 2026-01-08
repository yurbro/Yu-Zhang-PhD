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

OUT_INDEX = "GenAI/outputs/evidence_index_v1_3.csv"
OUT_QUEUE_TEXT = "GenAI/outputs/extraction_queue_step6_text.csv"
OUT_QUEUE_FIG = "GenAI/outputs/extraction_queue_step6_figure.csv"
OUT_QUEUE_SUPP = "GenAI/outputs/extraction_queue_step6_supp.csv"

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 60

MAX_PAGES_FOR_LLM = 12
ALWAYS_INCLUDE_FIRST_PAGES = 2
MAX_CHARS_TOTAL = 26000

CHECKPOINT_EVERY = 10
SLEEP_BETWEEN = 0.2

# Keyword list for page scoring (keep general to avoid excipient bias)
KEYWORDS = [
    "ibuprofen", " ibu ",
    "franz", "diffusion cell", "permeation cell", "vertical diffusion",
    "ivpt", "ivrt", "in vitro permeation", "in vitro release",
    "cumulative", "amount permeated", "amount released", "permeation", "release",
    "flux", "jss", "steady-state",
    "table", "figure", "fig.", "supporting information", "supplement",
    "formulation", "composition", "ingredients", "vehicle",
    "% w/w", "w/w", "mg/g", "mg g", "µg/cm", "ug/cm", "cm2", "cm^2",
    "diffusion area", "effective area", "exposed area",
    "skin", "human skin", "porcine", "rat skin", "mouse skin",
    "membrane", "nylon", "strat-m", "synthetic membrane"
]

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You are an evidence-indexing and routing assistant for building a dataset of ibuprofen diffusion-cell IVPT/IVRT studies. "
    "You MUST rely ONLY on the provided PDF text pages. Do NOT guess. "
    "You must (1) decide eligibility gates and (2) label where endpoint/formulation data live (table vs figure vs supplement vs narrative), "
    "each with a short evidence locator and snippet."
)

# ==============================
# Structured Outputs
# ==============================
class EvidenceIndexV13(BaseModel):
    paper_type: Literal["primary_experiment", "review", "clinical", "other", "uncertain"]
    mentions_ibuprofen: Literal["yes", "no", "uncertain"]

    mentions_diffusion_cell: Literal["yes", "no", "uncertain"]
    franz_confirmed: Literal["yes", "no", "uncertain"]
    where_diffusion_cell: str = ""
    where_franz: str = ""

    study_type: Literal["IVPT", "IVRT", "both", "uncertain"]
    barrier_category: Literal["skin", "synthetic_membrane", "both", "uncertain"]
    barrier_name_raw: str = ""

    # Endpoint existence
    endpoint_qfinal_found: Literal["yes", "no", "uncertain"]
    endpoint_time_found: Literal["yes", "no", "uncertain"]
    where_endpoint: str = ""
    figure_only: Literal["yes", "no", "uncertain"]

    # New: routing labels (endpoint + formulation)
    endpoint_carrier: Literal["table_text", "narrative", "figure", "supplement", "unknown"]
    endpoint_carrier_where: str = ""
    endpoint_carrier_snippet: str = ""

    formulation_carrier: Literal["table_text", "narrative", "supplement", "unknown"]
    formulation_carrier_where: str = ""
    formulation_carrier_snippet: str = ""

    endpoint_likely_in_supp: Literal["yes", "no", "uncertain"]
    formulation_table_missing: Literal["yes", "no", "uncertain"]

    notes: str = ""

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
            score += 3 if kw in ["ibuprofen", "franz", "table", "supporting information", "supplement",
                                 "cumulative", "amount permeated", "amount released", "% w/w", "mg/g"] else 1
    return score

def extract_relevant_pages(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    n = doc.page_count

    page_texts = []
    scores = []
    for pno in range(n):
        txt = doc.load_page(pno).get_text("text") or ""
        page_texts.append(txt)
        scores.append(score_page(txt))

    selected = []
    used = set()

    # always include first pages
    for pno in range(min(ALWAYS_INCLUDE_FIRST_PAGES, n)):
        used.add(pno)
        selected.append((999, pno, page_texts[pno]))

    # add top scored pages
    ranked = sorted([(scores[i], i) for i in range(n) if scores[i] > 0], reverse=True)
    for s, pno in ranked:
        if pno in used:
            continue
        selected.append((s, pno, page_texts[pno]))
        used.add(pno)
        if len(selected) >= MAX_PAGES_FOR_LLM:
            break

    if not selected:
        for pno in range(min(3, n)):
            selected.append((0, pno, page_texts[pno]))

    parts = []
    for s, pno, txt in sorted(selected, key=lambda x: x[1]):
        parts.append(f"\n--- PAGE {pno+1} (kw_score={s}) ---\n{txt}")

    doc.close()
    merged = norm_ws("\n".join(parts))
    return merged[:MAX_CHARS_TOTAL]

def backoff(attempt: int) -> None:
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def compute_route(res: dict) -> str:
    # endpoint route dominates
    c = res.get("endpoint_carrier")
    if c == "figure":
        return "figure"
    if c == "supplement":
        return "supplement"
    if c in ["table_text", "narrative"]:
        return "text"
    return "unknown"

def compute_ready_for_step6_text(res: dict) -> str:
    # NOTE: no 5% filter here
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
    # must be text-route
    if res.get("endpoint_carrier") not in ["table_text", "narrative"]:
        return "no"
    return "yes"

def index_one(doi: str, title: str, pdf_path: str) -> dict:
    pages_text = extract_relevant_pages(pdf_path)

    user = f"""
TASK: Evidence indexing + routing for diffusion-cell ibuprofen IVPT/IVRT dataset.

Return structured output ONLY. Do not guess; use ONLY provided pages.

You must decide:
A) Gates: primary experiment? ibuprofen studied? diffusion cell? Franz explicitly?
B) Endpoint existence: numeric cumulative amount at a timepoint/final time (Q_final@t_last) AND time present?
C) ROUTING (carrier labels) with evidence:
   - endpoint_carrier = table_text / narrative / figure / supplement / unknown
     Provide endpoint_carrier_where + endpoint_carrier_snippet that JUSTIFIES the label.
     Examples:
       * "Table 2 shows cumulative permeation..." -> table_text
       * "Fig. 3 shows cumulative permeation profiles..." and no numeric table -> figure
       * "see Supporting Information Table S1" -> supplement
       * numeric value stated directly in Results text -> narrative
   - formulation_carrier = table_text / narrative / supplement / unknown
     Provide formulation_carrier_where + formulation_carrier_snippet.

Also fill where_endpoint / where_franz / where_diffusion_cell if possible.

META:
DOI: {doi}
TITLE: {title}

PAGES:
{pages_text}
""".strip()

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": user}],
        text_format=EvidenceIndexV13
    )
    return resp.output_parsed.model_dump()

def main():
    df = pd.read_csv(INV_PATH)
    df["pdf_status"] = df["pdf_status"].fillna("").astype(str).str.lower()
    df_ok = df[df["pdf_status"] == "ok"].copy()

    out_file = Path(OUT_INDEX)
    if out_file.exists():
        out_file.unlink()

    rows = []
    total = len(df_ok)

    for k, (_, row) in enumerate(df_ok.iterrows(), start=1):
        doi = str(row.get("doi") or "").strip()
        title = str(row.get("title") or "").strip()
        pdf_path = str(row.get("pdf_path") or "").strip()

        attempt = 0
        while True:
            try:
                res = index_one(doi, title, pdf_path)
                route = compute_route(res)
                ready_text = compute_ready_for_step6_text(res)
                rec = {
                    "doi": doi,
                    "title": title,
                    "pdf_path": pdf_path,
                    "route": route,
                    "ready_for_step6_text": ready_text,
                    **res
                }
                rows.append(rec)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    rows.append({
                        "doi": doi, "title": title, "pdf_path": pdf_path,
                        "route": "unknown", "ready_for_step6_text": "no",
                        "paper_type": "uncertain",
                        "mentions_ibuprofen": "uncertain",
                        "mentions_diffusion_cell": "uncertain",
                        "franz_confirmed": "uncertain",
                        "where_diffusion_cell": "", "where_franz": "",
                        "study_type": "uncertain",
                        "barrier_category": "uncertain",
                        "barrier_name_raw": "",
                        "endpoint_qfinal_found": "uncertain",
                        "endpoint_time_found": "uncertain",
                        "where_endpoint": "",
                        "figure_only": "uncertain",
                        "endpoint_carrier": "unknown",
                        "endpoint_carrier_where": "",
                        "endpoint_carrier_snippet": "",
                        "formulation_carrier": "unknown",
                        "formulation_carrier_where": "",
                        "formulation_carrier_snippet": "",
                        "endpoint_likely_in_supp": "uncertain",
                        "formulation_table_missing": "uncertain",
                        "notes": f"error:{type(e).__name__}"
                    })
                    break
                backoff(attempt)

        if len(rows) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: {len(rows)}/{total}", flush=True)
        time.sleep(SLEEP_BETWEEN)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_file, index=False, encoding="utf-8-sig")

    # Build queues
    q_text = df_out[df_out["ready_for_step6_text"] == "yes"].copy()
    q_fig = df_out[df_out["route"] == "figure"].copy()
    q_supp = df_out[(df_out["route"] == "supplement") | (df_out["endpoint_likely_in_supp"] == "yes") | (df_out["formulation_table_missing"] == "yes")].copy()

    q_text.to_csv(OUT_QUEUE_TEXT, index=False, encoding="utf-8-sig")
    q_fig.to_csv(OUT_QUEUE_FIG, index=False, encoding="utf-8-sig")
    q_supp.to_csv(OUT_QUEUE_SUPP, index=False, encoding="utf-8-sig")

    print("Done. Saved:", OUT_INDEX)
    print("route counts:\n", df_out["route"].value_counts(dropna=False))
    print("ready_for_step6_text counts:\n", df_out["ready_for_step6_text"].value_counts(dropna=False))
    print("Saved text queue:", OUT_QUEUE_TEXT, "rows:", len(q_text))
    print("Saved figure queue:", OUT_QUEUE_FIG, "rows:", len(q_fig))
    print("Saved supp queue:", OUT_QUEUE_SUPP, "rows:", len(q_supp))

if __name__ == "__main__":
    main()
