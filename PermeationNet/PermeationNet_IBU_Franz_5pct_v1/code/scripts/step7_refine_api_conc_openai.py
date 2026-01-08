import json
import time
import random
import base64
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple

import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field

# -----------------------
# Inputs / Outputs
# -----------------------
NEED_MANUAL = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_need_manual_5pct.csv")
FORM_PAPER = Path("GenAI/outputs/figure_formulations_v1.csv")  # has pdf_path, pages_used
OUT_QC = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_qc_refined.csv")
OUT_PASS = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_pass5pct_refined.csv")
OUT_REJECT = Path("GenAI/outputs/PermeationNet_IBU_Franz_5pct_v1_figure_still_not5_or_unknown.csv")
OUT_JSONL = Path("GenAI/outputs/step7_refine_api_conc_runs.jsonl")

# -----------------------
# LLM config
# -----------------------
MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 90
MAX_RETRIES = 6
SLEEP_BETWEEN = 0.15

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You extract ibuprofen concentration for each formulation label from the provided PDF page text/images. "
    "Do not guess. If the concentration is not explicitly stated, return null and explain briefly."
)

# -----------------------
# Page selection config
# -----------------------
NEIGHBOR = 2
MAX_PAGES_TEXT = 8
MAX_IMAGES = 2
DPI = 200

# -----------------------
# Helpers: unit conversion to % w/w
# -----------------------
def to_percent_w_w(value: Optional[float], unit: str, basis: str) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    u = (unit or "").lower().replace("µ", "u").replace(" ", "")
    b = (basis or "").lower()

    v = float(value)

    # percent cases
    if "%" in u or "wt%" in u or "w/w" in u:
        return v

    # mg/g -> % (w/w): mg/g * 0.1
    if "mg/g" in u or "mgperg" in u:
        return v * 0.1

    # g/100g -> % (w/w): same numeric value
    if "g/100g" in u or "gper100g" in u:
        return v

    # mg per g sometimes written in basis field
    if ("mg/g" in b) and ("%" not in u):
        return v * 0.1

    return None

def is_5pct(p: Optional[float]) -> bool:
    return p is not None and abs(float(p) - 5.0) < 1e-6

def backoff(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def parse_pages_used(pages_used: str) -> List[int]:
    """
    pages_used is like "2,3,4,6,8" (1-indexed)
    """
    if pages_used is None:
        return []
    s = str(pages_used).strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return sorted(set(out))

def expand_pages(pages_1idx: List[int], doc_n: int, neighbor: int = NEIGHBOR) -> List[int]:
    s = set()
    for p in pages_1idx:
        for j in range(p - neighbor, p + neighbor + 1):
            if 1 <= j <= doc_n:
                s.add(j)
    return sorted(s)

def read_pages_text(pdf_path: str, pages_1idx: List[int], max_chars_per_page: int = 4500) -> str:
    doc = fitz.open(pdf_path)
    blocks = []
    for p1 in pages_1idx[:MAX_PAGES_TEXT]:
        page = doc.load_page(p1 - 1)
        txt = page.get_text("text") or ""
        txt = " ".join(txt.split())
        if len(txt) > max_chars_per_page:
            txt = txt[:max_chars_per_page] + " …[truncated]"
        blocks.append(f"--- PAGE {p1} ---\n{txt}")
    doc.close()
    return "\n\n".join(blocks)

def page_to_data_url(pdf_path: str, page_1idx: int, dpi: int = DPI) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_1idx - 1)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("jpg")
    doc.close()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# -----------------------
# Output schema
# -----------------------
class ApiConcItem(BaseModel):
    formulation_label: str
    api_value: Optional[float] = None
    api_unit: str = ""     # "% w/w", "%", "mg/g", "g/100g", etc
    api_basis: str = ""    # "w/w" if known
    evidence_page: Optional[int] = None
    evidence_snippet: str = ""  # <= 25 words
    confidence: float = Field(..., ge=0.0, le=1.0)

class ApiConcResult(BaseModel):
    items: List[ApiConcItem] = Field(default_factory=list)
    notes: str = ""

def call_llm(doi: str, title: str, labels: List[str], pages_text: str, images: List[Tuple[int, str]]) -> Dict[str, Any]:
    label_txt = "\n".join([f"- {x}" for x in labels]) or "(none)"

    prompt = f"""
Extract ibuprofen concentration for each formulation_label.

Paper:
DOI: {doi}
TITLE: {title}

Formulation labels (must output one item per label; if unknown -> null):
{label_txt}

Rules:
- Use ONLY the provided content (page text and optional page images).
- Do NOT guess.
- If ibuprofen concentration is given as %, mg/g, g/100 g, etc, record it.
- If the paper uses drug loading in mg/mL or similar, record that unit exactly (do not convert unless it is directly mg/g or g/100g).
- Evidence_snippet must be short (<=25 words) and include the key value if possible.

Provided pages:
{pages_text}

Return structured output only.
""".strip()

    content = [{"type": "input_text", "text": prompt}]
    for p1, data_url in images:
        content.append({"type": "input_image", "image_url": data_url})

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": content}],
        text_format=ApiConcResult
    )
    return resp.output_parsed.model_dump()

def main():
    df_need = pd.read_csv(NEED_MANUAL)
    df_form = pd.read_csv(FORM_PAPER)

    form_map = {str(r["doi"]).strip().lower(): r for _, r in df_form.iterrows()}

    out_rows = []
    print(f"[RefineAPI] need_manual rows: {len(df_need)} | DOIs: {df_need['doi'].nunique()}")

    for doi, sub in df_need.groupby(df_need["doi"].astype(str).str.strip().str.lower()):
        paper = form_map.get(doi, None)
        if paper is None:
            continue
        title = str(paper.get("title", "") or "").strip()
        pdf_path = str(paper.get("pdf_path", "") or "").strip()
        pages_used = parse_pages_used(str(paper.get("pages_used", "") or ""))

        doc = fitz.open(pdf_path)
        doc_n = doc.page_count
        doc.close()

        pages = expand_pages(pages_used, doc_n, neighbor=NEIGHBOR)
        pages_text = read_pages_text(pdf_path, pages)

        # Provide 1–2 page images: pick first two of pages_used (table pages) if available
        images = []
        for p1 in pages_used[:MAX_IMAGES]:
            images.append((p1, page_to_data_url(pdf_path, p1)))

        labels = sorted(set(sub["formulation_label"].astype(str).tolist()))

        attempt = 0
        while True:
            try:
                out = call_llm(doi=doi, title=title, labels=labels, pages_text=pages_text, images=images)
                # save run record
                OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "doi": doi,
                        "title": title,
                        "pdf_path": pdf_path,
                        "pages_used": pages_used,
                        "pages_sent": pages[:MAX_PAGES_TEXT],
                        "labels": labels,
                        "llm": out
                    }, ensure_ascii=False) + "\n")

                # build lookup
                items = out.get("items", []) or []
                item_map = {str(it.get("formulation_label","")).strip(): it for it in items}

                for _, r in sub.iterrows():
                    lab = str(r.get("formulation_label","")).strip()
                    it = item_map.get(lab, {})

                    api_val = it.get("api_value", None)
                    api_unit = str(it.get("api_unit","") or "")
                    api_basis = str(it.get("api_basis","") or "")

                    pct = to_percent_w_w(api_val, api_unit, api_basis)
                    out_rows.append({
                        **r.to_dict(),
                        "api_value_refined": api_val,
                        "api_unit_refined": api_unit,
                        "api_basis_refined": api_basis,
                        "api_pct_refined": pct,
                        "api_5pct_confirmed_refined": bool(is_5pct(pct)),
                        "api_refine_evidence_page": it.get("evidence_page", None),
                        "api_refine_evidence_snippet": str(it.get("evidence_snippet","") or ""),
                        "api_refine_confidence": it.get("confidence", None),
                    })
                break
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    for _, r in sub.iterrows():
                        out_rows.append({
                            **r.to_dict(),
                            "api_value_refined": None,
                            "api_unit_refined": "",
                            "api_basis_refined": "",
                            "api_pct_refined": None,
                            "api_5pct_confirmed_refined": False,
                            "api_refine_evidence_page": None,
                            "api_refine_evidence_snippet": f"error:{type(e).__name__}",
                            "api_refine_confidence": 0.0,
                        })
                    break
                backoff(attempt)
            time.sleep(SLEEP_BETWEEN)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_QC, index=False, encoding="utf-8-sig")

    df_pass = df_out[df_out["api_5pct_confirmed_refined"] == True].copy()
    df_rej = df_out[df_out["api_5pct_confirmed_refined"] == False].copy()
    df_pass.to_csv(OUT_PASS, index=False, encoding="utf-8-sig")
    df_rej.to_csv(OUT_REJECT, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Saved:", OUT_QC)
    print("Saved pass5pct refined:", OUT_PASS, "rows:", len(df_pass))
    print("Saved still not5/unknown:", OUT_REJECT, "rows:", len(df_rej))
    print("\napi_5pct_confirmed_refined counts:\n", df_out["api_5pct_confirmed_refined"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
