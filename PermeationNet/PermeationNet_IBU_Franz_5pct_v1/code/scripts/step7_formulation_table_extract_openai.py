import os
import re
import json
import time
import base64
import random
import ast
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple

import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field

# =========================
# Paths
# =========================
TRIAGE_CSV = Path("GenAI/outputs/figure_triage_v1.csv")
DIG_ENDPTS_CSV = Path("GenAI/outputs/figure_digitized_endpoints.csv")

OUT_JSONL = Path("GenAI/outputs/figure_formulations_v1.jsonl")
OUT_CSV = Path("GenAI/outputs/figure_formulations_v1.csv")
OUT_FLAT = Path("GenAI/outputs/figure_formulations_v1_flat.csv")

# =========================
# LLM config
# =========================
MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 90
SLEEP_BETWEEN = 0.15
CHECKPOINT_EVERY = 3
MAX_RETRIES = 6

client = OpenAI(timeout=TIMEOUT_SECONDS)

SYSTEM = (
    "You extract formulation composition tables from scientific PDFs. "
    "Use ONLY the provided page content/images. "
    "Do NOT guess concentrations. If missing/unclear, set null and explain with evidence."
)

# =========================
# Page picking config
# =========================
MAX_DOC_SCAN_PAGES = 200
MAX_PAGES_TO_SEND = 5         # text pages
NEIGHBORS = 1                 # include +/- neighbors around candidate pages
MAX_IMAGES_TO_SEND = 2        # optional images if text looks weak
DPI = 170                     # for page screenshots if needed

# Keywords for formulation tables
TAB_KW = [
    "table", "formulation", "formulations", "composition", "ingredient",
    "vehicle", "w/w", "wt%", "weight", "%", "mg", "g", "ml", "v/v",
    "propylene glycol", "ethanol", "poloxamer", "carbomer", "gel", "cream",
    "ibuprofen"
]

def norm_ws(s: str) -> str:
    return " ".join((s or "").split())

def safe_json_loads(s: str):
    s = (s or "").strip()
    if not s:
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

def page_graphics_flags(page: fitz.Page) -> Tuple[int, float]:
    # embedded images + vector drawings ratio
    try:
        has_img = 1 if len(page.get_images(full=True)) > 0 else 0
    except Exception:
        has_img = 0
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    page_area = float(page.rect.width * page.rect.height) or 1.0
    area = 0.0
    for d in drawings:
        r = d.get("rect", None)
        if r is None:
            continue
        area += float(r.width * r.height)
    draw_ratio = min(1.0, area / page_area) if page_area else 0.0
    return has_img, draw_ratio

def score_page_for_formulation_table(text: str) -> int:
    t = f" {text.lower()} "
    s = 0
    # strong signals
    if "table" in t:
        s += 6
    if "formulation" in t or "composition" in t:
        s += 6
    if "ingredient" in t:
        s += 3
    if "ibuprofen" in t:
        s += 3
    # unit-ish signals
    if "w/w" in t or "wt%" in t or "%" in t:
        s += 2
    if "mg" in t or " g " in t or "ml" in t or "v/v" in t:
        s += 1

    # general keywords
    for kw in TAB_KW:
        if kw in t:
            s += 1
    return s

def render_page_jpg_dataurl(pdf_path: str, page_index: int, dpi: int = DPI) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("jpg")
    doc.close()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def pick_candidate_pages(pdf_path: str) -> Tuple[List[int], str, List[int]]:
    """
    Return:
      pages_selected (0-index),
      debug string for top candidates,
      weak_text_pages (subset of pages_selected that look like tables but text is weak)
    """
    doc = fitz.open(pdf_path)
    n = doc.page_count
    n_scan = min(n, MAX_DOC_SCAN_PAGES)

    scored: List[Tuple[float, int, int, int, float]] = []
    # (score_total, pno, text_score, has_img, draw_ratio)

    for i in range(n_scan):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        txt_n = norm_ws(txt)
        text_score = score_page_for_formulation_table(txt_n)

        has_img, draw_ratio = page_graphics_flags(page)

        # If a page has high drawings/images but low text, it might be a table image -> still interesting.
        score_total = text_score + 5 * has_img + (4 if draw_ratio >= 0.02 else 0)

        if score_total > 0:
            scored.append((score_total, i, text_score, has_img, draw_ratio))

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = set()
    weak_text_pages = set()

    # take top scored pages
    for (score_total, pno, text_score, has_img, draw_ratio) in scored[:MAX_PAGES_TO_SEND]:
        selected.add(pno)
        # weak text but likely table graphics
        if text_score <= 3 and (has_img or draw_ratio >= 0.02):
            weak_text_pages.add(pno)

    # add neighbors
    expanded = set()
    for p in selected:
        for j in range(p - NEIGHBORS, p + NEIGHBORS + 1):
            if 0 <= j < n:
                expanded.add(j)
    selected = expanded

    pages = sorted(selected)
    # keep within MAX_PAGES_TO_SEND after expansion by re-ranking
    if len(pages) > MAX_PAGES_TO_SEND:
        # prioritize by scored order
        order = {pno: rank for rank, (_, pno, *_rest) in enumerate(scored)}
        pages.sort(key=lambda p: order.get(p, 10**9))
        pages = sorted(set(pages[:MAX_PAGES_TO_SEND]))

    # decide weak pages among final selection
    weak_final = [p for p in pages if p in weak_text_pages]

    dbg = []
    for (score_total, pno, text_score, has_img, draw_ratio) in scored[:10]:
        dbg.append(f"p{pno+1}:score={score_total:.1f},text={text_score},img={has_img},draw={draw_ratio:.3f}")
    debug = " | ".join(dbg)

    doc.close()
    return pages, debug, weak_final

def build_pages_text(pdf_path: str, pages0: List[int], max_chars_per_page: int = 4500) -> str:
    doc = fitz.open(pdf_path)
    blocks = []
    for p0 in pages0:
        page = doc.load_page(p0)
        txt = page.get_text("text") or ""
        txt = norm_ws(txt)
        if len(txt) > max_chars_per_page:
            txt = txt[:max_chars_per_page] + " …[truncated]"
        blocks.append(f"--- PAGE {p0+1} ---\n{txt}")
    doc.close()
    return "\n\n".join(blocks)

# =========================
# Output schema
# =========================
class Component(BaseModel):
    name_raw: str = ""
    concentration_value: Optional[float] = None
    concentration_unit: str = ""   # e.g., %, % w/w, mg/g, g/100g, v/v
    basis: str = ""                # e.g., w/w, v/v, w/v, mass fraction
    remark: str = ""               # e.g., "qs to 100", "balance", "range"

class Formulation(BaseModel):
    formulation_label: str = ""    # e.g., F1, Formulation A, Gel-2
    dosage_form: str = ""          # gel/cream/solution/etc if stated
    api_name: str = "ibuprofen"
    api_concentration_value: Optional[float] = None
    api_concentration_unit: str = ""
    api_basis: str = ""
    components: List[Component] = Field(default_factory=list)
    water_qs_or_balance: Literal["yes", "no", "uncertain"] = "uncertain"
    table_id: str = ""             # e.g., Table 1
    source_pages: List[int] = Field(default_factory=list)  # 1-indexed pages used
    evidence_snippet: str = ""     # short snippet proving the table/rows exist
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""

class ExtractionResult(BaseModel):
    formulations: List[Formulation] = Field(default_factory=list)
    formulation_table_found: Literal["yes", "no", "uncertain"] = "uncertain"
    formulation_table_in_image: Literal["yes", "no", "uncertain"] = "uncertain"
    recommended_next: Literal["ok", "need_more_pages", "need_supp", "manual_review"] = "manual_review"
    notes: str = ""

def extract_formulation_table(doi: str,
                              title: str,
                              curve_hints: str,
                              pages_desc: str,
                              images: List[Tuple[int, str]]) -> dict:
    """
    pages_desc: text blocks for pages
    images: list of (page_1idx, data_url) for weak-text pages (optional)
    """
    intro = f"""
You are extracting formulation composition tables from a paper (topical ibuprofen).

Goal:
- Extract ALL formulations/vehicles described in composition tables relevant to the in vitro permeation/release experiments.
- For each formulation, capture API (ibuprofen) concentration and all excipients with concentrations/units/basis as reported.

Rules:
- Use ONLY the provided page text and optional page images.
- Do NOT guess numeric values or units. If unclear, set fields to null/empty and explain.
- If water is described as "qs", "to 100%", "balance", mark water_qs_or_balance=yes and keep remark.
- If the table uses ranges or missing values, preserve as remark and leave numeric null if needed.
- Provide table_id (e.g., "Table 1") if stated and include source_pages.
- Evidence snippet must be short (<= ~25 words).

Curve hints (from figure digitization/legend; may help mapping labels):
{curve_hints}

Provided content:
{pages_desc}

Return structured output only.
""".strip()

    content = [{"type": "input_text", "text": intro}]
    for p1, data_url in images:
        content.append({"type": "input_image", "image_url": data_url})

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": content}],
        text_format=ExtractionResult
    )
    return resp.output_parsed.model_dump()

def load_done(no_resume: bool) -> set:
    done = set()
    if no_resume:
        return done
    if OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(str(obj.get("doi", "")).strip().lower())
                except Exception:
                    pass
    return done

def main(limit: Optional[int] = None, reset: bool = False, no_resume: bool = False):
    if reset:
        for p in [OUT_JSONL, OUT_CSV, OUT_FLAT]:
            if p.exists():
                p.unlink()

    # 1) Only process papers that successfully produced amount endpoints (i.e., the 10 candidates)
    df_end = pd.read_csv(DIG_ENDPTS_CSV)
    dois = sorted(set(str(x).strip() for x in df_end["doi"].dropna().tolist()))
    print(f"[FormTable] DOIs from digitized endpoints: {len(dois)}")

    if limit:
        dois = dois[:limit]
        print(f"[FormTable] Using limit={limit}. DOIs to process: {len(dois)}")

    df_tri = pd.read_csv(TRIAGE_CSV)
    tri_map = {str(r["doi"]).strip().lower(): r for _, r in df_tri.iterrows()}

    done = load_done(no_resume=no_resume)
    rows = []
    flat_rows = []

    processed = 0
    for doi in dois:
        if doi.lower() in done:
            continue

        tri = tri_map.get(doi.lower(), None)
        if tri is None:
            print(f"[FormTable] WARN: DOI not found in triage csv: {doi}")
            continue

        pdf_path = str(tri.get("pdf_path", "")).strip()
        title = str(tri.get("title", "")).strip()

        # curve hints from endpoints + legend
        sub_end = df_end[df_end["doi"].astype(str).str.strip().str.lower() == doi.lower()].copy()
        curve_summary = []
        if len(sub_end) > 0:
            for _, er in sub_end.iterrows():
                curve_summary.append(f"{er.get('curve_id','')}:{er.get('curve_color','')}")
        curve_summary = ", ".join([c for c in curve_summary if c]) or "none"

        legend = safe_json_loads(str(tri.get("legend", "")))
        legend_summary = ""
        if isinstance(legend, list) and legend:
            parts = []
            for item in legend:
                if isinstance(item, dict):
                    lab = str(item.get("label","")).strip()
                    col = str(item.get("color_hint","")).strip()
                    if lab or col:
                        parts.append(f"{lab} ({col})".strip())
            legend_summary = "; ".join(parts)

        curve_hints = f"- curves_extracted: {curve_summary}\n- legend_entries: {legend_summary or 'none'}"

        # 2) pick candidate pages for formulation table
        pages0, debug, weak_pages0 = pick_candidate_pages(pdf_path)
        pages_desc = build_pages_text(pdf_path, pages0)

        # 3) If some pages likely contain table as image, attach up to MAX_IMAGES_TO_SEND screenshots
        images = []
        if weak_pages0:
            for p0 in weak_pages0[:MAX_IMAGES_TO_SEND]:
                images.append((p0 + 1, render_page_jpg_dataurl(pdf_path, p0, dpi=DPI)))

        attempt = 0
        while True:
            try:
                out = extract_formulation_table(
                    doi=doi,
                    title=title,
                    curve_hints=curve_hints,
                    pages_desc=pages_desc,
                    images=images
                )

                rec = {
                    "doi": doi,
                    "title": title,
                    "pdf_path": pdf_path,
                    "pages_used": ",".join([str(p+1) for p in pages0]),
                    "page_pick_debug": debug,
                    "weak_pages_images": ",".join([str(p+1) for p in weak_pages0[:MAX_IMAGES_TO_SEND]]),
                    "curve_hints": curve_hints,
                    **out
                }

                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                rows.append(rec)
                done.add(doi.lower())
                processed += 1

                # flatten formulations
                for fm in out.get("formulations", []):
                    # store components as json string
                    comps = fm.get("components", [])
                    flat_rows.append({
                        "doi": doi,
                        "title": title,
                        "pdf_path": pdf_path,
                        "formulation_label": fm.get("formulation_label",""),
                        "dosage_form": fm.get("dosage_form",""),
                        "api_name": fm.get("api_name","ibuprofen"),
                        "api_concentration_value": fm.get("api_concentration_value", None),
                        "api_concentration_unit": fm.get("api_concentration_unit",""),
                        "api_basis": fm.get("api_basis",""),
                        "water_qs_or_balance": fm.get("water_qs_or_balance","uncertain"),
                        "table_id": fm.get("table_id",""),
                        "source_pages": ",".join(str(x) for x in (fm.get("source_pages", []) or [])),
                        "evidence_snippet": fm.get("evidence_snippet",""),
                        "confidence": fm.get("confidence", None),
                        "notes": fm.get("notes",""),
                        "components_json": json.dumps(comps, ensure_ascii=False)
                    })

                break
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    print(f"[FormTable] ERROR: {doi} failed with {type(e).__name__}")
                    # write a stub record
                    rec = {
                        "doi": doi, "title": title, "pdf_path": pdf_path,
                        "pages_used": ",".join([str(p+1) for p in pages0]),
                        "page_pick_debug": debug,
                        "weak_pages_images": ",".join([str(p+1) for p in weak_pages0[:MAX_IMAGES_TO_SEND]]),
                        "curve_hints": curve_hints,
                        "formulations": [],
                        "formulation_table_found": "uncertain",
                        "formulation_table_in_image": "uncertain",
                        "recommended_next": "manual_review",
                        "notes": f"error:{type(e).__name__}"
                    }
                    with open(OUT_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows.append(rec)
                    done.add(doi.lower())
                    processed += 1
                    break
                backoff(attempt)

        if processed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            pd.DataFrame(flat_rows).to_csv(OUT_FLAT, index=False, encoding="utf-8-sig")
            print(f"[FormTable] checkpoint processed={processed}", flush=True)

        time.sleep(SLEEP_BETWEEN)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(flat_rows).to_csv(OUT_FLAT, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_FLAT)
    if "formulation_table_found" in df_out.columns and len(df_out) > 0:
        print("\nformulation_table_found counts:\n", df_out["formulation_table_found"].value_counts(dropna=False))
        print("\nrecommended_next counts:\n", df_out["recommended_next"].value_counts(dropna=False))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    main(limit=args.limit, reset=args.reset, no_resume=args.no_resume)
