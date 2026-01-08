import re
import json
import time
import base64
import random
from pathlib import Path
from typing import List, Optional, Literal, Tuple

import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
QUEUE_PATH = "GenAI/outputs/extraction_queue_step6_figure.csv"
OUT_DIR = Path("GenAI/outputs")
IMG_DIR = OUT_DIR / "figure_triage_images"
OUT_JSONL = OUT_DIR / "figure_triage_v1.jsonl"
OUT_CSV = OUT_DIR / "figure_triage_v1.csv"

IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"   # vision-capable
TIMEOUT_SECONDS = 90

MAX_PAGES_TO_SEND_DEFAULT = 3
DPI_DEFAULT = 170
ANCHOR_NEIGHBOR = 2
MAX_DOC_SCAN_PAGES = 160

ALLOW_TEXT_ONLY_PAGES = 1  # at most 1 page without graphics

CHECKPOINT_EVERY = 5
SLEEP_BETWEEN = 0.2

client = OpenAI(timeout=TIMEOUT_SECONDS)

FIG_KW = [
    "figure", "fig.", "fig ", "cumulative", "permeat", "release",
    "ibuprofen", "flux", "jss", "profile", "time", "hours",
    "µg/cm", "ug/cm", "mg/cm", "ng/cm"
]

SYSTEM = (
    "You are a strict scientific figure triage + calibration assistant. "
    "Use ONLY the provided page images. Do not guess numeric values. "
    "If axis ranges/units or plot bounding box cannot be read, set them to null and explain."
)

class LegendEntry(BaseModel):
    label: str = ""
    color_hint: str = ""  # e.g., red/blue/black/green/orange/purple/gray

class FigureTriage(BaseModel):
    # --- triage ---
    endpoint_curve_present: Literal["yes", "no", "uncertain"]
    likely_endpoint_type: Literal["cumulative_amount", "flux", "jss", "unknown"]
    figure_id: str = ""                  # e.g., "Fig. 2", "Figure 3b"
    page_number: Optional[int] = None    # 1-indexed
    subplot: str = ""                    # e.g., "b", "panel a", "top-right"

    # IMPORTANT: "digitizable=yes" should imply bbox + axis ranges are readable enough.
    digitizable: Literal["yes", "no", "uncertain"]
    why_not_digitizable: str = ""

    ticks_readable: Literal["yes", "no", "uncertain"]
    legend_present: Literal["yes", "no", "uncertain"]
    approx_curves_count: Optional[int] = None
    legend: List[LegendEntry] = Field(default_factory=list)

    suggests_table_exists: Literal["yes", "no", "uncertain"]
    suggests_supp_exists: Literal["yes", "no", "uncertain"]
    recommended_route: Literal["digitize", "supp_needed", "text_table_maybe", "skip"] = "skip"

    # --- calibration (only if digitizable=yes; otherwise keep null) ---
    plot_bbox: Optional[List[float]] = None  # normalized [x0,y0,x1,y1] in the chosen PAGE image
    axes_x_label: str = ""
    axes_x_unit: str = ""
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    axes_y_label: str = ""
    axes_y_unit: str = ""
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    y_kind: Literal["amount_per_area", "amount_total", "percent", "unknown"] = "unknown"

    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""

def parse_anchor_pages(where_text: str) -> List[int]:
    if not where_text or str(where_text).strip() == "":
        return []
    t = str(where_text)
    nums = set()
    for m in re.finditer(r"\bpage\s*(\d+)\b", t, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"\bp\.?\s*(\d+)\b", t, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))
    out = []
    for n in sorted(nums):
        if n > 0:
            out.append(n - 1)  # 0-index
    return out

def score_text_for_figure(text: str) -> int:
    low = f" {text.lower()} "
    s = 0
    for kw in FIG_KW:
        if kw in low:
            s += 3 if kw in ["figure", "fig.", "cumulative", "permeat", "release"] else 1
    return s

def drawing_area_ratio(page: fitz.Page) -> float:
    try:
        drawings = page.get_drawings()
    except Exception:
        return 0.0
    if not drawings:
        return 0.0
    page_area = float(page.rect.width * page.rect.height) or 1.0
    area = 0.0
    for d in drawings:
        r = d.get("rect", None)
        if r is None:
            continue
        area += float(r.width * r.height)
    return min(1.0, area / page_area)

def page_graphics_flags(page: fitz.Page) -> Tuple[int, float]:
    try:
        has_img = 1 if len(page.get_images(full=True)) > 0 else 0
    except Exception:
        has_img = 0
    draw_ratio = drawing_area_ratio(page)
    return has_img, draw_ratio

def pick_candidate_pages(pdf_path: str, anchors: List[int], max_pages_to_send: int) -> Tuple[List[int], str]:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    n_scan = min(n, MAX_DOC_SCAN_PAGES)

    candidates: List[Tuple[float, int, int, int, float, int]] = []
    # (score_total, pno, text_score, has_img, draw_ratio, fig_mention)

    for i in range(n_scan):
        page = doc.load_page(i)
        txt = page.get_text("text") or ""
        text_score = score_text_for_figure(txt)

        has_img, draw_ratio = page_graphics_flags(page)
        fig_mention = 1 if ("fig." in txt.lower() or "figure" in txt.lower()) else 0
        has_draw = 1 if draw_ratio >= 0.02 else 0

        score_total = text_score + 7 * has_img + 5 * has_draw + 2 * fig_mention
        if score_total > 0:
            candidates.append((score_total, i, text_score, has_img, draw_ratio, fig_mention))

    # anchor window
    anchor_window = set()
    for a in anchors:
        for j in range(a - ANCHOR_NEIGHBOR, a + ANCHOR_NEIGHBOR + 1):
            if 0 <= j < n:
                anchor_window.add(j)

    candidates.sort(reverse=True, key=lambda x: x[0])

    selected: List[int] = []
    selected_set = set()
    text_only_used = 0

    # 1) graphics-first pages
    for score_total, pno, text_score, has_img, draw_ratio, fig_mention in candidates:
        if len(selected) >= max_pages_to_send:
            break
        if pno in selected_set:
            continue
        has_draw = draw_ratio >= 0.02
        if has_img or has_draw:
            selected.append(pno)
            selected_set.add(pno)

    # 2) anchor window pages (limit text-only)
    if len(selected) < max_pages_to_send and anchor_window:
        for pno in sorted(anchor_window):
            if len(selected) >= max_pages_to_send:
                break
            if pno in selected_set:
                continue
            page = doc.load_page(pno)
            has_img, draw_ratio = page_graphics_flags(page)
            has_draw = draw_ratio >= 0.02
            has_graphics = bool(has_img or has_draw)
            if not has_graphics:
                if text_only_used >= ALLOW_TEXT_ONLY_PAGES:
                    continue
                text_only_used += 1
            selected.append(pno)
            selected_set.add(pno)

    # 3) fill remaining with best scored pages (allow tiny text-only)
    for score_total, pno, text_score, has_img, draw_ratio, fig_mention in candidates:
        if len(selected) >= max_pages_to_send:
            break
        if pno in selected_set:
            continue
        has_draw = draw_ratio >= 0.02
        has_graphics = bool(has_img or has_draw)
        if not has_graphics:
            if text_only_used >= ALLOW_TEXT_ONLY_PAGES:
                continue
            text_only_used += 1
        selected.append(pno)
        selected_set.add(pno)

    if not selected:
        selected = [0]

    selected = sorted(selected)

    dbg_parts = []
    for score_total, pno, text_score, has_img, draw_ratio, fig_mention in candidates[:8]:
        dbg_parts.append(
            f"p{pno+1}:score={score_total:.1f},text={text_score},img={has_img},draw_ratio={draw_ratio:.3f},figmention={fig_mention}"
        )
    debug = " | ".join(dbg_parts)

    doc.close()
    return selected, debug

def render_page_image_bytes(pdf_path: str, page_index: int, dpi: int) -> Tuple[bytes, str]:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("jpg")  # smaller than png
    doc.close()
    return img_bytes, "image/jpeg"

def save_page_image(doi: str, page_no_1idx: int, img_bytes: bytes) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doi)[:80] or "no_doi"
    out = IMG_DIR / f"{safe}__p{page_no_1idx}.jpg"
    out.write_bytes(img_bytes)
    return str(out)

def backoff(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def triage_with_vision(doi: str, title: str, page_map: List[Tuple[int, str]]) -> dict:
    """
    page_map: list of (page_1idx, data_url)
    The model should choose page_number among these pages, and bbox/ranges must refer to that chosen page.
    """
    pages_desc = ", ".join([f"Image {i+1} = PAGE {p}" for i, (p, _) in enumerate(page_map)])

    intro = f"""
You will be shown 1–3 PAGE images from a scientific PDF.

Primary goal (Triage):
1) Determine whether a FIGURE on these pages contains an endpoint curve relevant to in vitro permeation/release (ideally cumulative amount vs time).
2) Decide if the curve is DIGITIZABLE reliably.

Routing:
- If digitizable=yes -> recommended_route must be "digitize".
- If endpoint seems in supplement -> "supp_needed".
- If you suspect numeric tables exist elsewhere in the main text -> "text_table_maybe".
- Otherwise -> "skip".

Calibration requirement (ONLY if digitizable=yes):
A) Choose the single page that contains the best target plot and set page_number to that PAGE.
B) Provide plot_bbox as normalized coordinates [x0,y0,x1,y1] in THAT chosen page image, tightly covering the plotting region (axes box + curves).
   - Do NOT include titles/captions outside the plot.
   - Coordinates must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1.
C) If readable, extract x_min, x_max, x_unit and y_min, y_max, y_unit (numbers must be read, never guessed).
   If any of these cannot be read, set them to null and explain in notes.
D) Set digitizable=yes ONLY if plot_bbox is provided AND axis ranges (x_min/x_max and y_min/y_max) are readable enough for digitization.

Legend:
If legend entries are readable, list them with a coarse color_hint.

Do NOT guess numeric values.

Paper:
DOI: {doi}
TITLE: {title}

Images correspond to pages:
{pages_desc}

Return structured output only.
""".strip()

    content = [{"type": "input_text", "text": intro}]
    for _, data_url in page_map:
        content.append({"type": "input_image", "image_url": data_url})

    resp = client.responses.parse(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM},
               {"role": "user", "content": content}],
        text_format=FigureTriage
    )
    return resp.output_parsed.model_dump()

def load_existing_done(no_resume: bool) -> set:
    done = set()
    if no_resume:
        return done
    if OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add((str(obj.get("doi", "")).strip().lower(), str(obj.get("pdf_path", "")).strip()))
                except Exception:
                    pass
    return done

def pick_selected_image_path(saved_imgs: List[str], pages_1idx: List[int], chosen_page: Optional[int]) -> str:
    if not saved_imgs:
        return ""
    if chosen_page is None:
        return saved_imgs[0]
    for p, path in zip(pages_1idx, saved_imgs):
        if p == chosen_page:
            return path
    return saved_imgs[0]

def main(limit: Optional[int] = None,
         no_resume: bool = False,
         reset: bool = False,
         max_pages_to_send: int = MAX_PAGES_TO_SEND_DEFAULT,
         dpi: int = DPI_DEFAULT):

    if reset:
        for p in [OUT_JSONL, OUT_CSV]:
            if p.exists():
                p.unlink()

    df = pd.read_csv(QUEUE_PATH)
    print(f"[FigureTriage] Loaded queue rows: {len(df)} from {QUEUE_PATH}", flush=True)
    if limit:
        df = df.head(limit)
        print(f"[FigureTriage] Using limit={limit}. Rows to process: {len(df)}", flush=True)

    done = load_existing_done(no_resume=no_resume)

    rows = []
    if (not no_resume) and OUT_CSV.exists():
        try:
            rows = pd.read_csv(OUT_CSV).to_dict("records")
        except Exception:
            rows = []

    processed = 0
    for _, r in df.iterrows():
        doi = str(r.get("doi") or "").strip()
        title = str(r.get("title") or "").strip()
        pdf_path = str(r.get("pdf_path") or "").strip()

        key = (doi.lower(), pdf_path)
        if key in done:
            continue

        anchors = []
        anchors += parse_anchor_pages(str(r.get("where_endpoint") or ""))
        anchors += parse_anchor_pages(str(r.get("endpoint_carrier_where") or ""))
        anchors += parse_anchor_pages(str(r.get("where_franz") or ""))
        anchors += parse_anchor_pages(str(r.get("where_diffusion_cell") or ""))
        anchors = sorted(set(anchors))

        attempt = 0
        while True:
            try:
                pages0, debug_top = pick_candidate_pages(pdf_path, anchors, max_pages_to_send)

                page_map: List[Tuple[int, str]] = []
                saved_imgs: List[str] = []
                pages_1idx: List[int] = []

                for p0 in pages0:
                    img_bytes, mime = render_page_image_bytes(pdf_path, p0, dpi)
                    img_path = save_page_image(doi, p0 + 1, img_bytes)
                    saved_imgs.append(img_path)
                    pages_1idx.append(p0 + 1)
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    data_url = f"data:{mime};base64,{b64}"
                    page_map.append((p0 + 1, data_url))

                tri = triage_with_vision(doi, title, page_map)

                chosen_page = tri.get("page_number", None)
                selected_image_path = pick_selected_image_path(saved_imgs, pages_1idx, chosen_page)

                tri_out = {
                    "doi": doi,
                    "title": title,
                    "pdf_path": pdf_path,
                    "candidate_pages": ",".join([str(p) for p in pages_1idx]),
                    "saved_images": "|".join(saved_imgs),
                    "selected_image_path": selected_image_path,
                    "page_pick_debug": debug_top,
                    **tri
                }

                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(tri_out, ensure_ascii=False) + "\n")

                rows.append(tri_out)
                done.add(key)
                processed += 1
                break

            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    tri_out = {
                        "doi": doi,
                        "title": title,
                        "pdf_path": pdf_path,
                        "candidate_pages": "",
                        "saved_images": "",
                        "selected_image_path": "",
                        "page_pick_debug": "",
                        "endpoint_curve_present": "uncertain",
                        "likely_endpoint_type": "unknown",
                        "figure_id": "",
                        "page_number": None,
                        "subplot": "",
                        "digitizable": "uncertain",
                        "why_not_digitizable": f"error:{type(e).__name__}",
                        "ticks_readable": "uncertain",
                        "legend_present": "uncertain",
                        "approx_curves_count": None,
                        "legend": [],
                        "suggests_table_exists": "uncertain",
                        "suggests_supp_exists": "uncertain",
                        "recommended_route": "skip",
                        "plot_bbox": None,
                        "axes_x_label": "",
                        "axes_x_unit": "",
                        "x_min": None,
                        "x_max": None,
                        "axes_y_label": "",
                        "axes_y_unit": "",
                        "y_min": None,
                        "y_max": None,
                        "y_kind": "unknown",
                        "confidence": 0.0,
                        "notes": ""
                    }
                    with open(OUT_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(tri_out, ensure_ascii=False) + "\n")
                    rows.append(tri_out)
                    done.add(key)
                    processed += 1
                    break
                backoff(attempt)

        if processed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: processed {processed}", flush=True)

        time.sleep(SLEEP_BETWEEN)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("Done.")
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("Rows:", len(df_out))

    if len(df_out) > 0:
        print("\ndigitizable counts:\n", df_out["digitizable"].value_counts(dropna=False))
        print("\nrecommended_route counts:\n", df_out["recommended_route"].value_counts(dropna=False))
        print("\nendpoint_curve_present counts:\n", df_out["endpoint_curve_present"].value_counts(dropna=False))

        # calibration completeness
        cal_ok = (
            df_out["plot_bbox"].notna()
            & df_out["x_min"].notna() & df_out["x_max"].notna()
            & df_out["y_min"].notna() & df_out["y_max"].notna()
        )
        print("\ncalibration_complete (bbox + x/y ranges) counts:\n", cal_ok.value_counts(dropna=False))
        if "digitizable" in df_out.columns:
            dig_yes = df_out["digitizable"] == "yes"
            if dig_yes.any():
                print("Among digitizable=yes, calibration_complete:", int((cal_ok & dig_yes).sum()), "/", int(dig_yes.sum()))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--reset", action="store_true", help="Delete existing outputs and rerun from scratch")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES_TO_SEND_DEFAULT)
    ap.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    args = ap.parse_args()

    main(
        limit=args.limit,
        no_resume=args.no_resume,
        reset=args.reset,
        max_pages_to_send=args.max_pages,
        dpi=args.dpi
    )
