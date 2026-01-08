# scripts/step3_stage2_openai.py
import os, time, random
import pandas as pd
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List

IN_PATH = "GenAI/outputs/screen_stage1_pass.csv"
OUT_DIR = Path("GenAI/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"  # 支持 Structured Outputs 的模型之一 :contentReference[oaicite:6]{index=6}

SYSTEM = (
    "You are a strict scientific literature triage assistant for building a dataset "
    "(PermeationNet) on ibuprofen in vitro permeation/release using diffusion cells (Franz included). "
    "Use ONLY the provided title and abstract. If unknown, set 'uncertain'."
)

class TriageResult(BaseModel):
    queue: Literal["now", "later", "park"] = Field(..., description="now=download fulltext soon; later=maybe; park=likely irrelevant")
    mentions_franz_in_abstract: Literal["yes", "no"]  # only whether explicitly mentioned
    likely_study_type: Literal["IVPT", "IVRT", "both", "uncertain"]
    likely_has_measurable_endpoint: Literal["yes", "no", "uncertain"]
    likely_has_formulation_info: Literal["yes", "no", "uncertain"]
    likely_barrier_category: Literal["skin", "synthetic_membrane", "both", "uncertain"]
    what_to_check_in_fulltext: List[str] = Field(..., min_length=1, max_length=5)
    rationale: str = Field(..., description="1–2 sentences, no citations")
    confidence: float = Field(..., ge=0.0, le=1.0)

client = OpenAI()  # reads OPENAI_API_KEY from env :contentReference[oaicite:7]{index=7}

def backoff_sleep(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.6) + random.random() * 0.4
    time.sleep(t)

def truncate(text: str, max_chars: int = 2000) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"

def triage_one(title: str, abstract: str) -> TriageResult:
    abstract = truncate(abstract, 2000)
    resp = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}"},
        ],
        text_format=TriageResult,  # Structured Outputs via Pydantic :contentReference[oaicite:8]{index=8}
    )
    return resp.output_parsed

def main(limit: int | None = None):
    df = pd.read_csv(IN_PATH)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    if limit:
        df = df.head(limit)

    out_path = OUT_DIR / "screen_stage2_ranked.csv"

    out_rows = []
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        out_rows = old.to_dict("records")
        for _, r in old.iterrows():
            done.add((str(r.get("doi","")).lower().strip(), str(r.get("title","")).strip()))

    for i, row in df.iterrows():
        key = (str(row.get("doi","")).lower().strip(), str(row.get("title","")).strip())
        if key in done:
            continue

        attempt = 0
        while True:
            try:
                tr = triage_one(row["title"], row["abstract"]).model_dump()
                rec = row.to_dict()
                rec.update(tr)
                out_rows.append(rec)
                done.add(key)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    rec = row.to_dict()
                    rec.update({
                        "queue": "park",
                        "mentions_franz_in_abstract": "no",
                        "likely_study_type": "uncertain",
                        "likely_has_measurable_endpoint": "uncertain",
                        "likely_has_formulation_info": "uncertain",
                        "likely_barrier_category": "uncertain",
                        "what_to_check_in_fulltext": ["API error; re-run this record."],
                        "rationale": f"error: {type(e).__name__}",
                        "confidence": 0.0
                    })
                    out_rows.append(rec)
                    done.add(key)
                    break
                backoff_sleep(attempt)

        if len(out_rows) % 25 == 0:
            pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: {len(out_rows)} records", flush=True)

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    df_fulltext = df_out[df_out["queue"].isin(["now", "later"])].copy()
    df_fulltext.to_csv(OUT_DIR / "screened_for_fulltext.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print("Ranked:", out_path)
    print("Now/Later:", len(df_fulltext))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    main(args.limit)
