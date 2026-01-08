import json
import time
import pandas as pd
from pathlib import Path

IN_PATH = "outputs/screen_stage1_pass.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = """You are a strict scientific literature triage assistant for building a dataset (PermeationNet) on ibuprofen skin permeation/release using Franz diffusion cells (IVPT/IVRT).
Your job is NOT to extract numbers yet. Your job is to rank whether the full text is worth downloading and processing.

Rules:
- Use ONLY the provided title and abstract. Do NOT assume details not stated.
- If information is missing, return "uncertain" rather than guessing.
- Output MUST be valid JSON only (no markdown, no commentary).

Return this JSON object with exactly these keys:
{{
  "priority": "high" | "medium" | "low",
  "likely_franz": "yes" | "no" | "uncertain",
  "likely_study_type": "IVPT" | "IVRT" | "both" | "uncertain",
  "likely_has_numeric_endpoint": "yes" | "no" | "uncertain",
  "likely_has_formulation_info": "yes" | "no" | "uncertain",
  "likely_barrier_category": "skin" | "synthetic_membrane" | "both" | "uncertain",
  "what_to_check_in_fulltext": ["...","...","..."],
  "exclude_reason_if_low": "...",
  "confidence": 0.0
}}

Interpretation guidance:
- "high": very likely relevant to in vitro permeation/release using diffusion cells (possibly Franz), with formulation/vehicle details and numeric results (Q(t), Q_final, flux, Jss).
- "medium": potentially relevant but unclear; needs full-text confirmation (e.g., diffusion cell not specified; endpoints not explicit).
- "low": likely irrelevant (in vivo/PK only, oral dosage forms only, no in vitro permeation/release, no numeric results, not topical/transdermal context).

Now triage the following record.

TITLE:
{title}

ABSTRACT:
{abstract}
"""

def call_llm(prompt: str) -> str:
    """
    Replace this stub with your chosen LLM API call.
    Must return a string that is a JSON object ONLY.
    """
    raise NotImplementedError("Implement call_llm() with your LLM provider.")

def safe_parse_json(s: str) -> dict:
    # Robust-ish parse: strip whitespace and try json.loads
    s = s.strip()
    return json.loads(s)

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["abstract"] = df["abstract"].fillna("")
    df["title"] = df["title"].fillna("")

    results = []
    for i, row in df.iterrows():
        prompt = PROMPT_TEMPLATE.format(title=row["title"], abstract=row["abstract"])
        # --- call LLM ---
        try:
            out = call_llm(prompt)
            parsed = safe_parse_json(out)
        except Exception as e:
            parsed = {
                "priority": "low",
                "likely_franz": "uncertain",
                "likely_study_type": "uncertain",
                "likely_has_numeric_endpoint": "uncertain",
                "likely_has_formulation_info": "uncertain",
                "likely_barrier_category": "uncertain",
                "what_to_check_in_fulltext": ["JSON parse/API error; re-run this record."],
                "exclude_reason_if_low": f"error: {e}",
                "confidence": 0.0
            }

        rec = row.to_dict()
        rec.update(parsed)
        results.append(rec)

        # gentle rate limit
        time.sleep(0.2)

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(df)}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / "screen_stage2_ranked.csv", index=False, encoding="utf-8-sig")

    df_fulltext = df_out[df_out["priority"].isin(["high", "medium"])].copy()
    df_fulltext.to_csv(OUT_DIR / "screened_for_fulltext.csv", index=False, encoding="utf-8-sig")

    print("Saved:", OUT_DIR / "screen_stage2_ranked.csv")
    print("Saved:", OUT_DIR / "screened_for_fulltext.csv")
    print("High/Medium count:", len(df_fulltext))
