import os, json, time, random, argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI
from GenAI.scripts.bin.triage_schema import TRIAGE_SCHEMA

IN_PATH = "GenAI/outputs/screen_stage1_pass.csv"
OUT_DIR = Path("GenAI/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "openai/gpt-oss-20b"  # 误判多再换 llama-3.3-70b-versatile

SYSTEM = (
    "You are a strict scientific literature triage assistant for building a dataset "
    "(PermeationNet) on ibuprofen permeation/release using Franz diffusion cells (IVPT/IVRT). "
    "Use ONLY the provided title and abstract. If unknown, output 'uncertain'. "
    "Return a JSON object that matches the provided JSON schema."
)

def get_client():
    key = os.environ.get("GROQ_API_KEY", "").strip().replace("gsk_gsk_", "gsk_")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

def backoff_sleep(attempt: int):
    t = min(20.0, (2 ** attempt) * 0.5) + random.random() * 0.3
    time.sleep(t)

def triage_one(client: OpenAI, title: str, abstract: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}"}
        ],
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "permeationnet_stage2_triage",
                "strict": False,
                "schema": TRIAGE_SCHEMA
            }
        }
    )
    return json.loads(resp.choices[0].message.content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows (for testing).")
    args = ap.parse_args()

    df = pd.read_csv(IN_PATH)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")

    if args.limit:
        df = df.head(args.limit)

    out_path = OUT_DIR / "screen_stage2_ranked.csv"
    out_rows = []
    done = set()

    if out_path.exists():
        old = pd.read_csv(out_path)
        out_rows = old.to_dict("records")
        for _, r in old.iterrows():
            done.add((str(r.get("doi","")).lower().strip(), str(r.get("title","")).strip()))

    client = get_client()

    for i, row in df.iterrows():
        key = (str(row.get("doi","")).lower().strip(), str(row.get("title","")).strip())
        if key in done:
            continue

        attempt = 0
        while True:
            try:
                res = triage_one(client, row["title"], row["abstract"])
                rec = row.to_dict()
                rec.update(res)
                out_rows.append(rec)
                done.add(key)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 6:
                    rec = row.to_dict()
                    rec.update({
                        "priority": "low",
                        "likely_franz": "uncertain",
                        "likely_study_type": "uncertain",
                        "likely_has_numeric_endpoint": "uncertain",
                        "likely_has_formulation_info": "uncertain",
                        "likely_barrier_category": "uncertain",
                        "what_to_check_in_fulltext": ["API error; re-run this record."],
                        "exclude_reason_if_low": f"error: {type(e).__name__}",
                        "confidence": 0.0
                    })
                    out_rows.append(rec)
                    done.add(key)
                    break
                backoff_sleep(attempt)

        if len(out_rows) % 25 == 0:
            pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Checkpoint saved: {len(out_rows)} records")

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    df_fulltext = df_out[df_out["priority"].isin(["high", "medium"])].copy()
    df_fulltext.to_csv(OUT_DIR / "screened_for_fulltext.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print("Ranked:", out_path)
    print("High/Medium:", len(df_fulltext))

if __name__ == "__main__":
    main()
