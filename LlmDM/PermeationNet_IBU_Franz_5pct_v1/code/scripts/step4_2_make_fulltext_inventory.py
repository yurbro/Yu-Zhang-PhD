# scripts/step4_make_fulltext_inventory.py
import pandas as pd
from pathlib import Path

IN_PATH = "GenAI/outputs/screened_for_fulltext.csv"
OUT_PATH = "GenAI/outputs/fulltext_inventory.csv"

df = pd.read_csv(IN_PATH)
for c in ["pdf_status","pdf_path","supp_status","supp_path","notes"]:
    if c not in df.columns:
        df[c] = ""

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Saved:", OUT_PATH, "rows:", len(df))
