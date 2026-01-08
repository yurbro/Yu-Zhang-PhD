import time
import requests
import pandas as pd

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# 召回优先的 query：不强制 Franz 出现在摘要/标题里
QUERY = (
    '(TITLE:"ibuprofen" OR ABSTRACT:"ibuprofen") AND '
    '(permeation OR permeat* OR diffusion OR "in vitro" OR release) AND '
    '(skin OR membrane OR topical OR transdermal OR "diffusion cell")'
)

PAGE_SIZE = 1000          # Europe PMC 单页上限常用 1000
MAX_RECORDS = 50000        # 先设个上限，pilot 足够；以后可放大

def epmc_search_all(query: str) -> pd.DataFrame:
    cursor = "*"
    rows = []
    fetched = 0

    while True:
        params = {
            "query": query,
            "format": "json",
            "pageSize": PAGE_SIZE,
            "cursorMark": cursor,
            "resultType": "core",
        }
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        hits = data.get("resultList", {}).get("result", [])
        if not hits:
            break

        for it in hits:
            rows.append({
                "source": it.get("source"),
                "id": it.get("id"),
                "pmid": it.get("pmid"),
                "doi": it.get("doi"),
                "title": it.get("title"),
                "abstract": it.get("abstractText"),
                "year": it.get("pubYear"),
                "journal": it.get("journalTitle"),
                "authors": it.get("authorString"),
                "url": it.get("doiUrl") or it.get("pmidUrl") or it.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url"),
                "query_used": query,
            })
            fetched += 1
            if fetched >= MAX_RECORDS:
                break

        if fetched >= MAX_RECORDS:
            break

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.2)  # 友好一点，别狂刷

    df = pd.DataFrame(rows)

    # 基本清洗
    df["title"] = df["title"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["abstract"] = df["abstract"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["doi"] = df["doi"].fillna("").str.lower().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # 去重：优先 DOI，其次 title+year
    df = df.sort_values(["doi", "year"], na_position="last")
    df = df.drop_duplicates(subset=["doi"], keep="first")
    df = df.drop_duplicates(subset=["title", "year"], keep="first")

    return df

if __name__ == "__main__":
    df = epmc_search_all(QUERY)
    print("Corpus size:", len(df))
    out_path = "GenAI\data\corpus_ibuprofen.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved:", out_path)
